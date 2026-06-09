#!/usr/bin/env python3
"""
Enhanced RAG v2.0 with Adaptive Grounding System
Eliminates forced grounding regression while preserving all benefits
"""

import sys
import os
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
rag_root = current_dir.parent
sys.path.append(str(rag_root / 'chunking_pys'))
sys.path.append(str(current_dir))

from therapy_rag import TherapyRAG, GenerationConfig
from core.query_intent_detector import QueryIntentDetector
from core.adaptive_router import AdaptiveGroundingRouter
from typing import Dict, List, Optional, Union
import logging
from collections import defaultdict
import json
import time

logger = logging.getLogger(__name__)


class EnhancedTherapyRAGv2(TherapyRAG):
    """
    Enhanced RAG v2.0 with 3-path adaptive grounding system.
    
    Routing Logic:
    - PATH_A (Strong): Relevant retrieval → mandate using it  
    - PATH_B (Soft): Partial relevance → blend with knowledge
    - PATH_C (Knowledge): Irrelevant retrieval → use clinical knowledge
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced RAG v2.0 with adaptive grounding."""
        super().__init__(*args, **kwargs)
        
        # Initialize v1.0 components
        self.intent_detector = QueryIntentDetector()
        
        # Initialize v2.0 adaptive grounding system
        api_key = getattr(self, 'api_key', None) or kwargs.get('api_key')
        self.adaptive_router = AdaptiveGroundingRouter(api_key=api_key)
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        # Metrics tracking
        self.routing_metrics = defaultdict(int)
        self.performance_metrics = {
            'total_queries': 0,
            'routing_confidence': [],
            'response_times': [],
            'path_distribution': defaultdict(int)
        }
        
        logger.info("Enhanced RAG v2.0 initialized with adaptive grounding")
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        templates = {}
        prompt_dir = Path(__file__).parent / 'prompts'
        
        for path_file in ['path_a_prompt.txt', 'path_b_prompt.txt', 'path_c_prompt.txt']:
            path_name = path_file.split('_')[1]  # Extract 'a', 'b', 'c'
            try:
                with open(prompt_dir / path_file, 'r') as f:
                    templates[f'PATH_{path_name.upper()}'] = f.read().strip()
            except FileNotFoundError:
                logger.warning(f"Prompt template {path_file} not found, using fallback")
                templates[f'PATH_{path_name.upper()}'] = self._get_fallback_template(path_name.upper())
        
        return templates
    
    def _get_fallback_template(self, path: str) -> str:
        """Fallback prompt templates if files not found."""
        fallbacks = {
            'A': """You are assisting a mental health clinician. 
Base your answer on the following retrieved information:

{context}

Question: {question}

Provide a comprehensive answer grounded in the retrieved information.""",
            
            'B': """You are assisting a mental health clinician.
Consider these reference materials while drawing on established clinical knowledge:

{context}

Question: {question}

Provide a comprehensive answer that synthesizes the reference materials with relevant clinical knowledge.""",
            
            'C': """You are assisting a mental health clinician. 

Question: {question}

Please answer based on established MI/CBT clinical knowledge. If this question is outside the scope of standard MI/CBT practice or you are genuinely uncertain, clearly state this limitation rather than speculating.

Note: Retrieved materials were not directly relevant to this question."""
        }
        return fallbacks.get(path, fallbacks['B'])
    
    def smart_retrieve_context(self, 
                             query: str, 
                             n_results: int = 5,
                             use_intent_detection: bool = True,
                             fallback_search: bool = True,
                             content_type_filter: str = None) -> Dict:
        """
        Enhanced context retrieval with v1.0 compatibility.
        Maintains the same interface for backward compatibility.
        """
        # Use v1.0 logic for context retrieval
        results = {
            'query': query,
            'intent_detection': {},
            'primary_results': {},
            'fallback_results': {},
            'combined_context': []
        }
        
        # Handle content type filter override
        if content_type_filter and content_type_filter not in ['Auto (use intent detection)', 'All types']:
            use_intent_detection = False
            primary_content_types = [content_type_filter]
            results['intent_detection'] = {
                'primary_intent': 'manual_override',
                'confidence': 1.0,
                'content_types': [content_type_filter],
                'reasoning': f'Manual override: searching only {content_type_filter} content'
            }
        elif use_intent_detection:
            strategy = self.intent_detector.get_enhanced_query_strategy(query)
            results['intent_detection'] = strategy['intent_detection']
            primary_content_types = strategy['primary_search']['content_types']
        else:
            primary_content_types = None
        
        # Perform retrieval (using v1.0 logic)
        if primary_content_types:
            primary_documents = []
            primary_metadatas = []
            primary_scores = []
            
            for content_type in primary_content_types:
                try:
                    context = self.retrieve_context(
                        query, 
                        n_results=max(1, n_results // len(primary_content_types)),
                        content_type_filter=content_type
                    )
                    
                    if context['documents']:
                        primary_documents.extend(context['documents'])
                        primary_metadatas.extend(context['metadatas'])
                        if 'distances' in context:
                            primary_scores.extend(context['distances'])
                
                except Exception as e:
                    logger.warning(f"Error searching {content_type}: {e}")
            
            results['primary_results'] = {
                'documents': primary_documents,
                'metadatas': primary_metadatas,
                'content_types_searched': primary_content_types,
                'total_found': len(primary_documents)
            }
            
            if primary_scores:
                results['primary_results']['distances'] = primary_scores
        
        # Fallback search if needed
        if fallback_search and (not results['primary_results'].get('documents') or len(results['primary_results']['documents']) < n_results):
            try:
                fallback_context = self.retrieve_context(query, n_results=n_results)
                results['fallback_results'] = fallback_context
            except Exception as e:
                logger.warning(f"Fallback search failed: {e}")
        
        # Combine results
        combined = []
        if results['primary_results'].get('documents'):
            for i, doc in enumerate(results['primary_results']['documents']):
                combined.append({
                    'text': doc,
                    'metadata': results['primary_results']['metadatas'][i] if results['primary_results'].get('metadatas') else {},
                    'source': 'primary'
                })
        
        if results['fallback_results'].get('documents'):
            existing_docs = {item['text'] for item in combined}
            for i, doc in enumerate(results['fallback_results']['documents']):
                if doc not in existing_docs:  # Avoid duplicates
                    combined.append({
                        'text': doc,
                        'metadata': results['fallback_results']['metadatas'][i] if results['fallback_results'].get('metadatas') else {},
                        'source': 'fallback'
                    })
        
        results['combined_context'] = combined
        return results
    
    def generate_response_v2(self, 
                           query: str,
                           context: List[Dict] = None,
                           config: GenerationConfig = None) -> Dict:
        """
        Generate response using v2.0 adaptive grounding system.
        
        Returns:
            dict with 'response', 'routing_decision', and 'metadata'
        """
        start_time = time.time()
        
        # Default config
        if config is None:
            config = GenerationConfig()
        
        # Get retrieved chunks for routing decision
        if context is None:
            retrieval_result = self.smart_retrieve_context(query)
            retrieved_chunks = retrieval_result['combined_context']
        else:
            retrieved_chunks = context
        
        # Route the query using adaptive grounding system
        routing_decision = self.adaptive_router.route(
            question=query,
            retrieved_chunks=retrieved_chunks,
            use_mock_relevance=False  # Use real classifier
        )
        
        # Select prompt template based on routing decision
        selected_path = routing_decision['path']
        prompt_template = self.prompt_templates[selected_path]
        
        # Format context for prompt
        if retrieved_chunks:
            context_text = "\n\n".join([
                f"Source: {chunk.get('metadata', {}).get('source_file', 'Unknown')}\n{chunk['text']}"
                for chunk in retrieved_chunks[:5]  # Limit to top 5
            ])
        else:
            context_text = "No relevant context retrieved."
        
        # Build final prompt
        if selected_path == 'PATH_C':
            # PATH_C doesn't use context in template (knowledge-first)
            final_prompt = prompt_template.format(question=query)
        else:
            # PATH_A and PATH_B use context
            final_prompt = prompt_template.format(
                question=query,
                context=context_text
            )
        
        # Generate response using LLM
        try:
            response = self._call_anthropic_api(final_prompt, config)
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            response = f"Error generating response: {e}"
        
        # Update metrics
        self.routing_metrics[selected_path] += 1
        self.performance_metrics['total_queries'] += 1
        self.performance_metrics['routing_confidence'].append(routing_decision['confidence'])
        self.performance_metrics['response_times'].append(time.time() - start_time)
        self.performance_metrics['path_distribution'][selected_path] += 1
        
        return {
            'response': response,
            'routing_decision': routing_decision,
            'metadata': {
                'path_used': selected_path,
                'prompt_template': selected_path,
                'response_time': time.time() - start_time,
                'context_chunks_used': len(retrieved_chunks) if retrieved_chunks else 0,
                'confidence': routing_decision['confidence']
            }
        }
    
    def generate_response_v1_compatible(self,
                                       query: str,
                                       context: List[Dict] = None,
                                       config: GenerationConfig = None) -> str:
        """
        v1.0 compatible response generation.
        Returns just the response text for backward compatibility.
        """
        result = self.generate_response_v2(query, context, config)
        return result['response']
    
    def generate_response(self, *args, **kwargs):
        """
        Main response generation method.
        Routes to v2.0 by default, with v1.0 compatibility option.
        """
        # Check if v1 compatibility is requested
        if kwargs.pop('v1_compatible', False):
            return self.generate_response_v1_compatible(*args, **kwargs)
        else:
            return self.generate_response_v2(*args, **kwargs)
    
    def force_path_for_testing(self, query: str, path: str, context: List[Dict] = None) -> Dict:
        """
        Force a specific path for testing purposes.
        
        Args:
            query: User question
            path: 'PATH_A', 'PATH_B', or 'PATH_C'
            context: Retrieved context (optional)
        """
        if path not in self.prompt_templates:
            raise ValueError(f"Invalid path: {path}. Must be PATH_A, PATH_B, or PATH_C")
        
        # Mock routing decision
        mock_routing = {
            'path': path,
            'confidence': 1.0,
            'explanation': f"Forced to {path} for testing",
            'knowledge_type': {'knowledge_type': 'TEST'},
            'relevance': {'relevance': 'TEST'}
        }
        
        # Generate response
        prompt_template = self.prompt_templates[path]
        
        if context:
            context_text = "\n\n".join([chunk['text'] for chunk in context[:5]])
        else:
            context_text = "No context provided for test."
        
        if path == 'PATH_C':
            final_prompt = prompt_template.format(question=query)
        else:
            final_prompt = prompt_template.format(question=query, context=context_text)
        
        try:
            response = self._call_anthropic_api(final_prompt, GenerationConfig())
        except Exception as e:
            response = f"Error in forced path test: {e}"
        
        return {
            'response': response,
            'routing_decision': mock_routing,
            'metadata': {
                'path_used': path,
                'test_mode': True
            }
        }
    
    def get_routing_statistics(self) -> Dict:
        """Get routing and performance statistics."""
        total = self.performance_metrics['total_queries']
        if total == 0:
            return {"message": "No queries processed yet"}
        
        return {
            'total_queries': total,
            'path_distribution': dict(self.performance_metrics['path_distribution']),
            'path_percentages': {
                path: (count / total) * 100 
                for path, count in self.performance_metrics['path_distribution'].items()
            },
            'average_confidence': sum(self.performance_metrics['routing_confidence']) / len(self.performance_metrics['routing_confidence']),
            'average_response_time': sum(self.performance_metrics['response_times']) / len(self.performance_metrics['response_times']),
            'routing_metrics': dict(self.routing_metrics)
        }
    
    def reset_metrics(self):
        """Reset all tracking metrics."""
        self.routing_metrics = defaultdict(int)
        self.performance_metrics = {
            'total_queries': 0,
            'routing_confidence': [],
            'response_times': [],
            'path_distribution': defaultdict(int)
        }
        logger.info("v2.0 metrics reset")


class CompatibilityWrapper:
    """
    Wrapper to ensure v1.0 API calls work seamlessly with v2.0 system.
    """
    
    def __init__(self, v2_system: EnhancedTherapyRAGv2):
        self.v2_system = v2_system
        self.force_v1_mode = False
    
    def set_v1_mode(self, enabled: bool):
        """Force v1.0 behavior (PATH_A only) for compatibility testing."""
        self.force_v1_mode = enabled
    
    def retrieve_and_generate(self, query: str, **kwargs) -> str:
        """v1.0 compatible method signature."""
        if self.force_v1_mode:
            # Force PATH_A (strong grounding) for all queries
            result = self.v2_system.force_path_for_testing(query, 'PATH_A')
            return result['response']
        else:
            # Use v2.0 adaptive routing
            return self.v2_system.generate_response_v1_compatible(query, **kwargs)


# Factory function for easy instantiation
def create_enhanced_rag_v2(**kwargs) -> EnhancedTherapyRAGv2:
    """
    Factory function to create EnhancedTherapyRAGv2 instance.
    """
    return EnhancedTherapyRAGv2(**kwargs)