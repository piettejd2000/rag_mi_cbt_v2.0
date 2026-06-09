#!/usr/bin/env python3
"""
Relevance Classifier for Phase 4 Adaptive Grounding.
Classifies retrieval relevance as RELEVANT, PARTIALLY_RELEVANT, or IRRELEVANT.
Uses Claude Haiku for high-accuracy relevance detection.
"""

import os
import json
from typing import Dict, List, Tuple
from anthropic import Anthropic

class RelevanceClassifier:
    """
    Classifies whether retrieved chunks are relevant to answering a question.
    Uses a lightweight LLM (Haiku) for accurate assessment.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize with Anthropic API key."""
        if api_key is None:
            # Try to load from secrets file
            secrets_file = "/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/.streamlit/secrets.toml"
            if os.path.exists(secrets_file):
                with open(secrets_file, 'r') as f:
                    for line in f:
                        if 'ANTHROPIC_API_KEY' in line:
                            api_key = line.split('=')[1].strip().strip('"')
                            break
        
        if api_key:
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-5-haiku-20241022"  # Updated model ID
        else:
            self.client = None
            print("Warning: No API key provided. Using mock classifier.")
    
    def classify(self, question: str, chunks: List[Dict[str, str]], 
                 use_mock: bool = False) -> Dict[str, any]:
        """
        Classify relevance of retrieved chunks to a question.
        
        Args:
            question: The user's question
            chunks: List of retrieved chunks with 'text' field
            use_mock: Use rule-based mock instead of API (for testing)
        
        Returns:
            dict with:
                - relevance: RELEVANT, PARTIALLY_RELEVANT, or IRRELEVANT
                - confidence: float 0-1
                - explanation: brief reasoning
                - relevant_chunks: list of chunk indices that are relevant
        """
        
        if use_mock or self.client is None:
            return self._mock_classify(question, chunks)
        
        # Prepare chunks text
        chunks_text = "\n\n".join([
            f"[Chunk {i+1}]\n{chunk.get('text', chunk.get('content', ''))[:500]}"
            for i, chunk in enumerate(chunks[:5])  # Top 5 chunks
        ])
        
        # Create prompt
        prompt = f"""Assess whether these retrieved text chunks contain information that would help answer the given question.

QUESTION: {question}

RETRIEVED CHUNKS:
{chunks_text}

Classify the relevance as:
- RELEVANT: The chunks directly answer the question or provide the key information needed
- PARTIALLY_RELEVANT: The chunks contain some helpful information but don't fully answer the question
- IRRELEVANT: The chunks don't help answer the question at all

Respond in JSON format:
{{
    "relevance": "RELEVANT|PARTIALLY_RELEVANT|IRRELEVANT",
    "confidence": 0.0-1.0,
    "explanation": "brief explanation",
    "relevant_chunks": [list of chunk numbers that are relevant, e.g., [1, 3]]
}}"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            response_text = response.content[0].text.strip()
            
            # Try to extract JSON
            if '{' in response_text and '}' in response_text:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                # Ensure all required fields
                return {
                    'relevance': result.get('relevance', 'IRRELEVANT'),
                    'confidence': float(result.get('confidence', 0.5)),
                    'explanation': result.get('explanation', ''),
                    'relevant_chunks': result.get('relevant_chunks', [])
                }
            else:
                # Fallback parsing
                relevance = 'IRRELEVANT'
                if 'RELEVANT' in response_text and 'PARTIALLY' not in response_text:
                    relevance = 'RELEVANT'
                elif 'PARTIALLY' in response_text:
                    relevance = 'PARTIALLY_RELEVANT'
                
                return {
                    'relevance': relevance,
                    'confidence': 0.7,
                    'explanation': 'Parsed from response',
                    'relevant_chunks': []
                }
                
        except Exception as e:
            print(f"API call failed: {e}")
            return self._mock_classify(question, chunks)
    
    def _mock_classify(self, question: str, chunks: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Mock classifier using simple rules (for testing without API).
        """
        question_lower = question.lower()
        
        # Check if any chunk contains key terms from question
        relevant_chunks = []
        relevance_score = 0
        
        # Extract key terms from question
        key_terms = []
        if 'oars' in question_lower:
            key_terms.append('oars')
        if 'reflective listening' in question_lower:
            key_terms.extend(['reflective', 'listening'])
        if 'cognitive triad' in question_lower:
            key_terms.extend(['cognitive', 'triad', 'beck'])
        if 'automatic thought' in question_lower:
            key_terms.extend(['automatic', 'thought'])
        
        # If no specific terms, extract nouns
        if not key_terms:
            import re
            # Simple noun extraction (words that might be important)
            words = re.findall(r'\b[a-z]{4,}\b', question_lower)
            key_terms = [w for w in words if w not in 
                        ['what', 'which', 'where', 'when', 'does', 'that', 'this']][:3]
        
        # Check chunks
        for i, chunk in enumerate(chunks[:5]):
            chunk_text = chunk.get('text', chunk.get('content', '')).lower()
            chunk_score = 0
            
            for term in key_terms:
                if term in chunk_text:
                    chunk_score += 1
            
            if chunk_score > 0:
                relevant_chunks.append(i + 1)
                relevance_score += chunk_score
        
        # Determine relevance level
        if relevance_score >= len(key_terms) * 0.8:
            relevance = 'RELEVANT'
            confidence = 0.8
        elif relevance_score >= len(key_terms) * 0.4:
            relevance = 'PARTIALLY_RELEVANT'
            confidence = 0.6
        else:
            relevance = 'IRRELEVANT'
            confidence = 0.7
        
        return {
            'relevance': relevance,
            'confidence': confidence,
            'explanation': f"Mock classification based on {len(key_terms)} key terms",
            'relevant_chunks': relevant_chunks
        }
    
    def classify_batch(self, queries: List[Tuple[str, List[Dict]]], 
                      use_mock: bool = False) -> List[Dict]:
        """
        Classify relevance for multiple query-chunks pairs.
        
        Args:
            queries: List of (question, chunks) tuples
            use_mock: Use mock classifier instead of API
        
        Returns:
            List of classification results
        """
        results = []
        for question, chunks in queries:
            result = self.classify(question, chunks, use_mock=use_mock)
            results.append(result)
        return results


def test_classifier():
    """Test the relevance classifier."""
    classifier = RelevanceClassifier()
    
    # Test cases
    test_cases = [
        {
            'question': "What does OARS stand for in Motivational Interviewing?",
            'chunks': [
                {'text': "OARS stands for Open questions, Affirmations, Reflections, and Summaries."},
                {'text': "The spirit of MI involves collaboration and evocation."},
            ]
        },
        {
            'question': "What is reflective listening?",
            'chunks': [
                {'text': "Behavioral activation involves scheduling pleasant activities."},
                {'text': "Cognitive restructuring helps identify distorted thoughts."},
            ]
        },
        {
            'question': "What is Beck's cognitive triad?",
            'chunks': [
                {'text': "The cognitive triad involves negative views about self, world, and future."},
                {'text': "Depression symptoms include low mood and anhedonia."},
            ]
        }
    ]
    
    print("Relevance Classifier Test Results")
    print("=" * 70)
    
    for test in test_cases:
        result = classifier.classify(test['question'], test['chunks'], use_mock=True)
        print(f"\nQ: {test['question'][:60]}...")
        print(f"Chunks provided: {len(test['chunks'])}")
        print(f"Result:")
        print(f"  Relevance: {result['relevance']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Relevant chunks: {result['relevant_chunks']}")
        print(f"  Explanation: {result['explanation']}")


if __name__ == "__main__":
    test_classifier()