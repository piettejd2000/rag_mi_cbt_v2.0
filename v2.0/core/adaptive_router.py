#!/usr/bin/env python3
"""
Adaptive Grounding Router for Phase 4.
Combines knowledge-type and relevance classification to route questions to appropriate paths.
"""

from typing import Dict, List, Tuple
from .knowledge_type_classifier import KnowledgeTypeClassifier
from .relevance_classifier import RelevanceClassifier

class AdaptiveGroundingRouter:
    """
    Routes questions to one of three paths based on knowledge type and retrieval relevance.
    
    Paths:
        A - Strong Grounding: Relevant retrieval, use it
        B - Soft Grounding: Partial relevance or mixed knowledge
        C - Knowledge-First: Irrelevant retrieval on canonical knowledge
    """
    
    def __init__(self, api_key: str = None):
        """Initialize with component classifiers."""
        self.knowledge_classifier = KnowledgeTypeClassifier()
        self.relevance_classifier = RelevanceClassifier(api_key=api_key)
        
        # Routing matrix based on the Phase 4 blueprint
        self.routing_matrix = {
            ('RELEVANT', 'CANONICAL_CLINICAL'): 'PATH_A',
            ('RELEVANT', 'CORPUS_SPECIFIC'): 'PATH_A', 
            ('RELEVANT', 'MIXED'): 'PATH_A',
            ('RELEVANT', 'UNKNOWN'): 'PATH_A',
            
            ('PARTIALLY_RELEVANT', 'CANONICAL_CLINICAL'): 'PATH_B',
            ('PARTIALLY_RELEVANT', 'CORPUS_SPECIFIC'): 'PATH_B',
            ('PARTIALLY_RELEVANT', 'MIXED'): 'PATH_B',
            ('PARTIALLY_RELEVANT', 'UNKNOWN'): 'PATH_B',
            
            ('IRRELEVANT', 'CANONICAL_CLINICAL'): 'PATH_C',
            ('IRRELEVANT', 'CORPUS_SPECIFIC'): 'PATH_C',
            ('IRRELEVANT', 'UNKNOWN'): 'PATH_C',
            ('IRRELEVANT', 'MIXED'): 'PATH_B',  # Mixed gets soft grounding
        }
    
    def route(self, question: str, retrieved_chunks: List[Dict] = None,
              use_mock_relevance: bool = False) -> Dict[str, any]:
        """
        Route a question to the appropriate response path.
        
        Args:
            question: The user's question
            retrieved_chunks: Retrieved context chunks (if available)
            use_mock_relevance: Use mock relevance classifier (for testing)
        
        Returns:
            dict with:
                - path: PATH_A, PATH_B, or PATH_C
                - knowledge_type: Classification result
                - relevance: Relevance assessment (if chunks provided)
                - confidence: Overall routing confidence
                - explanation: Reasoning for the routing decision
                - prompt_template: Suggested prompt approach
        """
        
        # Classify knowledge type
        knowledge_result = self.knowledge_classifier.classify(question)
        
        # If no chunks provided, make decision based on knowledge type alone
        if not retrieved_chunks:
            if knowledge_result['knowledge_type'] == 'CANONICAL_CLINICAL':
                path = 'PATH_C'
                explanation = "Canonical knowledge question, no retrieval provided"
            elif knowledge_result['knowledge_type'] == 'CORPUS_SPECIFIC':
                path = 'PATH_A'
                explanation = "Corpus-specific question, assuming retrieval will be done"
            else:
                path = 'PATH_B'
                explanation = "Mixed or unknown question type"
            
            return {
                'path': path,
                'knowledge_type': knowledge_result,
                'relevance': None,
                'confidence': knowledge_result['confidence'],
                'explanation': explanation,
                'prompt_template': self._get_prompt_template(path)
            }
        
        # Classify relevance if chunks provided
        relevance_result = self.relevance_classifier.classify(
            question, retrieved_chunks, use_mock=use_mock_relevance
        )
        
        # Determine path using routing matrix
        key = (relevance_result['relevance'], knowledge_result['knowledge_type'])
        path = self.routing_matrix.get(key, 'PATH_B')  # Default to soft grounding
        
        # Calculate combined confidence
        combined_confidence = (knowledge_result['confidence'] + relevance_result['confidence']) / 2
        
        # Generate explanation
        if path == 'PATH_A':
            explanation = f"Strong grounding: {relevance_result['relevance']} retrieval for {knowledge_result['knowledge_type']} question"
        elif path == 'PATH_B':
            explanation = f"Soft grounding: {relevance_result['relevance']} retrieval or mixed knowledge type"
        else:  # PATH_C
            explanation = f"Knowledge-first: {relevance_result['relevance']} retrieval for {knowledge_result['knowledge_type']} question"
        
        return {
            'path': path,
            'knowledge_type': knowledge_result,
            'relevance': relevance_result,
            'confidence': round(combined_confidence, 3),
            'explanation': explanation,
            'prompt_template': self._get_prompt_template(path)
        }
    
    def _get_prompt_template(self, path: str) -> str:
        """Get the appropriate prompt template for each path."""
        
        templates = {
            'PATH_A': """You are assisting a mental health clinician. 
Base your answer on the following retrieved information:

{context}

Question: {question}

Provide a comprehensive answer grounded in the retrieved information.""",
            
            'PATH_B': """You are assisting a mental health clinician.
Consider these reference materials while drawing on established clinical knowledge:

{context}

Question: {question}

Provide a comprehensive answer that synthesizes the reference materials with relevant clinical knowledge.""",
            
            'PATH_C': """You are assisting a mental health clinician.
Answer this question based on established MI/CBT clinical knowledge.

Question: {question}

If you are genuinely uncertain or the question is outside standard clinical scope, 
explicitly state this rather than speculating.

Reference materials (may not be directly relevant):
{context}"""
        }
        
        return templates.get(path, templates['PATH_B'])
    
    def batch_route(self, queries: List[Tuple[str, List[Dict]]], 
                   use_mock_relevance: bool = False) -> List[Dict]:
        """
        Route multiple queries.
        
        Args:
            queries: List of (question, chunks) tuples
            use_mock_relevance: Use mock relevance classifier
        
        Returns:
            List of routing decisions
        """
        results = []
        for question, chunks in queries:
            result = self.route(question, chunks, use_mock_relevance)
            results.append(result)
        return results


def test_router():
    """Test the adaptive grounding router."""
    router = AdaptiveGroundingRouter()
    
    test_cases = [
        {
            'question': "What does OARS stand for in MI?",
            'chunks': [
                {'text': "OARS represents Open questions, Affirmations, Reflections, Summaries"},
            ],
            'expected': 'PATH_A'
        },
        {
            'question': "What does OARS stand for in MI?", 
            'chunks': [
                {'text': "The spirit of MI involves collaboration and partnership"},
            ],
            'expected': 'PATH_C'
        },
        {
            'question': "According to the text, how is OARS used?",
            'chunks': [
                {'text': "OARS techniques are fundamental to the MI approach"},
            ],
            'expected': 'PATH_B'
        },
        {
            'question': "What medications are used for depression?",
            'chunks': [
                {'text': "CBT is an effective treatment for depression"},
            ],
            'expected': 'PATH_C'
        }
    ]
    
    print("Adaptive Grounding Router Test Results")
    print("=" * 80)
    
    for test in test_cases:
        result = router.route(test['question'], test['chunks'], use_mock_relevance=True)
        
        print(f"\nQ: {test['question'][:60]}...")
        print(f"Expected: {test['expected']}")
        print(f"Actual: {result['path']}")
        print(f"Knowledge Type: {result['knowledge_type']['knowledge_type']}")
        if result['relevance']:
            print(f"Relevance: {result['relevance']['relevance']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Explanation: {result['explanation']}")
        
        if result['path'] == test['expected']:
            print("✓ Correct routing")
        else:
            print("✗ Incorrect routing")
    
    print("\n" + "=" * 80)
    print("Prompt Template Examples:")
    print("-" * 80)
    
    for path in ['PATH_A', 'PATH_B', 'PATH_C']:
        print(f"\n{path}:")
        template = router._get_prompt_template(path)
        print(template[:200] + "...")


if __name__ == "__main__":
    test_router()