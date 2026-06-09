#!/usr/bin/env python3
"""
Simple Query Intent Detector for RAG v2.0
Provides basic intent detection functionality for backward compatibility
"""

from typing import Dict, List


class QueryIntentDetector:
    """
    Simple intent detector for v2.0 compatibility.
    Provides basic functionality to maintain v1.0 interface compatibility.
    """
    
    def __init__(self):
        """Initialize with basic content type mappings."""
        self.content_types = [
            'MI_techniques',
            'CBT_principles', 
            'clinical_guidelines',
            'case_studies',
            'therapeutic_interventions'
        ]
    
    def get_enhanced_query_strategy(self, query: str) -> Dict:
        """
        Enhanced query strategy for v1.0 compatibility.
        Returns a simplified strategy based on keyword detection.
        """
        query_lower = query.lower()
        
        # Simple keyword-based intent detection
        if any(keyword in query_lower for keyword in ['oars', 'motivational', 'mi', 'interviewing']):
            primary_intent = 'MI_techniques'
            primary_types = ['MI_techniques']
        elif any(keyword in query_lower for keyword in ['cbt', 'cognitive', 'behavioral', 'therapy']):
            primary_intent = 'CBT_principles'
            primary_types = ['CBT_principles']
        elif any(keyword in query_lower for keyword in ['case', 'patient', 'client']):
            primary_intent = 'case_studies'
            primary_types = ['case_studies']
        else:
            primary_intent = 'general_clinical'
            primary_types = ['clinical_guidelines', 'therapeutic_interventions']
        
        return {
            'intent_detection': {
                'primary_intent': primary_intent,
                'confidence': 0.8,
                'content_types': primary_types,
                'reasoning': f'Simple keyword-based detection for {primary_intent}'
            },
            'primary_search': {
                'content_types': primary_types
            }
        }
    
    def detect_intent(self, query: str) -> Dict:
        """Basic intent detection method."""
        strategy = self.get_enhanced_query_strategy(query)
        return strategy['intent_detection']