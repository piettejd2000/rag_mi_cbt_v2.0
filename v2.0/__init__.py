"""
RAG v2.0 - Adaptive Grounding System
Enhanced RAG with intelligent routing to eliminate canonical knowledge regression
"""

__version__ = "2.0.0"
__description__ = "RAG system with adaptive grounding - eliminates forced grounding regression"

from .enhanced_therapy_rag_v2 import EnhancedTherapyRAGv2, CompatibilityWrapper, create_enhanced_rag_v2

__all__ = [
    'EnhancedTherapyRAGv2',
    'CompatibilityWrapper', 
    'create_enhanced_rag_v2'
]