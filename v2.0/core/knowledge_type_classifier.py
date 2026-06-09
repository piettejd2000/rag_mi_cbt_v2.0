#!/usr/bin/env python3
"""
Knowledge-Type Classifier for Phase 4 Adaptive Grounding.
Classifies questions as CANONICAL_CLINICAL, CORPUS_SPECIFIC, MIXED, or UNKNOWN.
"""

import re
from typing import Dict, Tuple, List

class KnowledgeTypeClassifier:
    """
    Classifies questions based on whether they ask about canonical clinical knowledge
    (standard MI/CBT concepts) versus corpus-specific information.
    """
    
    def __init__(self):
        """Initialize with comprehensive pattern dictionaries."""
        
        # Canonical MI concepts
        self.mi_canonical_patterns = {
            'acronyms': [
                r'\bOARS\b', r'\bRULE\b', r'\bDARNA?\b', r'\bCATS\b',
                r'\bPARR\b',  # Spanish version of OARS
                r'siglas', r'acronym', r'acrónimo'
            ],
            'core_concepts': [
                r'reflective listening', r'escucha reflexiva',
                r'open.{0,10}closed.{0,10}question', r'pregunta.{0,10}abierta.{0,10}cerrada',
                r'four processes', r'cuatro procesos',
                r'spirit of MI', r'espíritu de.{0,10}entrevista motivacional',
                r'ambivalence', r'ambivalencia',
                r'change talk', r'discurso de cambio',
                r'sustain talk', r'discurso de mantenimiento',
                r'rolling with resistance', r'rodar con la resistencia',
                r'developing discrepancy', r'desarrollar discrepancia',
                r'self.efficacy', r'autoeficacia'
            ],
            'techniques': [
                r'affirmation', r'afirmación',
                r'summariz', r'resumir', r'resumen',
                r'evoking', r'evocar',
                r'focusing', r'enfocar',
                r'planning', r'planificar',
                r'engaging', r'comprometer'
            ]
        }
        
        # Canonical CBT/TCC concepts
        self.cbt_canonical_patterns = {
            'core_concepts': [
                r'automatic thought', r'pensamiento.{0,10}automático',
                r'cognitive triad', r'tríada cognitiva',
                r'Beck.{0,10}triad', r'tríada de Beck',
                r'ABC model', r'modelo ABC',
                r'cognitive restructuring', r'reestructuración cognitiva',
                r'core belief', r'creencia.{0,10}nuclear',
                r'intermediate belief', r'creencia.{0,10}intermedia',
                r'cognitive distortion', r'distorsión cognitiva'
            ],
            'distortions': [
                r'catastrophizing', r'catastrofización',
                r'all.or.nothing', r'todo o nada',
                r'overgeneralization', r'sobregeneralización',
                r'mind reading', r'lectura de mente',
                r'fortune telling', r'adivinación',
                r'emotional reasoning', r'razonamiento emocional',
                r'should statement', r'declaración.{0,10}deber',
                r'magnification', r'magnificación',
                r'minimization', r'minimización'
            ],
            'techniques': [
                r'Socratic questioning', r'cuestionamiento socrático',
                r'behavioral activation', r'activación conductual',
                r'exposure therapy', r'terapia de exposición',
                r'thought record', r'registro de pensamiento',
                r'behavioral experiment', r'experimento conductual',
                r'graded exposure', r'exposición graduada',
                r'activity scheduling', r'programación de actividades',
                r'relaxation', r'relajación'
            ]
        }
        
        # General therapeutic concepts
        self.therapeutic_patterns = {
            'alliance': [
                r'therapeutic alliance', r'alianza terapéutica',
                r'rapport', r'working relationship', r'relación de trabajo'
            ],
            'basic_concepts': [
                r'empathy', r'empatía',
                r'unconditional positive regard', r'consideración positiva incondicional',
                r'collaboration', r'colaboración',
                r'homework', r'tarea', r'tareas para casa'
            ]
        }
        
        # Corpus-specific indicators (suggests specific source material)
        self.corpus_indicators = [
            r'according to', r'según',
            r'in the (text|book|chapter)', r'en el (texto|libro|capítulo)',
            r'author.{0,10}(say|state|mention)', r'autor.{0,10}(dice|declara|menciona)',
            r'specific.{0,10}example', r'ejemplo específico',
            r'how does the text', r'cómo.{0,10}el texto',
            r'figure \d', r'figura \d',
            r'chapter \d', r'capítulo \d'
        ]
        
    def _check_patterns(self, text: str, patterns: List[str]) -> int:
        """Count how many patterns match in the text."""
        text_lower = text.lower()
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches += 1
        return matches
    
    def _is_canonical(self, text: str) -> Tuple[bool, float]:
        """Check if question asks about canonical clinical knowledge."""
        canonical_score = 0
        
        # Check MI patterns
        for category, patterns in self.mi_canonical_patterns.items():
            matches = self._check_patterns(text, patterns)
            if category == 'acronyms' and matches > 0:
                canonical_score += 3  # Acronyms are strongly canonical
            else:
                canonical_score += matches
        
        # Check CBT patterns
        for category, patterns in self.cbt_canonical_patterns.items():
            matches = self._check_patterns(text, patterns)
            if category == 'core_concepts' and matches > 0:
                canonical_score += 2  # Core concepts weighted higher
            else:
                canonical_score += matches
        
        # Check therapeutic patterns
        for category, patterns in self.therapeutic_patterns.items():
            canonical_score += self._check_patterns(text, patterns)
        
        # Normalize score
        confidence = min(canonical_score / 5.0, 1.0)  # Cap at 1.0
        
        return canonical_score > 0, confidence
    
    def _is_corpus_specific(self, text: str) -> Tuple[bool, float]:
        """Check if question asks about corpus-specific information."""
        corpus_score = self._check_patterns(text, self.corpus_indicators)
        
        # Strong indicators
        if any(phrase in text.lower() for phrase in ['according to', 'in the text', 'the author']):
            corpus_score += 2
        
        confidence = min(corpus_score / 3.0, 1.0)
        return corpus_score > 0, confidence
    
    def classify(self, question: str) -> Dict[str, any]:
        """
        Classify a question into knowledge type.
        
        Returns:
            dict with keys:
                - knowledge_type: CANONICAL_CLINICAL, CORPUS_SPECIFIC, MIXED, or UNKNOWN
                - confidence: float 0-1
                - canonical_score: float
                - corpus_score: float
                - detected_concepts: list of matched concept categories
        """
        is_canonical, canonical_conf = self._is_canonical(question)
        is_corpus, corpus_conf = self._is_corpus_specific(question)
        
        # Detect specific concepts for reporting
        detected_concepts = []
        question_lower = question.lower()
        
        if 'oars' in question_lower or 'parr' in question_lower:
            detected_concepts.append('MI_acronym')
        if 'automatic thought' in question_lower or 'pensamiento automático' in question_lower:
            detected_concepts.append('CBT_core')
        if 'reflective listening' in question_lower or 'escucha reflexiva' in question_lower:
            detected_concepts.append('MI_technique')
        if 'according to' in question_lower or 'según' in question_lower:
            detected_concepts.append('corpus_reference')
        
        # Determine classification
        if is_canonical and is_corpus:
            knowledge_type = 'MIXED'
            confidence = min(canonical_conf, corpus_conf)
        elif is_canonical:
            knowledge_type = 'CANONICAL_CLINICAL'
            confidence = canonical_conf
        elif is_corpus:
            knowledge_type = 'CORPUS_SPECIFIC'
            confidence = corpus_conf
        else:
            # Check if it's still a clinical question but not canonical
            clinical_keywords = ['patient', 'paciente', 'client', 'cliente', 
                               'treatment', 'tratamiento', 'therapy', 'terapia']
            if self._check_patterns(question, clinical_keywords) > 0:
                knowledge_type = 'UNKNOWN'
                confidence = 0.3
            else:
                knowledge_type = 'UNKNOWN'
                confidence = 0.1
        
        return {
            'knowledge_type': knowledge_type,
            'confidence': round(confidence, 3),
            'canonical_score': round(canonical_conf, 3),
            'corpus_score': round(corpus_conf, 3),
            'detected_concepts': detected_concepts
        }
    
    def classify_batch(self, questions: List[str]) -> List[Dict]:
        """Classify multiple questions."""
        return [self.classify(q) for q in questions]


def test_classifier():
    """Test the classifier with example questions."""
    classifier = KnowledgeTypeClassifier()
    
    test_cases = [
        # Canonical MI/CBT
        "What does OARS stand for in Motivational Interviewing?",
        "¿Qué son los pensamientos automáticos en la TCC?",
        "What is reflective listening?",
        "What is Beck's cognitive triad?",
        
        # Corpus-specific
        "According to the text, what are the three tools of communication?",
        "How does the author describe emotional reasoning?",
        "In Chapter 5, what example is given for behavioral activation?",
        
        # Mixed
        "According to the text, how is OARS used in practice?",
        
        # Unknown
        "What medications are used for depression?",
        "How long should therapy last?"
    ]
    
    print("Knowledge-Type Classifier Test Results")
    print("=" * 70)
    
    for question in test_cases:
        result = classifier.classify(question)
        print(f"\nQ: {question[:60]}...")
        print(f"   Type: {result['knowledge_type']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Canonical: {result['canonical_score']}, Corpus: {result['corpus_score']}")
        if result['detected_concepts']:
            print(f"   Concepts: {', '.join(result['detected_concepts'])}")


if __name__ == "__main__":
    test_classifier()