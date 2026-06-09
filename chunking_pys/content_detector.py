"""
Content Type Detector for Clinical Psychology Documents
Detects dialogue, procedures, case examples, and conceptual content
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    content_type: str
    confidence: float
    detection_method: str
    markers_found: List[str]


class ContentTypeDetector:
    """
    Detects content types in clinical psychology texts using pattern matching
    with Claude API fallback for uncertain cases.
    """
    
    def __init__(self):
        # Dialogue patterns for various formats - Enhanced for MI content
        self.dialogue_patterns = [
            (r'^[TP]:', 'T:/P: format'),
            (r'^(THERAPIST|TERAPEUTA|CLINICIAN|COUNSELOR|PRACTITIONER|NURSE|DOCTOR|PHYSICIAN):', 'Full clinician label'),
            (r'^(Therapist|Terapeuta|Clinician|Counselor|Practitioner|Nurse|Doctor|Physician):', 'Capitalized clinician'),
            (r'^(PATIENT|PACIENTE|CLIENT|CLIENTE):', 'Full patient label'),
            (r'^(Patient|Paciente|Client|Cliente):', 'Capitalized patient'),
            (r'^\d+\.\s*[TP]:', 'Numbered dialogue'),
            (r'^(Th|Pt|Pr|Dr|Nr):', 'Abbreviated format'),
            (r'^-\s*(Therapist|Patient|Practitioner|Clinician|T|P|Pr):', 'Dash prefix dialogue'),
        ]
        
        # Procedural indicators - Enhanced for MI content
        self.procedural_patterns = [
            (r'(?:Step|Paso)\s+\d+[:\.]', 'Numbered steps'),
            (r'^\d+\.\s+\w+', 'Numbered list'),
            (r'^(?:First|Primero|Initially|Inicialmente)[,:\s]', 'Sequence starters'),
            (r'^(?:Next|Siguiente|Then|Luego|Después)[,:\s]', 'Sequence continuation'),
            (r'^(?:Finally|Finalmente|Lastly|Por último)[,:\s]', 'Sequence endings'),
            (r'(?:Procedure|Procedimiento|Protocol|Protocolo):', 'Procedure labels'),
            (r'Session\s+(?:structure|agenda)|Estructura\s+de\s+(?:la\s+)?sesión', 'Session structure'),
            (r'Homework:|Tarea:', 'Homework assignments'),
            # MI-specific patterns
            (r'RULE|DARN|OARS', 'MI frameworks'),
            (r'(?:Four|Cuatro)\s+(?:principles|principios)', 'MI principles'),
            (r'(?:Stages?\s+of\s+change|Etapas?\s+de\s+cambio)', 'Change stages'),
            (r'(?:Motivational\s+interviewing|Entrevista\s+motivacional)', 'MI technique'),
        ]
        
        # Case example indicators - Enhanced for MI communication examples
        self.case_patterns = [
            (r'(?:Case|Caso)\s+(?:example|ejemplo|\d+)', 'Case labels'),
            (r'(?:Example|Ejemplo):\s*(?:A|Un)', 'Example introduction'),
            (r'(?:patient|cliente)\s+(?:named|llamado)', 'Patient introduction'),
            (r'(?:presenting|presented)\s+with', 'Case presentation'),
            (r'\d+[-\s]?year[-\s]?old\s+(?:male|female|man|woman)', 'Age/gender description'),
            (r'años\s+de\s+edad', 'Spanish age indicator'),
            # MI-specific communication examples
            (r'(?:Setting|Context):\s+(?:Trauma|Emergency|Clinic)', 'MI setting description'),
            (r'(?:Challenge|Goal):\s+To\s+encourage', 'MI challenge statement'),
            (r'(?:Communication\s+example|Ejemplo\s+de\s+comunicación)', 'Communication example'),
        ]
        
        # New content type patterns for MI communication techniques
        self.mi_communication_patterns = [
            (r'PRACTITIONER:\s+.*PATIENT:\s+.*PRACTITIONER:', 'MI dialogue exchange'),
            (r'(?:Asking|Listening|Informing)\s+and\s+(?:Asking|Listening|Informing)', 'MI skill combinations'),
            (r'(?:Change\s+talk|Sustain\s+talk)', 'MI change language'),
            (r'(?:Reflective\s+listening|Empathic\s+response)', 'MI listening techniques'),
            (r'(?:Open\s+question|Closed\s+question)', 'MI questioning techniques'),
        ]
        
        # Conceptual/theoretical indicators
        self.conceptual_patterns = [
            (r'(?:defined?|definition)\s+(?:as|of)', 'Definition markers'),
            (r'(?:theory|teoría)\s+(?:of|de)', 'Theory markers'),
            (r'(?:research|investigación)\s+(?:shows|muestra)', 'Research references'),
            (r'(?:according\s+to|según)', 'Citation markers'),
            (r'(?:consists?\s+of|consiste\s+en)', 'Explanatory language'),
        ]
    
    def detect(self, text: str, section_title: str = "") -> DetectionResult:
        """
        Detect content type with confidence scoring.
        
        Args:
            text: The text to analyze
            section_title: Optional section title for context
            
        Returns:
            DetectionResult with type, confidence, and method
        """
        # Normalize text for analysis
        lines = text.strip().split('\n')
        first_10_lines = '\n'.join(lines[:min(10, len(lines))])
        
        # Check each content type
        dialogue_score, dialogue_markers = self._score_dialogue(lines)
        procedural_score, procedural_markers = self._score_procedural(text, section_title)
        case_score, case_markers = self._score_case_example(text)
        mi_comm_score, mi_comm_markers = self._score_mi_communication(text)
        conceptual_score, conceptual_markers = self._score_conceptual(text)
        
        # Determine best match
        scores = {
            'dialogue': (dialogue_score, dialogue_markers),
            'procedure': (procedural_score, procedural_markers),
            'case_example': (case_score, case_markers),
            'mi_communication': (mi_comm_score, mi_comm_markers),
            'conceptual': (conceptual_score, conceptual_markers)
        }
        
        best_type = max(scores.keys(), key=lambda k: scores[k][0])
        best_score, best_markers = scores[best_type]
        
        # If confidence is low, mark for Claude API
        if best_score < 0.6:
            return DetectionResult(
                content_type='uncertain',
                confidence=best_score,
                detection_method='pattern_low_confidence',
                markers_found=best_markers
            )
        
        return DetectionResult(
            content_type=best_type,
            confidence=best_score,
            detection_method='pattern',
            markers_found=best_markers
        )
    
    def _score_dialogue(self, lines: List[str]) -> Tuple[float, List[str]]:
        """Score text for dialogue content."""
        dialogue_lines = 0
        markers_found = []
        
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            line_stripped = line.strip()
            for pattern, marker_name in self.dialogue_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    dialogue_lines += 1
                    if marker_name not in markers_found:
                        markers_found.append(marker_name)
                    break
        
        # Calculate score based on dialogue density
        if dialogue_lines >= 5:
            confidence = min(0.95, 0.5 + (dialogue_lines * 0.05))
        elif dialogue_lines >= 3:
            confidence = 0.7
        elif dialogue_lines >= 1:
            confidence = 0.4
        else:
            confidence = 0.0
        
        return confidence, markers_found
    
    def _score_procedural(self, text: str, section_title: str) -> Tuple[float, List[str]]:
        """Score text for procedural content."""
        markers_found = []
        pattern_matches = 0
        
        # Check section title first
        if section_title:
            title_lower = section_title.lower()
            if any(term in title_lower for term in ['procedure', 'protocol', 'steps', 'estructura', 'sesión']):
                markers_found.append('Section title indicator')
                pattern_matches += 2
        
        # Check patterns
        for pattern, marker_name in self.procedural_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                pattern_matches += 1
                if marker_name not in markers_found:
                    markers_found.append(marker_name)
        
        # Score based on pattern density
        if pattern_matches >= 4:
            confidence = 0.9
        elif pattern_matches >= 2:
            confidence = 0.7
        elif pattern_matches >= 1:
            confidence = 0.4
        else:
            confidence = 0.1
        
        return confidence, markers_found
    
    def _score_case_example(self, text: str) -> Tuple[float, List[str]]:
        """Score text for case example content."""
        markers_found = []
        pattern_matches = 0
        
        for pattern, marker_name in self.case_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_matches += 1
                if marker_name not in markers_found:
                    markers_found.append(marker_name)
        
        # Case examples often have dialogue too
        lines = text.split('\n')
        dialogue_score, _ = self._score_dialogue(lines)
        
        if pattern_matches >= 2:
            confidence = 0.85
        elif pattern_matches >= 1 and dialogue_score > 0.3:
            confidence = 0.7
        elif pattern_matches >= 1:
            confidence = 0.5
        else:
            confidence = 0.1
        
        return confidence, markers_found
    
    def _score_mi_communication(self, text: str) -> Tuple[float, List[str]]:
        """Score text for MI communication techniques content."""
        markers_found = []
        pattern_matches = 0
        
        for pattern, marker_name in self.mi_communication_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                pattern_matches += 1
                if marker_name not in markers_found:
                    markers_found.append(marker_name)
        
        # Check for multiple PRACTITIONER/PATIENT exchanges (strong indicator)
        practitioner_count = len(re.findall(r'PRACTITIONER:', text, re.IGNORECASE))
        patient_count = len(re.findall(r'PATIENT:', text, re.IGNORECASE))
        
        if practitioner_count >= 2 and patient_count >= 2:
            pattern_matches += 2
            markers_found.append('Multiple practitioner-patient exchanges')
        
        # Score based on pattern density and exchange count
        if pattern_matches >= 3:
            confidence = 0.9
        elif pattern_matches >= 2:
            confidence = 0.75
        elif pattern_matches >= 1:
            confidence = 0.6
        else:
            confidence = 0.1
        
        return confidence, markers_found
    
    def _score_conceptual(self, text: str) -> Tuple[float, List[str]]:
        """Score text for conceptual/theoretical content."""
        markers_found = []
        pattern_matches = 0
        
        for pattern, marker_name in self.conceptual_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_matches += 1
                if marker_name not in markers_found:
                    markers_found.append(marker_name)
        
        # Default to conceptual if no strong indicators of other types
        if pattern_matches >= 2:
            confidence = 0.7
        elif pattern_matches >= 1:
            confidence = 0.5
        else:
            # Base confidence for conceptual (default type)
            confidence = 0.3
        
        return confidence, markers_found
    
    def normalize_dialogue_speakers(self, text: str) -> str:
        """
        Standardize different dialogue formats to consistent THERAPIST:/PATIENT: format.
        Enhanced to handle all clinician and patient variants.
        """
        replacements = [
            # Clinician variations (all become THERAPIST:)
            (r'^T:', 'THERAPIST:'),
            (r'^Th:', 'THERAPIST:'),
            (r'^Pr:', 'THERAPIST:'),
            (r'^Dr:', 'THERAPIST:'),
            (r'^Nr:', 'THERAPIST:'),
            (r'^TERAPEUTA:', 'THERAPIST:'),
            (r'^Terapeuta:', 'THERAPIST:'),
            (r'^CLINICIAN:', 'THERAPIST:'),
            (r'^Clinician:', 'THERAPIST:'),
            (r'^COUNSELOR:', 'THERAPIST:'),
            (r'^Counselor:', 'THERAPIST:'),
            (r'^PRACTITIONER:', 'THERAPIST:'),
            (r'^Practitioner:', 'THERAPIST:'),
            (r'^NURSE:', 'THERAPIST:'),
            (r'^Nurse:', 'THERAPIST:'),
            (r'^DOCTOR:', 'THERAPIST:'),
            (r'^Doctor:', 'THERAPIST:'),
            (r'^PHYSICIAN:', 'THERAPIST:'),
            (r'^Physician:', 'THERAPIST:'),
            # Patient variations (all become PATIENT:)
            (r'^P:', 'PATIENT:'),
            (r'^Pt:', 'PATIENT:'),
            (r'^PACIENTE:', 'PATIENT:'),
            (r'^Paciente:', 'PATIENT:'),
            (r'^CLIENT:', 'PATIENT:'),
            (r'^Client:', 'PATIENT:'),
            (r'^CLIENTE:', 'PATIENT:'),
            (r'^Cliente:', 'PATIENT:'),
        ]
        
        lines = text.split('\n')
        normalized = []
        
        for line in lines:
            normalized_line = line
            for pattern, replacement in replacements:
                normalized_line = re.sub(pattern, replacement, normalized_line, flags=re.IGNORECASE)
            normalized.append(normalized_line)
        
        return '\n'.join(normalized)