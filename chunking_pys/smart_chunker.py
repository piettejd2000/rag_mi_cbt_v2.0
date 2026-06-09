"""
Smart Document Chunker with Dynamic Sizing Based on Content Type
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from content_detector import ContentTypeDetector, DetectionResult
from claude_classifier import ClaudeClassifier, ClassificationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: Dict
    chunk_id: str
    
    def to_dict(self):
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'metadata': self.metadata
        }


class SmartChunker:
    """
    Intelligent document chunker that adapts strategy based on content type.
    """
    
    def __init__(self, use_claude: bool = True, claude_api_key: Optional[str] = None):
        self.detector = ContentTypeDetector()
        self.use_claude = use_claude
        self.claude_classifier = None
        
        if use_claude:
            try:
                self.claude_classifier = ClaudeClassifier(api_key=claude_api_key)
            except Exception as e:
                logger.warning(f"Claude API initialization failed: {e}. Falling back to pattern-only mode.")
                self.use_claude = False
        
        # Chunking parameters by content type - Enhanced for MI content
        self.chunk_params = {
            'dialogue': {
                'target_size': 800,
                'max_size': 1500,
                'min_size': 300,
                'preserve_boundaries': True
            },
            'procedure': {
                'target_size': 1000,
                'max_size': 2000,
                'min_size': 400,
                'preserve_boundaries': True
            },
            'case_example': {
                'target_size': 1200,
                'max_size': 2500,
                'min_size': 500,
                'preserve_boundaries': True
            },
            'mi_communication': {
                'target_size': 1000,
                'max_size': 2000,
                'min_size': 400,
                'preserve_boundaries': True
            },
            'conceptual': {
                'target_size': 800,
                'max_size': 1200,
                'min_size': 400,
                'preserve_boundaries': False
            }
        }
    
    def chunk_document(self, 
                      text: str, 
                      doc_metadata: Dict,
                      source_filename: str) -> List[Chunk]:
        """
        Process an entire document into intelligent chunks.
        
        Args:
            text: Full document text
            doc_metadata: Base metadata for the document
            source_filename: Name of the source file
            
        Returns:
            List of Chunk objects
        """
        # Extract document structure
        sections = self._segment_by_headers(text)
        
        all_chunks = []
        uncertain_chunks = []
        chunk_counter = 0
        
        logger.info(f"Processing {len(sections)} sections from {source_filename}")
        
        # Phase 1: Pattern-based detection
        for section_idx, section in enumerate(sections):
            section_text = section['text']
            section_title = section.get('title', '')
            
            # Detect content type
            detection = self.detector.detect(section_text, section_title)
            
            if detection.content_type == 'uncertain':
                # Queue for Claude processing
                uncertain_chunks.append({
                    'section_idx': section_idx,
                    'text': section_text,
                    'section_title': section_title,
                    'detection': detection
                })
                continue
            
            # Process with detected type
            chunks = self._chunk_by_type(
                section_text, 
                detection.content_type,
                section_title
            )
            
            # Add metadata to chunks
            for chunk_text in chunks:
                chunk_counter += 1
                chunk = self._create_chunk(
                    text=chunk_text,
                    chunk_id=f"{Path(source_filename).stem}_chunk_{chunk_counter}",
                    doc_metadata=doc_metadata,
                    section_title=section_title,
                    content_type=detection.content_type,
                    detection_method='pattern',
                    confidence=detection.confidence
                )
                all_chunks.append(chunk)
        
        # Phase 2: Process uncertain chunks with Claude
        if uncertain_chunks and self.use_claude and self.claude_classifier:
            logger.info(f"Processing {len(uncertain_chunks)} uncertain sections with Claude API")
            
            # Batch process uncertain chunks
            for i in range(0, len(uncertain_chunks), self.claude_classifier.batch_size):
                batch = uncertain_chunks[i:i + self.claude_classifier.batch_size]
                
                # Prepare batch for Claude
                texts_for_claude = [
                    {'text': uc['text'], 'section_title': uc['section_title']} 
                    for uc in batch
                ]
                
                # Get classifications
                classifications = self.claude_classifier.classify_batch(texts_for_claude)
                
                # Process each classified chunk
                for uc, classification in zip(batch, classifications):
                    chunks = self._chunk_by_type(
                        uc['text'],
                        classification.content_type,
                        uc['section_title']
                    )
                    
                    for chunk_text in chunks:
                        chunk_counter += 1
                        chunk = self._create_chunk(
                            text=chunk_text,
                            chunk_id=f"{Path(source_filename).stem}_chunk_{chunk_counter}",
                            doc_metadata=doc_metadata,
                            section_title=uc['section_title'],
                            content_type=classification.content_type,
                            detection_method='claude',
                            confidence=classification.confidence,
                            topic_tags=classification.topic_tags,
                            technique_tags=classification.technique_tags,
                            bilingual_concepts=classification.bilingual_concepts
                        )
                        all_chunks.append(chunk)
        
        elif uncertain_chunks:
            # Fallback: treat uncertain chunks as conceptual
            logger.warning(f"Claude unavailable. Treating {len(uncertain_chunks)} uncertain chunks as conceptual.")
            for uc in uncertain_chunks:
                chunks = self._chunk_by_type(uc['text'], 'conceptual', uc['section_title'])
                for chunk_text in chunks:
                    chunk_counter += 1
                    chunk = self._create_chunk(
                        text=chunk_text,
                        chunk_id=f"{Path(source_filename).stem}_chunk_{chunk_counter}",
                        doc_metadata=doc_metadata,
                        section_title=uc['section_title'],
                        content_type='conceptual',
                        detection_method='fallback',
                        confidence=0.3
                    )
                    all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from {source_filename}")
        return all_chunks
    
    def _segment_by_headers(self, text: str) -> List[Dict[str, str]]:
        """
        Segment document by headers and section boundaries.
        """
        # Common section patterns
        section_patterns = [
            r'^#{1,3}\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS headers
            r'^(\d+\.\s+[A-Z].+)$',  # Numbered sections
            r'^(Capítulo|Chapter|Sección|Section)\s+\d+',  # Chapter markers
        ]
        
        lines = text.split('\n')
        sections = []
        current_section = {'title': 'Introduction', 'text': ''}
        
        for line in lines:
            # Check if line is a header
            is_header = False
            header_title = None
            
            for pattern in section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    is_header = True
                    header_title = match.group(1) if match.groups() else line.strip()
                    break
            
            if is_header and len(current_section['text']) > 100:
                # Save current section and start new one
                if current_section['text'].strip():
                    sections.append(current_section)
                current_section = {'title': header_title, 'text': ''}
            else:
                current_section['text'] += line + '\n'
        
        # Don't forget the last section
        if current_section['text'].strip():
            sections.append(current_section)
        
        # If no sections found, treat entire document as one section
        if not sections:
            sections = [{'title': 'Full Document', 'text': text}]
        
        return sections
    
    def _chunk_by_type(self, text: str, content_type: str, section_title: str) -> List[str]:
        """
        Apply content-type-specific chunking strategy.
        """
        if content_type == 'dialogue':
            return self._chunk_dialogue(text)
        elif content_type == 'procedure':
            return self._chunk_procedure(text)
        elif content_type == 'case_example':
            return self._chunk_case_example(text)
        elif content_type == 'mi_communication':
            return self._chunk_mi_communication(text)
        else:  # conceptual
            return self._chunk_conceptual(text)
    
    def _chunk_dialogue(self, text: str) -> List[str]:
        """
        Chunk dialogue keeping exchanges together.
        """
        # Normalize speakers first
        text = self.detector.normalize_dialogue_speakers(text)
        
        params = self.chunk_params['dialogue']
        chunks = []
        current_chunk = []
        current_size = 0
        
        lines = text.split('\n')
        
        for line in lines:
            line_size = len(line.split())
            
            # Check if this is a speaker change
            is_speaker = re.match(r'^(THERAPIST|PATIENT):', line.strip())
            
            # If we're at a speaker change and chunk is big enough, save it
            if is_speaker and current_size >= params['min_size']:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            # If chunk would be too big, save current and start new
            elif current_size + line_size > params['max_size']:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Merge very small chunks
        merged_chunks = []
        for chunk in chunks:
            if merged_chunks and len(chunk.split()) < params['min_size'] // 2:
                merged_chunks[-1] += '\n' + chunk
            else:
                merged_chunks.append(chunk)
        
        return merged_chunks if merged_chunks else [text]
    
    def _chunk_mi_communication(self, text: str) -> List[str]:
        """
        Chunk MI communication examples keeping complete exchanges together.
        Enhanced to preserve PRACTITIONER/PATIENT dialogue flows.
        """
        # First normalize all speaker variants
        text = self.detector.normalize_dialogue_speakers(text)
        
        params = self.chunk_params['mi_communication']
        chunks = []
        current_chunk = []
        current_size = 0
        
        lines = text.split('\n')
        
        # Look for MI frameworks (RULE, DARN, etc.) and keep them complete
        mi_framework_markers = ['RULE', 'DARN', 'OARS', 'Four principles', 'Stages of change']
        in_framework = False
        framework_content = []
        
        for line in lines:
            line_size = len(line.split())
            
            # Check if we're starting or ending a framework section
            if any(marker in line for marker in mi_framework_markers):
                if in_framework and framework_content:
                    # End previous framework
                    chunks.append('\n'.join(framework_content))
                    framework_content = []
                in_framework = True
                framework_content.append(line)
                continue
            
            # If we're in a framework, keep collecting
            if in_framework:
                framework_content.append(line)
                # Check if framework is complete (empty line or new section)
                if line.strip() == '' and len(framework_content) > 3:
                    chunks.append('\n'.join(framework_content))
                    framework_content = []
                    in_framework = False
                continue
            
            # Regular chunking for non-framework content
            is_speaker = re.match(r'^(THERAPIST|PATIENT):', line.strip())
            
            # If we hit a speaker change and chunk is big enough, save it
            if is_speaker and current_size >= params['min_size']:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            # If chunk would be too big, save current and start new
            elif current_size + line_size > params['max_size']:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Handle any remaining framework content
        if framework_content:
            chunks.append('\n'.join(framework_content))
        
        # Add remaining regular chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Merge very small chunks
        merged_chunks = []
        for chunk in chunks:
            if merged_chunks and len(chunk.split()) < params['min_size'] // 2:
                merged_chunks[-1] += '\n' + chunk
            else:
                merged_chunks.append(chunk)
        
        return merged_chunks if merged_chunks else [text]
    
    def _chunk_procedure(self, text: str) -> List[str]:
        """
        Chunk procedures keeping steps intact.
        """
        params = self.chunk_params['procedure']
        
        # Split by step markers
        step_patterns = [
            r'(?=Step\s+\d+[:\.])',
            r'(?=Paso\s+\d+[:\.])',
            r'(?=\d+\.\s+[A-Z])',
            r'(?=First[,:])',
            r'(?=Next[,:])',
            r'(?=Finally[,:])',
        ]
        
        chunks = []
        remaining_text = text
        
        for pattern in step_patterns:
            parts = re.split(pattern, remaining_text)
            if len(parts) > 1:
                # Found step structure
                for part in parts:
                    if len(part.split()) >= params['min_size']:
                        chunks.append(part.strip())
                    elif chunks:
                        # Append to previous chunk if too small
                        chunks[-1] += '\n' + part.strip()
                return chunks if chunks else [text]
        
        # No clear step structure, chunk by size
        return self._chunk_by_size(text, params)
    
    def _chunk_case_example(self, text: str) -> List[str]:
        """
        Keep case examples mostly intact.
        """
        params = self.chunk_params['case_example']
        word_count = len(text.split())
        
        # If case is small enough, keep it whole
        if word_count <= params['max_size']:
            return [text]
        
        # Otherwise, try to split at natural boundaries
        return self._chunk_by_size(text, params, preserve_paragraphs=True)
    
    def _chunk_conceptual(self, text: str) -> List[str]:
        """
        Standard semantic chunking for conceptual content.
        """
        params = self.chunk_params['conceptual']
        return self._chunk_by_size(text, params, preserve_paragraphs=True)
    
    def _chunk_by_size(self, text: str, params: Dict, preserve_paragraphs: bool = False) -> List[str]:
        """
        Generic size-based chunking with overlap.
        """
        if preserve_paragraphs:
            # Split by paragraphs
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para.split())
                
                if current_size + para_size > params['max_size'] and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(para)
                current_size += para_size
                
                if current_size >= params['target_size']:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            return chunks if chunks else [text]
        
        else:
            # Simple word-based chunking
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), params['target_size']):
                chunk = ' '.join(words[i:i + params['max_size']])
                if chunk:
                    chunks.append(chunk)
            
            return chunks if chunks else [text]
    
    def _create_chunk(self,
                     text: str,
                     chunk_id: str,
                     doc_metadata: Dict,
                     section_title: str,
                     content_type: str,
                     detection_method: str,
                     confidence: float,
                     topic_tags: Optional[List[str]] = None,
                     technique_tags: Optional[List[str]] = None,
                     bilingual_concepts: Optional[List[Tuple[str, str]]] = None) -> Chunk:
        """
        Create a chunk with full metadata.
        """
        # Detect language
        language = self._detect_language(text)
        
        metadata = {
            **doc_metadata,  # Include base document metadata
            'section_title': section_title,
            'content_type': content_type,
            'detection_method': detection_method,
            'confidence': confidence,
            'language': language,
            'word_count': len(text.split()),
            'char_count': len(text),
        }
        
        # Add optional metadata
        if topic_tags:
            metadata['topic_tags'] = topic_tags
        if technique_tags:
            metadata['technique_tags'] = technique_tags
        if bilingual_concepts:
            metadata['bilingual_concepts'] = bilingual_concepts
        
        return Chunk(
            text=text,
            metadata=metadata,
            chunk_id=chunk_id
        )
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on common words.
        """
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'por', 'con', 'para']
        english_words = ['the', 'is', 'and', 'to', 'of', 'in', 'that', 'it', 'for', 'with']
        
        text_lower = text.lower()
        spanish_count = sum(1 for word in spanish_words if f' {word} ' in text_lower)
        english_count = sum(1 for word in english_words if f' {word} ' in text_lower)
        
        if spanish_count > english_count:
            return 'es'
        elif english_count > spanish_count:
            return 'en'
        else:
            return 'unknown'