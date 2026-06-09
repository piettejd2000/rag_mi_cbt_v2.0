"""
Claude API Integration for Content Classification
Handles uncertain content type detection and metadata extraction
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import anthropic
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    content_type: str
    topic_tags: List[str]
    technique_tags: List[str]
    bilingual_concepts: List[Tuple[str, str]]  # (spanish, english) pairs
    confidence: float


class ClaudeClassifier:
    """
    Uses Claude API for sophisticated content classification and metadata extraction.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude client with API key from various sources.
        """
        # Try to get API key from various sources
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._load_api_key()
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Please provide it or save it in .anthropic_key file")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.batch_size = 5  # Process 5 chunks at once to reduce API calls
        
    def _load_api_key(self) -> Optional[str]:
        """
        Load API key from various possible locations.
        """
        # Check environment variable
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if api_key:
            return api_key
        
        # Check .anthropic_key file in root directory
        root_dir = Path('/Users/johnpiette/healthcare_rl')
        key_files = [
            root_dir / '.anthropic_key',
            root_dir / '.env',
            Path.home() / '.anthropic_key',
        ]
        
        for key_file in key_files:
            if key_file.exists():
                with open(key_file, 'r') as f:
                    content = f.read().strip()
                    # Handle .env format
                    if 'ANTHROPIC_API_KEY=' in content:
                        for line in content.split('\n'):
                            if line.startswith('ANTHROPIC_API_KEY='):
                                return line.split('=', 1)[1].strip().strip('"').strip("'")
                    # Direct key format
                    elif content and not '=' in content:
                        return content
        
        return None
    
    def classify_batch(self, texts: List[Dict[str, str]], max_retries: int = 5) -> List[ClassificationResult]:
        """
        Classify multiple text chunks in a single API call with aggressive retry.
        
        Args:
            texts: List of dicts with 'text' and optional 'section_title'
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of ClassificationResult objects
        """
        prompt = self._build_batch_prompt(texts)
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",  # Fast and cheap
                    max_tokens=1000,
                    temperature=0,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                # Parse the response
                return self._parse_batch_response(response.content[0].text, len(texts))
                
            except anthropic.RateLimitError as e:
                wait_time = min(2 ** attempt * 2, 60)  # Exponential backoff, max 60s
                logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                
            except anthropic.APIError as e:
                wait_time = min(2 ** attempt, 30)
                logger.warning(f"API error: {e}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise
        
        # If all retries failed, return default classifications
        logger.error("All retry attempts failed. Using default classification.")
        return [self._default_classification() for _ in texts]
    
    def _build_batch_prompt(self, texts: List[Dict[str, str]]) -> str:
        """
        Build a prompt for batch classification.
        """
        prompt = """You are a clinical psychology content analyzer. Classify each text excerpt and extract metadata.

For each text, provide:
1. content_type: One of [dialogue, procedure, case_example, mi_communication, conceptual]
2. topic_tags: Clinical topics mentioned (e.g., anxiety, depression, PTSD, panic, phobia, behavior_change, motivation)
3. technique_tags: Therapeutic techniques mentioned (e.g., CBT, MI, exposure, behavioral_activation, cognitive_restructuring, RULE, DARN, OARS)
4. bilingual_concepts: Key clinical terms in both Spanish and English as pairs

Respond in JSON format with an array matching the input order.

Example response format:
[
  {
    "content_type": "mi_communication",
    "topic_tags": ["anxiety", "panic"],
    "technique_tags": ["MI", "RULE"],
    "bilingual_concepts": [["motivacion", "motivation"], ["entrevista", "interview"]],
    "confidence": 0.9
  }
]

Texts to classify:
"""
        
        for i, text_dict in enumerate(texts, 1):
            text_sample = text_dict['text'][:500] if len(text_dict['text']) > 500 else text_dict['text']
            section = text_dict.get('section_title', 'No title')
            prompt += f"\n---TEXT {i}---\n"
            prompt += f"Section: {section}\n"
            prompt += f"Content: {text_sample}\n"
        
        prompt += "\n---END---\n\nProvide the JSON array response:"
        
        return prompt
    
    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[ClassificationResult]:
        """
        Parse Claude's response into ClassificationResult objects.
        """
        try:
            # Extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            results_data = json.loads(response_text.strip())
            
            # Ensure we have the right number of results
            if not isinstance(results_data, list):
                results_data = [results_data]
            
            results = []
            for i, data in enumerate(results_data):
                if i >= expected_count:
                    break
                
                result = ClassificationResult(
                    content_type=data.get('content_type', 'conceptual'),
                    topic_tags=data.get('topic_tags', []),
                    technique_tags=data.get('technique_tags', []),
                    bilingual_concepts=[tuple(pair) for pair in data.get('bilingual_concepts', [])],
                    confidence=data.get('confidence', 0.8)
                )
                results.append(result)
            
            # Pad with defaults if needed
            while len(results) < expected_count:
                results.append(self._default_classification())
            
            return results
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse Claude response: {e}")
            logger.debug(f"Response was: {response_text[:500]}")
            return [self._default_classification() for _ in range(expected_count)]
    
    def _default_classification(self) -> ClassificationResult:
        """
        Return a default classification when API fails.
        """
        return ClassificationResult(
            content_type='conceptual',
            topic_tags=[],
            technique_tags=[],
            bilingual_concepts=[],
            confidence=0.3
        )
    
    def extract_bilingual_concepts(self, text: str, max_concepts: int = 10) -> List[Tuple[str, str]]:
        """
        Extract key bilingual concept pairs from text.
        """
        prompt = f"""Identify the {max_concepts} most important clinical psychology concepts in this text.
For each concept, provide the Spanish and English terms as a pair.

Text: {text[:1000]}

Return only a JSON array of [spanish, english] pairs, like:
[["ansiedad", "anxiety"], ["terapia cognitiva", "cognitive therapy"]]

JSON response:"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            concepts = json.loads(response_text.strip())
            return [tuple(pair) for pair in concepts if len(pair) == 2]
            
        except Exception as e:
            logger.error(f"Failed to extract bilingual concepts: {e}")
            return []