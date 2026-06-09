#!/usr/bin/env python3
"""
Simple RAG Interface that doesn't require external embedding models.
Uses TF-IDF or basic embeddings as fallback, focused on demonstrating parameter control.
"""

import os
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
import requests
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for response generation parameters."""
    temperature: float = 0.7          # Creativity (0.0-2.0)
    max_tokens: int = 512            # Maximum response length
    top_p: float = 0.9               # Nucleus sampling (0.0-1.0)
    top_k: int = 40                  # Top-k sampling
    repeat_penalty: float = 1.1      # Repetition penalty (1.0-2.0)
    stop_sequences: List[str] = None # Stop generation at these tokens
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = ["Human:", "Q:", "Question:", "Pregunta:"]


class SimpleTherapyRAG:
    """
    Simple RAG system using TF-IDF for retrieval and local Llama for generation.
    """
    
    def __init__(self, 
                 chunks_dir: str = "/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chunked_docs",
                 model_name: str = "llama3.2:3b",
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialize the simple RAG system.
        """
        self.chunks_dir = Path(chunks_dir)
        self.model_name = model_name
        self.ollama_host = ollama_host
        
        # Load all chunks
        logger.info("Loading chunks from JSON files...")
        self.chunks = self._load_all_chunks()
        logger.info(f"Loaded {len(self.chunks)} chunks")
        
        # Initialize TF-IDF vectorizer
        logger.info("Building TF-IDF index...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        # Fit vectorizer on all chunk texts
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
        logger.info("TF-IDF index built successfully")
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        # Response style templates
        self.style_prompts = {
            "clinical": "Respond as an experienced clinical supervisor. Be precise, evidence-based, and professional.",
            "brief": "Provide a concise, direct answer in 1-2 sentences.",
            "detailed": "Give a comprehensive explanation with examples and practical applications.",
            "practical": "Focus on concrete, actionable steps that a therapist can implement.",
            "educational": "Explain this concept as if teaching a psychology student.",
            "patient_friendly": "Explain in simple, non-technical language that a patient would understand.",
            "protocol": "Provide a step-by-step protocol. Use numbered steps.",
            "troubleshooting": "Address this as a clinical problem to solve with multiple strategies."
        }
    
    def _load_all_chunks(self) -> List[Dict]:
        """Load all chunks from JSON files."""
        all_chunks = []
        
        json_files = list(self.chunks_dir.glob("*_chunks.json"))
        if not json_files:
            raise FileNotFoundError(f"No chunk files found in {self.chunks_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunks = data.get('chunks', [])
                    
                    # Add source file info to each chunk
                    for chunk in chunks:
                        chunk['source_file'] = json_file.name
                        all_chunks.append(chunk)
                        
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        return all_chunks
    
    def _test_ollama_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                if self.model_name in models:
                    logger.info(f"✅ Ollama connected. Model '{self.model_name}' available.")
                else:
                    logger.warning(f"⚠️  Model '{self.model_name}' not found. Available: {models}")
            else:
                logger.error(f"❌ Ollama server error: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
    
    def retrieve_context(self, 
                        query: str, 
                        n_results: int = 5,
                        content_type_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve relevant context using TF-IDF similarity.
        """
        # Transform query to TF-IDF
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Filter by content type if specified
        if content_type_filter:
            filtered_indices = []
            for i, chunk in enumerate(self.chunks):
                if chunk.get('metadata', {}).get('content_type') == content_type_filter:
                    filtered_indices.append(i)
            
            if filtered_indices:
                filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
                # Sort by similarity
                filtered_similarities.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in filtered_similarities[:n_results]]
            else:
                logger.warning(f"No chunks found with content_type: {content_type_filter}")
                top_indices = similarities.argsort()[-n_results:][::-1]
        else:
            # Get top N most similar chunks
            top_indices = similarities.argsort()[-n_results:][::-1]
        
        # Return relevant chunks with similarity scores
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['similarity_score'] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def generate_response(self, 
                         prompt: str, 
                         config: GenerationConfig = None) -> str:
        """
        Generate response using local Llama model.
        """
        if config is None:
            config = GenerationConfig()
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repeat_penalty": config.repeat_penalty,
                "stop": config.stop_sequences
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                logger.error(f"Ollama generation failed: {response.status_code}")
                return "Error: Could not generate response"
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {e}"
    
    def query(self, 
              question: str,
              response_style: str = "clinical",
              temperature: float = 0.7,
              max_tokens: int = 512,
              n_context_chunks: int = 5,
              content_type_filter: Optional[str] = None,
              language: str = "auto",
              custom_instructions: str = "",
              **generation_kwargs) -> Dict:
        """
        Complete RAG query with full parameter control.
        """
        # Step 1: Retrieve relevant context
        relevant_chunks = self.retrieve_context(
            query=question,
            n_results=n_context_chunks,
            content_type_filter=content_type_filter
        )
        
        # Step 2: Build context string
        context_chunks = []
        for chunk in relevant_chunks:
            metadata = chunk.get('metadata', {})
            content_type = metadata.get('content_type', 'unknown')
            section = metadata.get('section_title', 'N/A')
            score = chunk.get('similarity_score', 0)
            
            chunk_info = f"[{content_type.title()}] {section} (relevancia: {score:.3f}):\n{chunk['text']}"
            context_chunks.append(chunk_info)
        
        context_text = "\n\n".join(context_chunks)
        
        # Step 3: Build prompt with strong language enforcement
        style_instruction = self.style_prompts.get(response_style, "Respond professionally.")
        
        # Build language-appropriate prompt with strong language enforcement
        if language.lower() in ["english", "en"]:
            prompt = f"""You are an expert clinical psychologist specializing in CBT and anxiety disorders.

CRITICAL INSTRUCTION: You MUST respond in English only, regardless of the language of the source material below.

Use the following context from clinical literature to answer the question. The context may be in any language, but you must ALWAYS respond in English.

Context from Clinical Literature (may be in various languages):
{context_text}

Question: {question}

Instructions:
{style_instruction}
{custom_instructions}

IMPORTANT: Your entire response MUST be in English. Do not use Spanish or any other language, even if the source context is in another language. Translate any relevant concepts to English.

Response in English:"""
        elif language.lower() in ["spanish", "es", "español"]:
            prompt = f"""Eres un psicólogo clínico experto en TCC y trastornos de ansiedad.

INSTRUCCIÓN CRÍTICA: DEBES responder únicamente en español, sin importar el idioma del material fuente a continuación.

Usa el siguiente contexto de la literatura clínica para responder la pregunta. El contexto puede estar en cualquier idioma, pero SIEMPRE debes responder en español.

Contexto de Literatura Clínica (puede estar en varios idiomas):
{context_text}

Pregunta: {question}

Instrucciones:
{style_instruction}
{custom_instructions}

IMPORTANTE: Tu respuesta completa DEBE estar en español. No uses inglés u otro idioma, incluso si el contexto fuente está en otro idioma. Traduce cualquier concepto relevante al español.

Respuesta en español:"""
        else:  # auto mode - detect from question
            if any(word in question.lower() for word in ["qué", "cómo", "cuándo", "dónde", "por qué", "puedo", "debe", "tiene", "está"]):
                prompt = f"""Eres un psicólogo clínico experto en TCC y trastornos de ansiedad. Usa el siguiente contexto de la literatura clínica para responder la pregunta.

Contexto de Literatura Clínica:
{context_text}

Pregunta: {question}

Instrucciones:
{style_instruction}
{custom_instructions}

Respuesta:"""
            else:
                prompt = f"""You are an expert clinical psychologist specializing in CBT and anxiety disorders. Use the following context from clinical literature to answer the question.

Context from Clinical Literature:
{context_text}

Question: {question}

Instructions:
{style_instruction}
{custom_instructions}

Response:"""
        
        # Step 4: Generate response
        config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            **generation_kwargs
        )
        
        response = self.generate_response(prompt, config)
        
        # Step 5: Return complete result
        return {
            'response': response,
            'question': question,
            'context_chunks_used': len(relevant_chunks),
            'context_sources': [
                {
                    'content_type': chunk.get('metadata', {}).get('content_type'),
                    'section': chunk.get('metadata', {}).get('section_title'),
                    'similarity': chunk.get('similarity_score', 0),
                    'source_file': chunk.get('source_file')
                }
                for chunk in relevant_chunks
            ],
            'generation_config': {
                'temperature': temperature,
                'max_tokens': max_tokens,
                'style': response_style
            }
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        content_types = {}
        languages = {}
        
        for chunk in self.chunks:
            metadata = chunk.get('metadata', {})
            ct = metadata.get('content_type', 'unknown')
            lang = metadata.get('language', 'unknown')
            content_types[ct] = content_types.get(ct, 0) + 1
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'content_types': content_types,
            'languages': languages,
            'available_styles': list(self.style_prompts.keys())
        }


def demonstrate_parameter_control():
    """Show exactly how to control each response parameter."""
    
    print("\n" + "="*80)
    print("RAG PARAMETER CONTROL DEMONSTRATION")
    print("="*80)
    
    try:
        rag = SimpleTherapyRAG()
        
        # Print system info
        stats = rag.get_stats()
        print(f"Knowledge Base: {stats['total_chunks']} chunks")
        print(f"Content Types: {stats['content_types']}")
        
        question = "¿Cómo ayudo a un paciente con ataques de pánico?"
        
        print(f"\nDemo Question: {question}")
        print("="*80)
        
        # 1. TEMPERATURE CONTROL
        print("\n1. TEMPERATURE CONTROL (Creativity)")
        print("-" * 50)
        
        for temp, description in [(0.2, "Focused"), (0.7, "Balanced"), (1.2, "Creative")]:
            result = rag.query(
                question=question,
                temperature=temp,
                max_tokens=150,
                response_style="clinical"
            )
            print(f"\n🌡️ Temperature {temp} ({description}):")
            print(f"   {result['response'][:200]}...")
        
        # 2. RESPONSE STYLE CONTROL
        print("\n\n2. RESPONSE STYLE CONTROL")
        print("-" * 50)
        
        for style in ["brief", "practical", "patient_friendly"]:
            result = rag.query(
                question=question,
                response_style=style,
                max_tokens=200
            )
            print(f"\n🎭 Style '{style}':")
            print(f"   {result['response'][:250]}...")
        
        # 3. LENGTH CONTROL
        print("\n\n3. RESPONSE LENGTH CONTROL")
        print("-" * 50)
        
        for tokens, desc in [(100, "Short"), (300, "Medium"), (600, "Long")]:
            result = rag.query(
                question=question,
                max_tokens=tokens,
                response_style="detailed"
            )
            print(f"\n📏 {desc} ({tokens} tokens):")
            print(f"   Length: {len(result['response'])} chars")
            print(f"   Preview: {result['response'][:150]}...")
        
        print("\n" + "="*80)
        print("PARAMETER CONTROL DEMO COMPLETE!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nError: {e}")
        print("Make sure:")
        print("1. Chunks are created (run chunk_documents.py first)")
        print("2. Ollama is running (ollama serve)")
        print("3. Llama model is available (ollama pull llama3.2:3b)")


if __name__ == "__main__":
    demonstrate_parameter_control()