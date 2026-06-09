#!/usr/bin/env python3
"""
Complete RAG Interface for Therapy Knowledge Base
Connects chunked documents to local Llama 3.2 model with full parameter control
"""

import os
import json
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from dataclasses import dataclass

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


class TherapyRAG:
    """
    Complete RAG system for therapy knowledge base with local Llama 3.2.
    """
    
    def __init__(self, 
                 chroma_path: str = "/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chroma_db",
                 collection_name: str = "therapy_knowledge",
                 model_name: str = "llama3.2:3b",
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialize the RAG system.
        
        Args:
            chroma_path: Path to ChromaDB database
            collection_name: Name of the ChromaDB collection
            model_name: Name of the Ollama model
            ollama_host: Ollama server URL
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        
        # Initialize embedding model (try alternatives if multilingual fails)
        logger.info("Loading embedding model...")
        try:
            self.embedder = SentenceTransformer('intfloat/multilingual-e5-base')
        except Exception as e:
            logger.warning(f"Failed to load multilingual-e5-base: {e}")
            logger.info("Trying alternative: all-MiniLM-L6-v2")
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        logger.info(f"Connecting to ChromaDB at {chroma_path}")
        try:
            self.client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.client.get_collection(name=collection_name)
            doc_count = self.collection.count()
            logger.info(f"Connected to collection '{collection_name}' with {doc_count} documents")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        # Response style templates
        self.style_prompts = {
            "clinical": "Respond as an experienced clinical supervisor. Be precise, evidence-based, and professional. Include specific techniques and considerations.",
            
            "brief": "Provide a concise, direct answer in 1-2 sentences. Focus on the most essential information.",
            
            "detailed": "Give a comprehensive explanation with examples, background context, and practical applications. Include relevant techniques and research.",
            
            "practical": "Focus on concrete, actionable steps that a therapist can implement. Provide specific guidance and techniques.",
            
            "educational": "Explain this concept as if teaching a psychology student. Include definitions, theory, and examples.",
            
            "patient_friendly": "Explain in simple, non-technical language that a patient would understand. Be empathetic and reassuring.",
            
            "protocol": "Provide a step-by-step protocol or procedure. Use numbered steps and be very specific about implementation.",
            
            "troubleshooting": "Address this as a clinical problem to solve. Identify potential causes and provide multiple solution strategies."
        }
    
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
                        content_type_filter: Optional[str] = None) -> Dict:
        """
        Retrieve relevant context from the knowledge base.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            content_type_filter: Optional filter by content type
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query, convert_to_tensor=False)
        
        # Prepare search filters
        where_clause = None
        if content_type_filter:
            where_clause = {"content_type": content_type_filter}
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_clause
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def generate_response(self, 
                         prompt: str, 
                         config: GenerationConfig = None) -> str:
        """
        Generate response using local Llama model.
        
        Args:
            prompt: Complete prompt including context
            config: Generation configuration
            
        Returns:
            Generated response text
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
        
        Args:
            question: User's question
            response_style: Style of response (clinical, brief, detailed, etc.)
            temperature: Creativity level (0.0-2.0)
            max_tokens: Maximum response length
            n_context_chunks: Number of context chunks to retrieve
            content_type_filter: Filter by content type (dialogue, procedure, etc.)
            language: Response language preference
            custom_instructions: Additional instructions for the model
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary with response and metadata
        """
        # Step 1: Retrieve relevant context
        context_data = self.retrieve_context(
            query=question,
            n_results=n_context_chunks,
            content_type_filter=content_type_filter
        )
        
        # Step 2: Build context string
        context_chunks = []
        for doc, meta in zip(context_data['documents'], context_data['metadatas']):
            content_type = meta.get('content_type', 'unknown')
            section = meta.get('section_title', 'N/A')
            chunk_info = f"[{content_type.title()}] {section}: {doc}"
            context_chunks.append(chunk_info)
        
        context_text = "\n\n".join(context_chunks)
        
        # Step 3: Build prompt
        style_instruction = self.style_prompts.get(response_style, "Respond professionally.")
        
        # Build language-appropriate prompt with strong language enforcement
        if language.lower() in ["english", "en"]:
            prompt = f"""You are an expert clinical psychologist with extensive knowledge of CBT and anxiety disorders. 

CRITICAL INSTRUCTION: You MUST respond in English only, regardless of the language of the source material below.

Use the following context from clinical literature to answer the question. Note that the context may be in any language, but you must ALWAYS respond in English.

Context from Clinical Literature (may be in various languages):
{context_text}

Question: {question}

Instructions:
{style_instruction}
{custom_instructions}

IMPORTANT: Your entire response MUST be in English. Do not use Spanish or any other language, even if the source context is in another language. Translate any relevant concepts to English.

Response in English:"""
        elif language.lower() in ["spanish", "es", "español"]:
            prompt = f"""Eres un psicólogo clínico experto con amplio conocimiento de TCC y trastornos de ansiedad.

INSTRUCCIÓN CRÍTICA: DEBES responder únicamente en español, sin importar el idioma del material fuente a continuación.

Usa el siguiente contexto de la literatura clínica para responder la pregunta. Ten en cuenta que el contexto puede estar en cualquier idioma, pero SIEMPRE debes responder en español.

Contexto de Literatura Clínica (puede estar en varios idiomas):
{context_text}

Pregunta: {question}

Instrucciones:
{style_instruction}
{custom_instructions}

IMPORTANTE: Tu respuesta completa DEBE estar en español. No uses inglés u otro idioma, incluso si el contexto fuente está en otro idioma. Traduce cualquier concepto relevante al español.

Respuesta en español:"""
        else:  # auto mode
            # Detect language from question
            if any(word in question.lower() for word in ["qué", "cómo", "cuándo", "dónde", "por qué", "puedo", "debe", "tiene", "está"]):
                language_instruction = "Responde en español."
            else:
                language_instruction = "Respond in English."
                
            prompt = f"""You are an expert clinical psychologist with extensive knowledge of CBT and anxiety disorders. Use the following context from clinical literature to answer the question.

Context from Clinical Literature:
{context_text}

Question: {question}

Instructions:
{style_instruction}
{language_instruction}
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
            'context_chunks_used': len(context_data['documents']),
            'context_sources': [
                {
                    'content_type': meta.get('content_type'),
                    'section': meta.get('section_title'),
                    'confidence': 1 - dist  # Convert distance to confidence
                }
                for meta, dist in zip(context_data['metadatas'], context_data['distances'])
            ],
            'generation_config': {
                'temperature': temperature,
                'max_tokens': max_tokens,
                'style': response_style
            }
        }
    
    def batch_query(self, questions: List[str], **kwargs) -> List[Dict]:
        """Process multiple questions with the same parameters."""
        return [self.query(q, **kwargs) for q in questions]
    
    def list_available_styles(self) -> Dict[str, str]:
        """Get available response styles and their descriptions."""
        return {
            style: prompt.split('.')[0] + '.'
            for style, prompt in self.style_prompts.items()
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        total_docs = self.collection.count()
        
        # Get ALL documents to analyze complete content type distribution
        # Use batching for large collections to avoid memory issues
        content_types = {}
        languages = {}
        batch_size = 1000
        offset = 0
        
        while offset < total_docs:
            batch = self.collection.get(
                limit=min(batch_size, total_docs - offset),
                offset=offset
            )
            
            for meta in batch['metadatas']:
                ct = meta.get('content_type', 'unknown')
                lang = meta.get('language', 'unknown')
                content_types[ct] = content_types.get(ct, 0) + 1
                languages[lang] = languages.get(lang, 0) + 1
            
            offset += batch_size
        
        return {
            'total_documents': total_docs,
            'content_types': content_types,
            'languages': languages,
            'available_styles': list(self.style_prompts.keys())
        }


def main():
    """Demo the RAG system with various parameter configurations."""
    rag = TherapyRAG()
    
    # Print system info
    print("\n" + "="*80)
    print("THERAPY RAG SYSTEM - PARAMETER CONTROL DEMO")
    print("="*80)
    
    stats = rag.get_collection_stats()
    print(f"Knowledge Base: {stats['total_documents']} documents")
    print(f"Content Types: {stats['content_types']}")
    print(f"Available Styles: {stats['available_styles']}")
    
    # Demo questions
    demo_questions = [
        "¿Qué es la terapia cognitivo-conductual?",
        "How do I help a patient with panic attacks?",
        "What should I do when a patient won't do homework?"
    ]
    
    # Demo different parameter configurations
    configs = [
        {
            "name": "Brief Clinical Response",
            "params": {
                "response_style": "brief",
                "temperature": 0.3,
                "max_tokens": 100
            }
        },
        {
            "name": "Detailed Educational",
            "params": {
                "response_style": "detailed", 
                "temperature": 0.7,
                "max_tokens": 600
            }
        },
        {
            "name": "Practical Protocol",
            "params": {
                "response_style": "protocol",
                "temperature": 0.5,
                "max_tokens": 400,
                "content_type_filter": "procedure"
            }
        }
    ]
    
    # Run demos
    for question in demo_questions[:1]:  # Just first question for demo
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")
        
        for config in configs:
            print(f"\n{'-'*40}")
            print(f"CONFIG: {config['name']}")
            print(f"PARAMS: {config['params']}")
            print(f"{'-'*40}")
            
            result = rag.query(question, **config['params'])
            print(f"RESPONSE: {result['response']}")
            print(f"SOURCES: {len(result['context_chunks_used'])} chunks used")


if __name__ == "__main__":
    main()