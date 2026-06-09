#!/usr/bin/env python3

import sys
import logging
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

def test_enhanced_rag(question):
    """Test Enhanced RAG directly"""
    try:
        # Load embedding model
        model = SentenceTransformer('intfloat/multilingual-e5-base')
        
        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(
            path="/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chroma_db"
        )
        collection = chroma_client.get_collection("therapy_knowledge")
        
        # Get embeddings for question
        question_embedding = model.encode([question])
        
        # Search for relevant chunks
        results = collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=5
        )
        
        # Build context from retrieved chunks
        context_chunks = results['documents'][0]
        context = "\n\n".join(context_chunks)
        
        # Create prompt
        prompt = f"""Eres un asistente especializado en psicología clínica. Usa la siguiente información de libros de texto para responder la pregunta del usuario de manera precisa y profesional.

CONTEXTO CLÍNICO:
{context}

PREGUNTA: {question}

RESPUESTA:"""

        # Query Ollama
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': 'llama3.2:3b',
                                   'prompt': prompt,
                                   'stream': False,
                                   'options': {
                                       'temperature': 0.3,
                                       'num_predict': 200
                                   }
                               })
        
        if response.status_code == 200:
            result = response.json()['response'].strip()
            return f"{result}\n\n[Sources: {len(context_chunks)} clinical documents used]"
        else:
            return f"Error querying Ollama: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    question = "¿Qué es la terapia cognitivo-conductual y cómo se aplica para tratar la depresión?"
    
    print("=" * 100)
    print("ENHANCED RAG TEST - COMPLETE KNOWLEDGE BASE")
    print("=" * 100)
    print(f"QUESTION: {question}")
    print("=" * 100)
    
    print("\n🔸 ENHANCED RAG (Multilingual Semantic + 1040 Clinical Documents)")
    print("-" * 80)
    enhanced_response = test_enhanced_rag(question)
    print(enhanced_response)
    
    print("\n" + "=" * 100)
    print("FINAL KNOWLEDGE BASE STATS:")
    print("• Total documents: 1040+ clinical psychology chapters")
    print("• Books processed:")
    print("  - 12 Anxiety disorder chapters")
    print("  - 10 MI healthcare chapters") 
    print("  - 15 MI psychology chapters")
    print("  - 21 CBT comprehensive chapters")
    print("  - 19 Spanish depression CBT chapters")
    print("  - 9 Spanish behavioral activation chapters")
    print("  - 14 TCC tema files (NEW)")
    print("• Languages: Spanish & English")
    print("• Content types: CBT, MI, Depression, Anxiety, Behavioral Activation")
    print("=" * 100)

if __name__ == "__main__":
    main()