#!/usr/bin/env python3
"""
Compare Base Llama vs TF-IDF RAG vs Enhanced Multilingual RAG
Shows the difference in response quality across all three systems.
"""

import sys
import json
import requests
import logging
from pathlib import Path
from typing import Dict, List
import time

# Add the current directory to path for imports
sys.path.append('/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chunking_pys')

logging.basicConfig(level=logging.WARNING)  # Reduce noise

class SystemComparison:
    """Compare Base Llama, TF-IDF RAG, and Enhanced RAG systems."""
    
    def __init__(self):
        self.ollama_host = "http://localhost:11434"
        self.model_name = "llama3.2:3b"
        
        # Test questions for comparison
        self.test_questions = [
            {
                "question": "¿Qué es la terapia cognitivo-conductual?",
                "description": "Spanish question about CBT - tests multilingual capability"
            },
            {
                "question": "How do I help a patient with panic attacks?",
                "description": "Clinical intervention question - tests practical guidance"
            },
            {
                "question": "What are the symptoms of anxiety?",
                "description": "Factual knowledge question - tests retrieval accuracy"
            },
            {
                "question": "Patient refuses to do homework assignments",
                "description": "Clinical problem-solving - tests context relevance"
            }
        ]
    
    def query_base_llama(self, question: str) -> str:
        """Query base Llama without any context."""
        prompt = f"""You are a clinical psychologist. Answer this question about mental health:

Question: {question}

Answer:"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.6,
                "num_predict": 300
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_host}/api/generate", json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return f"Error: Status {response.status_code}"
        except Exception as e:
            return f"Error: {e}"
    
    def query_tfidf_rag(self, question: str) -> Dict:
        """Query TF-IDF based RAG system."""
        try:
            from simple_rag import SimpleTherapyRAG
            rag = SimpleTherapyRAG()
            result = rag.query(
                question=question,
                response_style="clinical",
                temperature=0.6,
                max_tokens=300
            )
            return result
        except Exception as e:
            return {"response": f"TF-IDF RAG Error: {e}", "context_chunks_used": 0}
    
    def query_enhanced_rag(self, question: str) -> Dict:
        """Query Enhanced multilingual RAG system."""
        try:
            from therapy_rag import TherapyRAG
            rag = TherapyRAG()
            result = rag.query(
                question=question,
                response_style="clinical",
                temperature=0.6,
                max_tokens=300
            )
            return result
        except Exception as e:
            return {"response": f"Enhanced RAG Error: {e}", "context_chunks_used": 0}
    
    def run_comparison(self):
        """Run complete comparison across all systems."""
        print("🔬 SYSTEM COMPARISON: Base Llama vs TF-IDF RAG vs Enhanced RAG")
        print("=" * 80)
        
        for i, test in enumerate(self.test_questions, 1):
            question = test["question"]
            description = test["description"]
            
            print(f"\n{i}. TEST: {description}")
            print(f"   Question: {question}")
            print("=" * 80)
            
            # Base Llama
            print(f"\n📝 BASE LLAMA (No Context):")
            print("-" * 40)
            base_response = self.query_base_llama(question)
            print(f"{base_response[:400]}...")
            
            # TF-IDF RAG  
            print(f"\n🔍 TF-IDF RAG (Keyword Matching):")
            print("-" * 40)
            tfidf_result = self.query_tfidf_rag(question)
            print(f"Response: {tfidf_result['response'][:400]}...")
            print(f"Context chunks used: {tfidf_result.get('context_chunks_used', 0)}")
            
            # Enhanced RAG
            print(f"\n🌟 ENHANCED RAG (Multilingual Semantic):")
            print("-" * 40)
            enhanced_result = self.query_enhanced_rag(question)
            print(f"Response: {enhanced_result['response'][:400]}...")
            print(f"Context chunks used: {enhanced_result.get('context_chunks_used', 0)}")
            
            if i < len(self.test_questions):
                print(f"\n{'-' * 80}")
                input("Press Enter to continue to next test...")
        
        self._print_summary()
    
    def _print_summary(self):
        """Print comparison summary."""
        print("\n" + "=" * 80)
        print("📊 COMPARISON SUMMARY")
        print("=" * 80)
        
        print("""
🏆 WHEN TO USE EACH SYSTEM:

📝 BASE LLAMA:
   ✅ General psychology knowledge
   ✅ Quick responses without context
   ❌ May hallucinate or give generic advice
   ❌ No access to specific clinical protocols

🔍 TF-IDF RAG:
   ✅ Fast keyword-based retrieval
   ✅ Works offline without HuggingFace
   ❌ Misses semantic relationships
   ❌ Poor cross-language performance
   ❌ "ansiedad" ≠ "anxiety" in search

🌟 ENHANCED RAG:
   ✅ Semantic understanding across languages
   ✅ "ansiedad" = "anxiety" in retrieval
   ✅ Better clinical context matching
   ✅ Finds related concepts not just keywords
   ❌ Requires HuggingFace setup (one-time)

RECOMMENDATION: Use Enhanced RAG for best results with your bilingual clinical content.
        """)

def demonstrate_parameter_control():
    """Show parameter control with the enhanced system."""
    print("\n" + "=" * 80)
    print("🎛️  PARAMETER CONTROL DEMONSTRATION")
    print("=" * 80)
    
    try:
        from therapy_rag import TherapyRAG
        rag = TherapyRAG()
        
        question = "¿Cómo estructuro la primera sesión de TCC?"
        
        print(f"Demo Question: {question}")
        print("=" * 60)
        
        # Parameter variations
        configs = [
            {
                "name": "🎯 FOCUSED (Temperature 0.2)",
                "params": {"temperature": 0.2, "response_style": "protocol", "max_tokens": 200}
            },
            {
                "name": "⚖️  BALANCED (Temperature 0.7)",
                "params": {"temperature": 0.7, "response_style": "clinical", "max_tokens": 200}
            },
            {
                "name": "🎨 CREATIVE (Temperature 1.2)",
                "params": {"temperature": 1.2, "response_style": "practical", "max_tokens": 200}
            },
            {
                "name": "📋 BRIEF STYLE",
                "params": {"response_style": "brief", "max_tokens": 100}
            },
            {
                "name": "👥 PATIENT-FRIENDLY",
                "params": {"response_style": "patient_friendly", "language": "spanish", "max_tokens": 200}
            }
        ]
        
        for config in configs:
            print(f"\n{config['name']}:")
            print("-" * 50)
            result = rag.query(question, **config['params'])
            print(f"Response: {result['response'][:300]}...")
            print(f"Params: {config['params']}")
            
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    comparison = SystemComparison()
    
    # Check Ollama connection
    try:
        response = requests.get(f"{comparison.ollama_host}/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama not running. Start with: ollama serve")
            sys.exit(1)
    except:
        print("❌ Cannot connect to Ollama. Make sure it's running: ollama serve")
        sys.exit(1)
    
    print("🚀 Starting system comparison...")
    print("This will compare Base Llama vs TF-IDF RAG vs Enhanced RAG")
    print("\nPress Ctrl+C to stop at any time\n")
    
    try:
        comparison.run_comparison()
        demonstrate_parameter_control()
    except KeyboardInterrupt:
        print("\n\n👋 Comparison stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()