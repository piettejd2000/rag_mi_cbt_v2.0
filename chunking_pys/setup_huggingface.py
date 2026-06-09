#!/usr/bin/env python3
"""
Script to test and setup HuggingFace authentication for better embeddings.
"""

import os
import sys
from pathlib import Path

def test_huggingface_auth():
    """Test if HuggingFace authentication is working."""
    print("Testing HuggingFace Authentication...")
    print("="*50)
    
    # Check for token in various locations
    token_sources = [
        ("Environment Variable", os.environ.get('HUGGINGFACE_HUB_TOKEN')),
        ("Environment Variable (alt)", os.environ.get('HF_TOKEN')),
        ("Local file", None)  # We'll check file below
    ]
    
    # Check local file
    token_file = Path('/Users/johnpiette/healthcare_rl/.huggingface_token')
    if token_file.exists():
        try:
            with open(token_file, 'r') as f:
                local_token = f.read().strip()
            token_sources[2] = ("Local file", local_token[:10] + "..." if local_token else None)
        except:
            pass
    
    # Display token status
    token_found = False
    for source, token in token_sources:
        if token:
            print(f"✅ {source}: {token[:10]}...")
            token_found = True
            break
        else:
            print(f"❌ {source}: Not found")
    
    if not token_found:
        print("\n❌ No HuggingFace token found!")
        print("\nTo fix this, run ONE of these commands:")
        print("\n1. Environment variable (recommended):")
        print("   export HUGGINGFACE_HUB_TOKEN='your_token_here'")
        print("\n2. HuggingFace CLI (easiest):")
        print("   pip install huggingface_hub")
        print("   huggingface-cli login")
        print("\n3. Local file:")
        print("   echo 'your_token_here' > /Users/johnpiette/healthcare_rl/.huggingface_token")
        return False
    
    # Test model download
    print("\nTesting model download...")
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Downloading multilingual-e5-base (this may take a few minutes first time)...")
        model = SentenceTransformer('intfloat/multilingual-e5-base')
        
        # Test encoding
        test_texts = [
            "¿Qué es la terapia cognitivo-conductual?",
            "What is cognitive behavioral therapy?",
            "anxiety disorders treatment protocol"
        ]
        
        embeddings = model.encode(test_texts)
        print(f"✅ Successfully encoded {len(test_texts)} test sentences")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

def setup_enhanced_rag():
    """Set up the enhanced RAG system with proper embeddings."""
    print("\n" + "="*60)
    print("SETTING UP ENHANCED RAG SYSTEM")
    print("="*60)
    
    # Test authentication first
    if not test_huggingface_auth():
        return False
    
    print("\nCreating enhanced embedding loader...")
    
    # Update the embedding loader to use the authenticated model
    try:
        # Import and run the embedding loader
        sys.path.append('/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chunking_pys')
        from embedding_loader import EmbeddingLoader
        
        # This should now work with authentication
        loader = EmbeddingLoader(
            chunks_dir=Path('/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chunked_docs'),
            embedding_model='intfloat/multilingual-e5-base'
        )
        
        print("✅ Enhanced embedding model loaded successfully!")
        
        # Process all chunk files
        print("Loading chunks to ChromaDB...")
        loader.process_all_files()
        
        # Test retrieval
        print("\nTesting multilingual retrieval...")
        test_queries = [
            "¿Qué es la ansiedad?",
            "How to help panic attacks?",
            "técnicas de exposición"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            loader.test_retrieval(query, n_results=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced RAG setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 ENHANCED RAG SETUP")
    print("This will set up better multilingual embeddings for your RAG system")
    print("="*70)
    
    success = setup_enhanced_rag()
    
    if success:
        print("\n" + "="*70)
        print("✅ ENHANCED RAG SYSTEM READY!")
        print("="*70)
        print("\nBenefits of the enhanced system:")
        print("• Better Spanish/English cross-language retrieval")
        print("• More accurate semantic matching")
        print("• Better clinical terminology understanding")
        print("• Improved relevance scoring")
        print("\nYou can now use therapy_rag.py for optimal results!")
    else:
        print("\n" + "="*70)
        print("❌ SETUP INCOMPLETE")
        print("="*70)
        print("You can still use simple_rag.py with TF-IDF retrieval")
        print("Or follow the authentication steps above and try again")