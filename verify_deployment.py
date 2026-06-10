#!/usr/bin/env python3
"""
Deployment verification script for RAG v2.0 system
Tests all components before Streamlit Cloud deployment
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    try:
        import streamlit
        print("✅ Streamlit imported")
        
        import anthropic
        print("✅ Anthropic imported")
        
        import chromadb
        print("✅ ChromaDB imported")
        
        import sentence_transformers
        print("✅ Sentence Transformers imported")
        
        # Test v2.0 imports
        sys.path.append('v2.0')
        from enhanced_therapy_rag_v2 import create_enhanced_rag_v2
        print("✅ Enhanced RAG v2.0 imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_knowledge_base():
    """Test knowledge base exists and is populated."""
    print("\nTesting knowledge base...")
    
    chroma_path = Path("chroma_db")
    if chroma_path.exists():
        print("✅ ChromaDB directory exists")
        
        # Count files
        db_files = list(chroma_path.rglob("*"))
        if db_files:
            total_size = sum(f.stat().st_size for f in db_files if f.is_file())
            print(f"✅ Database size: {total_size / (1024*1024):.1f} MB")
            return True
        else:
            print("❌ Database directory is empty")
            return False
    else:
        print("❌ ChromaDB directory not found")
        return False

def test_v2_system():
    """Test v2.0 system initialization."""
    print("\nTesting v2.0 system...")
    try:
        sys.path.append('v2.0')
        from enhanced_therapy_rag_v2 import create_enhanced_rag_v2
        
        # Create system (without API key for now)
        rag = create_enhanced_rag_v2()
        print("✅ Enhanced RAG v2.0 created")
        
        # Test knowledge base connection
        stats = rag.get_collection_stats()
        if stats['total_documents'] > 0:
            print(f"✅ Knowledge base loaded: {stats['total_documents']} documents")
            return True
        else:
            print("⚠️  Knowledge base is empty")
            return False
            
    except Exception as e:
        print(f"❌ v2.0 system test failed: {e}")
        return False

def test_streamlit_app():
    """Test streamlit app compilation."""
    print("\nTesting Streamlit app...")
    try:
        import py_compile
        py_compile.compile('streamlit_app.py', doraise=True)
        print("✅ Streamlit app compiles successfully")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Streamlit app compilation failed: {e}")
        return False

def main():
    print("🚀 RAG v2.0 Deployment Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_knowledge_base, 
        test_v2_system,
        test_streamlit_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Review before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)