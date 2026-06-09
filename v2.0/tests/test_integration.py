#!/usr/bin/env python3
"""
Integration test for RAG v2.0 system
Tests basic functionality and adaptive routing
"""

import sys
import os
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent.parent
rag_root = current_dir.parent
sys.path.append(str(rag_root / 'chunking_pys'))
sys.path.append(str(current_dir))

try:
    # Import v2.0 components
    sys.path.append(str(current_dir))
    from enhanced_therapy_rag_v2 import EnhancedTherapyRAGv2
    from core.adaptive_router import AdaptiveGroundingRouter
    from core.knowledge_type_classifier import KnowledgeTypeClassifier
    from core.relevance_classifier import RelevanceClassifier
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all v2.0 components are properly installed")
    sys.exit(1)


def test_component_imports():
    """Test that all components can be imported."""
    print("🔍 Testing component imports...")
    
    try:
        # Test classifier imports
        knowledge_classifier = KnowledgeTypeClassifier()
        print("✅ KnowledgeTypeClassifier imported successfully")
        
        # Test basic classification
        result = knowledge_classifier.classify("What does OARS stand for?")
        print(f"✅ Knowledge classification works: {result['knowledge_type']}")
        
        return True
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


def test_mock_integration():
    """Test v2.0 system with mock components (no API key needed)."""
    print("\n🔍 Testing mock integration...")
    
    try:
        # Initialize without API key for testing
        rag_system = EnhancedTherapyRAGv2()
        print("✅ RAG v2.0 system initialized")
        
        # Test routing with mock data
        test_question = "What does OARS stand for in Motivational Interviewing?"
        test_context = [
            {"text": "OARS stands for Open questions, Affirmations, Reflections, Summaries"},
            {"text": "These are core techniques in motivational interviewing"}
        ]
        
        # Test forced path (no API needed)
        result = rag_system.force_path_for_testing(test_question, "PATH_C", test_context)
        print(f"✅ Forced path test passed: {result['routing_decision']['path']}")
        
        # Test metrics
        stats = rag_system.get_routing_statistics()
        print(f"✅ Metrics system works: {stats.get('message', 'Stats available')}")
        
        return True
    except Exception as e:
        print(f"❌ Mock integration test failed: {e}")
        return False


def test_prompt_templates():
    """Test that prompt templates are loaded correctly."""
    print("\n🔍 Testing prompt templates...")
    
    try:
        # Test template loading
        rag_system = EnhancedTherapyRAGv2()
        
        # Check that all templates are loaded
        required_paths = ['PATH_A', 'PATH_B', 'PATH_C']
        for path in required_paths:
            if path in rag_system.prompt_templates:
                print(f"✅ {path} template loaded")
            else:
                print(f"❌ {path} template missing")
                return False
        
        # Test template formatting
        template_c = rag_system.prompt_templates['PATH_C']
        formatted = template_c.format(question="Test question")
        if "Test question" in formatted:
            print("✅ Template formatting works")
        else:
            print("❌ Template formatting failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("🚀 RAG v2.0 Integration Test Suite")
    print("=" * 60)
    
    tests = [
        test_component_imports,
        test_mock_integration, 
        test_prompt_templates
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RAG v2.0 integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)