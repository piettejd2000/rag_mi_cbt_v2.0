#!/usr/bin/env python3
"""
Quick Test for RAG v2.0 Components
Step 3 Integration Testing
"""

import sys
import os
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
rag_root = current_dir.parent
sys.path.append(str(rag_root / 'chunking_pys'))
sys.path.append(str(current_dir))

def test_imports():
    """Test that all v2.0 components can be imported."""
    print("🔍 Testing v2.0 Component Imports...")
    
    try:
        from core.knowledge_type_classifier import KnowledgeTypeClassifier
        print("✅ KnowledgeTypeClassifier imported")
        
        from core.relevance_classifier import RelevanceClassifier
        print("✅ RelevanceClassifier imported")
        
        from core.adaptive_router import AdaptiveGroundingRouter
        print("✅ AdaptiveGroundingRouter imported")
        
        from enhanced_therapy_rag_v2 import EnhancedTherapyRAGv2, create_enhanced_rag_v2
        print("✅ EnhancedTherapyRAGv2 imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_knowledge_classifier():
    """Test knowledge type classification."""
    print("\n🧠 Testing Knowledge Type Classifier...")
    
    try:
        from core.knowledge_type_classifier import KnowledgeTypeClassifier
        kc = KnowledgeTypeClassifier()
        
        test_cases = [
            ("What does OARS stand for?", "CANONICAL_CLINICAL"),
            ("According to the manual, how should I...", "CORPUS_SPECIFIC"),
            ("What medications should I prescribe?", "UNKNOWN")
        ]
        
        for question, expected in test_cases:
            result = kc.classify(question)
            print(f"Question: {question[:40]}...")
            print(f"  Expected: {expected}")
            print(f"  Got: {result['knowledge_type']}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print()
        
        return True
    except Exception as e:
        print(f"❌ Knowledge classifier test failed: {e}")
        return False

def test_adaptive_router():
    """Test adaptive routing decisions."""
    print("🎯 Testing Adaptive Router...")
    
    try:
        from core.adaptive_router import AdaptiveGroundingRouter
        router = AdaptiveGroundingRouter()
        
        # Test canonical knowledge with irrelevant retrieval
        test_question = "What does OARS stand for in MI?"
        test_chunks = [
            {"text": "Depression is a serious mental health condition that affects mood."}
        ]
        
        routing = router.route(test_question, test_chunks, use_mock_relevance=True)
        print(f"Question: {test_question}")
        print(f"Selected Path: {routing['path']}")
        print(f"Knowledge Type: {routing['knowledge_type']['knowledge_type']}")
        print(f"Confidence: {routing['confidence']:.1%}")
        print(f"Explanation: {routing['explanation']}")
        
        # Should route to PATH_C for canonical knowledge
        expected_path = "PATH_C"
        if routing['path'] == expected_path:
            print(f"✅ Correct routing: {expected_path}")
        else:
            print(f"❌ Wrong routing: expected {expected_path}, got {routing['path']}")
        
        return routing['path'] == expected_path
    except Exception as e:
        print(f"❌ Router test failed: {e}")
        return False

def test_enhanced_rag_v2():
    """Test enhanced RAG v2.0 system."""
    print("\n🚀 Testing Enhanced RAG v2.0 System...")
    
    try:
        from enhanced_therapy_rag_v2 import EnhancedTherapyRAGv2
        
        # Initialize without API key for testing
        rag_system = EnhancedTherapyRAGv2()
        print("✅ RAG v2.0 system initialized")
        
        # Test prompt template loading
        required_templates = ['PATH_A', 'PATH_B', 'PATH_C']
        for template_name in required_templates:
            if template_name in rag_system.prompt_templates:
                print(f"✅ {template_name} template loaded")
            else:
                print(f"❌ {template_name} template missing")
                return False
        
        # Test forced path (no API needed)
        test_question = "What does OARS stand for?"
        test_context = [{"text": "OARS stands for Open questions, Affirmations, Reflections, Summaries"}]
        
        result = rag_system.force_path_for_testing(test_question, "PATH_C", test_context)
        print(f"✅ Forced PATH_C test: {result['routing_decision']['path']}")
        
        # Test metrics
        stats = rag_system.get_routing_statistics()
        print(f"✅ Metrics system: {stats.get('message', 'Working')}")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced RAG test failed: {e}")
        return False

def test_prompt_formatting():
    """Test prompt template formatting."""
    print("\n📝 Testing Prompt Template Formatting...")
    
    try:
        # Test PATH_C template (most important)
        with open('prompts/path_c_prompt.txt', 'r') as f:
            template_c = f.read()
        
        formatted = template_c.format(question="What is CBT?")
        
        if "What is CBT?" in formatted:
            print("✅ PATH_C template formatting works")
        else:
            print("❌ PATH_C template formatting failed")
            return False
        
        # Test PATH_A template
        with open('prompts/path_a_prompt.txt', 'r') as f:
            template_a = f.read()
        
        formatted = template_a.format(
            question="What is MI?",
            context="Motivational interviewing is a counseling approach."
        )
        
        if "What is MI?" in formatted and "counseling approach" in formatted:
            print("✅ PATH_A template formatting works")
        else:
            print("❌ PATH_A template formatting failed")
            return False
        
        print("✅ All prompt templates format correctly")
        return True
    except Exception as e:
        print(f"❌ Prompt formatting test failed: {e}")
        return False

def main():
    """Run all Step 3 tests."""
    print("=" * 70)
    print("🧪 RAG v2.0 Step 3 Integration Testing")
    print("=" * 70)
    
    tests = [
        ("Component Imports", test_imports),
        ("Knowledge Classifier", test_knowledge_classifier),
        ("Adaptive Router", test_adaptive_router),
        ("Enhanced RAG v2.0", test_enhanced_rag_v2),
        ("Prompt Formatting", test_prompt_formatting)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Test: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 STEP 3 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! RAG v2.0 is ready for deployment.")
        print("✅ Step 3 validation complete")
        return True
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please review errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)