#!/usr/bin/env python3
"""
Concrete examples of how to control RAG response parameters.
Shows exact usage patterns for different clinical scenarios.
"""

from therapy_rag import TherapyRAG, GenerationConfig

def demonstrate_parameter_control():
    """Show exactly how to control each response parameter."""
    
    rag = TherapyRAG()
    
    print("\n" + "="*80)
    print("RAG PARAMETER CONTROL EXAMPLES")
    print("="*80)
    
    question = "¿Cómo ayudo a un paciente con ataques de pánico?"
    
    # ==========================================
    # 1. TEMPERATURE CONTROL (Creativity)
    # ==========================================
    print("\n1. TEMPERATURE CONTROL (Creativity Level)")
    print("-" * 50)
    
    # Very focused, deterministic (good for protocols)
    result = rag.query(
        question=question,
        temperature=0.1,  # Very focused
        response_style="clinical"
    )
    print(f"🎯 LOW TEMPERATURE (0.1) - Focused/Deterministic:")
    print(f"   {result['response'][:200]}...\n")
    
    # Balanced creativity (good for general advice)
    result = rag.query(
        question=question,
        temperature=0.7,  # Balanced
        response_style="clinical"
    )
    print(f"⚖️  MEDIUM TEMPERATURE (0.7) - Balanced:")
    print(f"   {result['response'][:200]}...\n")
    
    # High creativity (good for brainstorming)
    result = rag.query(
        question=question,
        temperature=1.3,  # Creative
        response_style="clinical"
    )
    print(f"🎨 HIGH TEMPERATURE (1.3) - Creative:")
    print(f"   {result['response'][:200]}...\n")
    
    # ==========================================
    # 2. RESPONSE LENGTH CONTROL
    # ==========================================
    print("\n2. RESPONSE LENGTH CONTROL")
    print("-" * 50)
    
    # Very short responses
    result = rag.query(
        question=question,
        max_tokens=50,  # Very short
        response_style="brief"
    )
    print(f"📝 SHORT (50 tokens): {result['response']}\n")
    
    # Medium responses
    result = rag.query(
        question=question,
        max_tokens=200,  # Medium
        response_style="practical"
    )
    print(f"📄 MEDIUM (200 tokens): {result['response'][:300]}...\n")
    
    # Long detailed responses
    result = rag.query(
        question=question,
        max_tokens=800,  # Long
        response_style="detailed"
    )
    print(f"📚 LONG (800 tokens): {result['response'][:400]}...\n")
    
    # ==========================================
    # 3. RESPONSE STYLE CONTROL
    # ==========================================
    print("\n3. RESPONSE STYLE CONTROL")
    print("-" * 50)
    
    styles_to_demo = ["brief", "clinical", "practical", "patient_friendly"]
    
    for style in styles_to_demo:
        result = rag.query(
            question=question,
            response_style=style,
            max_tokens=150
        )
        print(f"🎭 STYLE '{style.upper()}': {result['response'][:200]}...\n")
    
    # ==========================================
    # 4. CONTENT TYPE FILTERING
    # ==========================================
    print("\n4. CONTENT TYPE FILTERING")
    print("-" * 50)
    
    # Only use procedure/protocol content
    result = rag.query(
        question="How do I structure a CBT session?",
        content_type_filter="procedure",  # Only procedures
        response_style="protocol"
    )
    print(f"🔧 PROCEDURES ONLY: {result['response'][:300]}...\n")
    
    # Only use dialogue examples
    result = rag.query(
        question="Show me how to respond to patient resistance",
        content_type_filter="dialogue",  # Only dialogues
        response_style="practical"
    )
    print(f"💬 DIALOGUE ONLY: {result['response'][:300]}...\n")
    
    # ==========================================
    # 5. LANGUAGE CONTROL
    # ==========================================
    print("\n5. LANGUAGE CONTROL")
    print("-" * 50)
    
    # Force Spanish response
    result = rag.query(
        question="What is cognitive restructuring?",  # English question
        language="spanish",  # Force Spanish response
        response_style="educational"
    )
    print(f"🇪🇸 FORCE SPANISH: {result['response'][:200]}...\n")
    
    # Force English response
    result = rag.query(
        question="¿Qué es la reestructuración cognitiva?",  # Spanish question
        language="english",  # Force English response
        response_style="educational"
    )
    print(f"🇺🇸 FORCE ENGLISH: {result['response'][:200]}...\n")
    
    # ==========================================
    # 6. CUSTOM INSTRUCTIONS
    # ==========================================
    print("\n6. CUSTOM INSTRUCTIONS")
    print("-" * 50)
    
    result = rag.query(
        question="How do I treat social anxiety?",
        custom_instructions="""
        - Include at least 3 specific techniques
        - Mention potential challenges
        - Provide a timeline for treatment
        - Use bullet points for clarity
        """,
        response_style="clinical",
        max_tokens=600
    )
    print(f"📋 CUSTOM INSTRUCTIONS: {result['response'][:400]}...\n")
    
    # ==========================================
    # 7. ADVANCED GENERATION PARAMETERS
    # ==========================================
    print("\n7. ADVANCED GENERATION PARAMETERS")
    print("-" * 50)
    
    result = rag.query(
        question="Explain exposure therapy",
        temperature=0.6,
        max_tokens=400,
        top_p=0.8,        # Nucleus sampling
        top_k=30,         # Top-k sampling  
        repeat_penalty=1.2,  # Reduce repetition
        stop_sequences=["Patient:", "Therapist:", "Example:"]  # Stop at these
    )
    print(f"⚙️  ADVANCED PARAMS: {result['response'][:300]}...\n")


def clinical_scenario_examples():
    """Real clinical scenarios with optimal parameter settings."""
    
    rag = TherapyRAG()
    
    print("\n" + "="*80)
    print("CLINICAL SCENARIO EXAMPLES")
    print("="*80)
    
    scenarios = [
        {
            "scenario": "Quick consultation during supervision",
            "question": "Patient is avoiding homework. Quick advice?",
            "params": {
                "response_style": "brief",
                "temperature": 0.3,  # Focused
                "max_tokens": 100,   # Short
            }
        },
        {
            "scenario": "Detailed treatment planning meeting",
            "question": "Comprehensive CBT plan for GAD patient",
            "params": {
                "response_style": "detailed",
                "temperature": 0.5,  # Balanced
                "max_tokens": 800,   # Long
                "content_type_filter": "procedure"  # Focus on protocols
            }
        },
        {
            "scenario": "Training new therapist",
            "question": "Explain cognitive restructuring to a beginner",
            "params": {
                "response_style": "educational",
                "temperature": 0.7,  # More creative for teaching
                "max_tokens": 600,
                "custom_instructions": "Use simple examples and explain step-by-step"
            }
        },
        {
            "scenario": "Crisis intervention guidance",
            "question": "Patient having panic attack in session",
            "params": {
                "response_style": "protocol",
                "temperature": 0.2,  # Very focused
                "max_tokens": 300,
                "content_type_filter": "procedure"
            }
        },
        {
            "scenario": "Patient psychoeducation material",
            "question": "¿Qué es la ansiedad y por qué la tengo?",
            "params": {
                "response_style": "patient_friendly",
                "temperature": 0.6,
                "max_tokens": 400,
                "language": "spanish",
                "custom_instructions": "Use reassuring tone and simple language"
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. SCENARIO: {scenario['scenario'].upper()}")
        print(f"   Question: {scenario['question']}")
        print(f"   Parameters: {scenario['params']}")
        print("-" * 60)
        
        result = rag.query(scenario['question'], **scenario['params'])
        print(f"   Response: {result['response'][:400]}...")
        print(f"   Sources used: {result['context_chunks_used']}")
        print(f"   Config: {result['generation_config']}")


def parameter_reference_guide():
    """Complete reference guide for all parameters."""
    
    print("\n" + "="*80)
    print("COMPLETE PARAMETER REFERENCE GUIDE")
    print("="*80)
    
    guide = {
        "temperature": {
            "range": "0.0 - 2.0",
            "default": 0.7,
            "purpose": "Controls creativity/randomness",
            "examples": {
                "0.1-0.3": "Very focused, deterministic (protocols, facts)",
                "0.4-0.7": "Balanced creativity (general advice)",
                "0.8-1.2": "Creative, varied responses (teaching, brainstorming)",
                "1.3-2.0": "Very creative, experimental (research ideas)"
            }
        },
        "max_tokens": {
            "range": "1 - 4096",
            "default": 512,
            "purpose": "Maximum response length",
            "examples": {
                "50-100": "Brief answers, quick consultations",
                "200-400": "Standard clinical responses",
                "500-800": "Detailed explanations, treatment plans", 
                "800+": "Comprehensive guides, educational material"
            }
        },
        "response_style": {
            "options": [
                "brief", "clinical", "detailed", "practical", 
                "educational", "patient_friendly", "protocol", "troubleshooting"
            ],
            "purpose": "Controls response tone and structure",
            "examples": {
                "brief": "1-2 sentences, essential info only",
                "clinical": "Professional, evidence-based guidance",
                "practical": "Actionable steps and techniques",
                "patient_friendly": "Simple language, reassuring tone"
            }
        },
        "content_type_filter": {
            "options": ["dialogue", "procedure", "case_example", "conceptual"],
            "purpose": "Limit context to specific content types",
            "examples": {
                "procedure": "Treatment protocols, session structures",
                "dialogue": "Therapist-patient conversation examples",
                "case_example": "Clinical case studies",
                "conceptual": "Theory, definitions, research"
            }
        },
        "language": {
            "options": ["auto", "english", "spanish"],
            "purpose": "Control response language",
            "examples": {
                "auto": "Match query language",
                "english": "Force English responses",
                "spanish": "Force Spanish responses"
            }
        },
        "advanced_params": {
            "top_p": "0.0-1.0, nucleus sampling (0.9 default)",
            "top_k": "1-100, top-k sampling (40 default)", 
            "repeat_penalty": "1.0-2.0, reduce repetition (1.1 default)",
            "stop_sequences": "List of strings to stop generation"
        }
    }
    
    for param, info in guide.items():
        print(f"\n📌 {param.upper()}")
        print("-" * 40)
        if isinstance(info, dict):
            for key, value in info.items():
                print(f"   {key}: {value}")
        else:
            print(f"   {info}")


if __name__ == "__main__":
    demonstrate_parameter_control()
    clinical_scenario_examples()
    parameter_reference_guide()