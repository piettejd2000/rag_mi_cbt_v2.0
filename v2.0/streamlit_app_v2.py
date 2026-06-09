#!/usr/bin/env python3
"""
RAG v2.0 Streamlit Interface with Adaptive Grounding
Enhanced UI showing routing decisions and performance metrics
"""

import streamlit as st
import anthropic
import os
import logging
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add paths for v2.0 components
current_dir = Path(__file__).parent
rag_root = current_dir.parent
sys.path.append(str(rag_root / 'chunking_pys'))
sys.path.append(str(current_dir))

try:
    from enhanced_therapy_rag_v2 import EnhancedTherapyRAGv2, CompatibilityWrapper, create_enhanced_rag_v2
except ImportError:
    st.error("⚠️ v2.0 components not found. Please ensure all files are properly installed.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG v2.0 para Psicoterapeutas", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for v2.0 styling
st.markdown("""
<style>
.version-indicator {
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
    z-index: 1000;
    font-family: monospace;
}
.v2-header {
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 1rem;
}
.routing-info {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #2196F3;
}
.path-a { border-left-color: #4CAF50; }
.path-b { border-left-color: #FF9800; }
.path-c { border-left-color: #9C27B0; }
.metrics-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for v2.0
if 'v2_system' not in st.session_state:
    st.session_state.v2_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'last_routing_decision' not in st.session_state:
    st.session_state.last_routing_decision = None
if 'conversation_history_v2' not in st.session_state:
    st.session_state.conversation_history_v2 = []
if 'show_v1_comparison' not in st.session_state:
    st.session_state.show_v1_comparison = False

# UI Text for bilingual support
UI_TEXT = {
    'spanish': {
        'title': '🧠 RAG v2.0 - Sistema de Fundamentación Adaptativa',
        'subtitle': 'Inteligencia Artificial Avanzada para Consultas Clínicas (MI/CBT)',
        'version_info': '🆕 Versión 2.0 - Elimina regresión en conocimiento canónico (-56.9% → ~0%)',
        'system_status': 'Estado del Sistema',
        'system_ready': '✅ RAG v2.0 Listo',
        'system_loading': '⏳ Inicializando RAG v2.0...',
        'routing_analysis': 'Análisis de Enrutamiento',
        'routing_path': 'Ruta Seleccionada',
        'routing_confidence': 'Confianza',
        'routing_explanation': 'Explicación',
        'knowledge_type': 'Tipo de Conocimiento',
        'relevance_assessment': 'Evaluación de Relevancia',
        'performance_metrics': 'Métricas de Rendimiento',
        'query_input': 'Consulta Clínica',
        'query_placeholder': 'Ej: ¿Qué significa OARS en Entrevista Motivacional?',
        'generate_response': '🔍 Generar Respuesta v2.0',
        'generating': 'Generando respuesta con enrutamiento adaptativo...',
        'conversation_history': 'Historial de Conversación',
        'clear_history': '🗑️ Limpiar Historial',
        'v1_comparison': 'Comparación v1.0',
        'show_v1': 'Mostrar comparación v1.0',
        'path_a_desc': 'Fundamentación Fuerte - Usa recuperación relevante',
        'path_b_desc': 'Fundamentación Suave - Mezcla recuperación con conocimiento',  
        'path_c_desc': 'Conocimiento Primero - Usa conocimiento clínico establecido',
        'total_queries': 'Total de Consultas',
        'path_distribution': 'Distribución de Rutas',
        'avg_confidence': 'Confianza Promedio',
        'avg_response_time': 'Tiempo de Respuesta Promedio',
        'examples_header': 'Ejemplos Rápidos:',
        'example_oars': 'OARS en MI',
        'example_cbt': '¿Qué es CBT?', 
        'example_reflection': 'Técnicas de reflexión',
        'example_medication': '¿Qué medicamentos prescribir?',
        'v2_features': 'Características v2.0',
        'feature_routing': '🎯 Enrutamiento Inteligente',
        'feature_performance': '📊 Métricas en Tiempo Real', 
        'feature_compatibility': '🔄 Compatibilidad v1.0',
        'footer_warning': '⚠️ Herramienta de apoyo educativo. Consulte supervisión clínica apropiada.',
    },
    'english': {
        'title': '🧠 RAG v2.0 - Adaptive Grounding System',
        'subtitle': 'Advanced AI for Clinical Queries (MI/CBT)',
        'version_info': '🆕 Version 2.0 - Eliminates canonical knowledge regression (-56.9% → ~0%)',
        'system_status': 'System Status',
        'system_ready': '✅ RAG v2.0 Ready',
        'system_loading': '⏳ Initializing RAG v2.0...',
        'routing_analysis': 'Routing Analysis',
        'routing_path': 'Selected Path',
        'routing_confidence': 'Confidence',
        'routing_explanation': 'Explanation',
        'knowledge_type': 'Knowledge Type',
        'relevance_assessment': 'Relevance Assessment',
        'performance_metrics': 'Performance Metrics',
        'query_input': 'Clinical Query',
        'query_placeholder': 'E.g.: What does OARS stand for in Motivational Interviewing?',
        'generate_response': '🔍 Generate v2.0 Response',
        'generating': 'Generating response with adaptive routing...',
        'conversation_history': 'Conversation History',
        'clear_history': '🗑️ Clear History',
        'v1_comparison': 'v1.0 Comparison',
        'show_v1': 'Show v1.0 comparison',
        'path_a_desc': 'Strong Grounding - Uses relevant retrieval',
        'path_b_desc': 'Soft Grounding - Blends retrieval with knowledge',
        'path_c_desc': 'Knowledge First - Uses established clinical knowledge',
        'total_queries': 'Total Queries',
        'path_distribution': 'Path Distribution',
        'avg_confidence': 'Average Confidence',
        'avg_response_time': 'Average Response Time',
        'examples_header': 'Quick Examples:',
        'example_oars': 'OARS in MI',
        'example_cbt': 'What is CBT?',
        'example_reflection': 'Reflection techniques',
        'example_medication': 'What medications to prescribe?',
        'v2_features': 'v2.0 Features',
        'feature_routing': '🎯 Smart Routing',
        'feature_performance': '📊 Real-time Metrics',
        'feature_compatibility': '🔄 v1.0 Compatibility',
        'footer_warning': '⚠️ Educational support tool. Consult appropriate clinical supervision.',
    }
}

# Language selection
lang = st.sidebar.selectbox(
    "🌐 Language / Idioma", 
    options=['spanish', 'english'],
    index=0
)
ui = UI_TEXT[lang]

# Version indicator in upper left corner
st.markdown('<div class="version-indicator">RAG v2.0</div>', unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="v2-header">
    <h1>{ui['title']}</h1>
    <p>{ui['subtitle']}</p>
    <p><small>{ui['version_info']}</small></p>
</div>
""", unsafe_allow_html=True)

# Sidebar - System Configuration
with st.sidebar:
    st.header("⚙️ " + ui['system_status'])
    
    # Initialize system
    if not st.session_state.system_initialized:
        if st.button("🚀 Initialize RAG v2.0 System"):
            with st.spinner(ui['system_loading']):
                try:
                    api_key = st.secrets.get("ANTHROPIC_API_KEY")
                    if not api_key:
                        st.error("❌ API key not found in secrets")
                    else:
                        st.session_state.v2_system = create_enhanced_rag_v2(api_key=api_key)
                        st.session_state.system_initialized = True
                        st.success("✅ RAG v2.0 initialized successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Initialization failed: {e}")
    else:
        st.success(ui['system_ready'])
        
        # v2.0 Features
        st.subheader(ui['v2_features'])
        st.markdown(f"""
        - {ui['feature_routing']}
        - {ui['feature_performance']}
        - {ui['feature_compatibility']}
        """)
        
        # Performance Metrics
        if st.session_state.v2_system:
            stats = st.session_state.v2_system.get_routing_statistics()
            if stats.get('total_queries', 0) > 0:
                st.subheader("📊 " + ui['performance_metrics'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(ui['total_queries'], stats['total_queries'])
                    st.metric(ui['avg_confidence'], f"{stats['average_confidence']:.1%}")
                with col2:
                    st.metric(ui['avg_response_time'], f"{stats['average_response_time']:.2f}s")
                
                # Path distribution
                st.subheader(ui['path_distribution'])
                for path, percentage in stats['path_percentages'].items():
                    st.progress(percentage / 100, text=f"{path}: {percentage:.1f}%")
        
        # v1.0 Comparison toggle
        st.session_state.show_v1_comparison = st.checkbox(
            ui['show_v1'], 
            value=st.session_state.show_v1_comparison
        )
        
        # Clear metrics button
        if st.session_state.system_initialized:
            if st.button("🔄 Reset Metrics"):
                st.session_state.v2_system.reset_metrics()
                st.success("✅ Metrics reset")

# Main interface
if not st.session_state.system_initialized:
    st.warning("⚠️ Please initialize the RAG v2.0 system in the sidebar to begin.")
else:
    # Example questions
    st.subheader(ui['examples_header'])
    col1, col2, col3, col4 = st.columns(4)
    
    examples = [
        (ui['example_oars'], "¿Qué significa OARS en Entrevista Motivacional?"),
        (ui['example_cbt'], "¿Qué es la Terapia Cognitivo Conductual?"),
        (ui['example_reflection'], "¿Cómo uso técnicas de escucha reflexiva?"),
        (ui['example_medication'], "¿Qué medicamentos debo prescribir para la depresión?")
    ]
    
    for i, (col, (label, question)) in enumerate(zip([col1, col2, col3, col4], examples)):
        with col:
            if st.button(label, key=f"example_{i}"):
                st.session_state.current_question = question
    
    # Query input
    st.subheader("💬 " + ui['query_input'])
    question = st.text_area(
        ui['query_input'],
        value=getattr(st.session_state, 'current_question', ''),
        placeholder=ui['query_placeholder'],
        height=100,
        label_visibility="collapsed"
    )
    
    # Generate response
    if st.button(ui['generate_response'], disabled=not question.strip()):
        if question.strip():
            with st.spinner(ui['generating']):
                try:
                    # Generate v2.0 response
                    start_time = time.time()
                    v2_result = st.session_state.v2_system.generate_response_v2(question)
                    v2_time = time.time() - start_time
                    
                    st.session_state.last_routing_decision = v2_result['routing_decision']
                    
                    # Add to conversation history
                    st.session_state.conversation_history_v2.append({
                        'question': question,
                        'v2_response': v2_result['response'],
                        'routing': v2_result['routing_decision'],
                        'timestamp': time.strftime('%H:%M:%S'),
                        'response_time': v2_time
                    })
                    
                    # Display results
                    st.success("✅ Response generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")
                    logger.error(f"Response generation error: {e}")
    
    # Display latest conversation
    if st.session_state.conversation_history_v2:
        latest = st.session_state.conversation_history_v2[-1]
        
        # Question
        st.subheader("❓ Question")
        st.write(latest['question'])
        
        # Routing Analysis
        routing = latest['routing']
        st.subheader("🎯 " + ui['routing_analysis'])
        
        path_colors = {
            'PATH_A': 'path-a',
            'PATH_B': 'path-b', 
            'PATH_C': 'path-c'
        }
        
        path_class = path_colors.get(routing['path'], 'routing-info')
        
        st.markdown(f"""
        <div class="routing-info {path_class}">
            <strong>{ui['routing_path']}:</strong> {routing['path']}<br>
            <strong>{ui['routing_confidence']}:</strong> {routing['confidence']:.1%}<br>
            <strong>{ui['routing_explanation']}:</strong> {routing['explanation']}<br>
            <strong>{ui['knowledge_type']}:</strong> {routing['knowledge_type']['knowledge_type']}<br>
            <strong>{ui['relevance_assessment']}:</strong> {routing['relevance']['relevance'] if routing['relevance'] else 'N/A'}
        </div>
        """, unsafe_allow_html=True)
        
        # Path descriptions
        path_descriptions = {
            'PATH_A': ui['path_a_desc'],
            'PATH_B': ui['path_b_desc'],
            'PATH_C': ui['path_c_desc']
        }
        
        st.info(f"**{routing['path']}**: {path_descriptions[routing['path']]}")
        
        # Response
        st.subheader("💬 RAG v2.0 Response")
        st.write(latest['v2_response'])
        
        # Metadata
        st.caption(f"⏱️ Response time: {latest['response_time']:.2f}s | 🕒 {latest['timestamp']}")
    
    # Conversation History
    if st.session_state.conversation_history_v2:
        st.subheader("📝 " + ui['conversation_history'])
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(ui['clear_history']):
                st.session_state.conversation_history_v2 = []
                st.session_state.last_routing_decision = None
                st.rerun()
        
        # Display conversation history (newest first)
        for i, entry in enumerate(reversed(st.session_state.conversation_history_v2)):
            if i >= 5:  # Limit display to last 5 conversations
                break
                
            with st.expander(f"🕒 {entry['timestamp']} - {entry['question'][:50]}..."):
                st.write(f"**Question:** {entry['question']}")
                st.write(f"**Response:** {entry['v2_response']}")
                st.write(f"**Path:** {entry['routing']['path']} (Confidence: {entry['routing']['confidence']:.1%})")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666;">
    <p>{ui['footer_warning']}</p>
    <p><strong>RAG v2.0</strong> - Adaptive Grounding System | 
    Eliminates canonical knowledge regression while preserving all benefits</p>
</div>
""", unsafe_allow_html=True)