#!/usr/bin/env python3
"""
Streamlit UI for Therapy RAG System
A comprehensive interface for psychotherapists to interact with the RAG knowledge base.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import logging
import traceback
from typing import Dict, Any

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our RAG systems
try:
    from therapy_rag import TherapyRAG, GenerationConfig
    from simple_rag import SimpleTherapyRAG
except ImportError as e:
    st.error(f"Error importing RAG modules: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG para Psicoterapeutas",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .response-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .source-info {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'advanced_rag_system' not in st.session_state:
    st.session_state.advanced_rag_system = None
if 'tfidf_rag_system' not in st.session_state:
    st.session_state.tfidf_rag_system = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'systems_initialized' not in st.session_state:
    st.session_state.systems_initialized = {'advanced': False, 'tfidf': False}
if 'ui_language' not in st.session_state:
    st.session_state.ui_language = 'spanish'  # Default to Spanish
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = ""

# UI Text Dictionary for bilingual support
UI_TEXT = {
    'english': {
        'title': '🧠 RAG System for Psychotherapists',
        'subtitle': 'Intelligent Assistant for Clinical Queries in CBT and Anxiety Disorders',
        'sidebar_header': '⚙️ System Configuration',
        'system_type_header': '🔧 System Type',
        'system_type_instruction': 'Select one or more systems to compare:',
        'base_llama': '💬 Base Llama (No RAG)',
        'base_llama_help': 'Llama model without access to clinical literature',
        'tfidf_rag': '📊 TF-IDF RAG',
        'tfidf_rag_help': 'RAG system using TF-IDF search',
        'advanced_rag': '🚀 Advanced RAG',
        'advanced_rag_help': 'RAG system with ChromaDB and semantic embeddings',
        'initialize_button': '🔄 Initialize Selected Systems',
        'initializing': 'Initializing systems...',
        'system_stats': '📊 System Statistics',
        'language_header': '🌐 Language Configuration',
        'response_language': 'Response Language:',
        'ui_language': 'Interface Language:',
        'response_params': '📝 Response Parameters',
        'response_style': 'Response Style:',
        'response_length': 'Response Length:',
        'complexity': 'Complexity/Creativity:',
        'complexity_help': '0.1 = Very conservative, 1.5 = Very creative',
        'tone': 'Response Tone:',
        'advanced_options': '🔧 Advanced Options',
        'context_chunks': 'Context Fragments:',
        'context_chunks_help': 'Number of relevant fragments to use',
        'content_filter': 'Filter by Content Type:',
        'all_types': 'All types',
        'no_systems_warning': '⚠️ Systems Not Initialized',
        'no_systems_message': 'Please select one or more systems in the sidebar and click "Initialize Selected Systems" to begin.',
        'clinical_query': '💬 Clinical Query',
        'query_label': 'Enter your clinical question:',
        'query_placeholder': 'Example: How do I help a patient with panic attacks who is not responding to breathing techniques?',
        'query_help': 'You can write in Spanish or English',
        'examples_label': '**Examples:**',
        'generate_button': '🔍 Generate Responses',
        'generating': 'Generating responses from multiple systems...',
        'conversation_history': '📝 Conversation History',
        'clear_history': '🗑️ Clear History',
        'question': 'Question',
        'response': 'Response',
        'sources': 'Sources',
        'config': 'Configuration',
        'systems_used': 'Systems Used',
        'fragments_used': 'fragments used',
        'footer_title': '🧠 RAG System for Psychotherapists - Developed to support evidence-based clinical practice',
        'footer_warning': '⚠️ This tool is for educational support. Always consult with appropriate clinical supervision.',
        'select_system_warning': '⚠️ Select at least one system',
        'init_success': '✅ initialized successfully',
        'init_error': '❌ Error initializing',
        'documents': 'Documents',
        'fragments': 'Fragments',
        'content_types': 'Content Types:',
        'relevance': 'Relevance',
        'similarity': 'Similarity',
        'confidence': 'Confidence',
        # Response style options
        'style_clinical': '👩‍⚕️ Clinical',
        'style_brief': '⚡ Brief',
        'style_detailed': '📚 Detailed',
        'style_practical': '🔨 Practical',
        'style_educational': '🎓 Educational',
        'style_patient_friendly': '😊 Patient-Friendly',
        'style_protocol': '📋 Protocol',
        'style_troubleshooting': '🔍 Problem-Solving',
        # Tone options
        'tone_professional': '💼 Professional',
        'tone_empathetic': '❤️ Empathetic',
        'tone_authoritative': '📖 Authoritative',
        'tone_collaborative': '🤝 Collaborative',
        # Language options
        'lang_auto': '🔄 Automatic',
        'lang_spanish': '🇪🇸 Spanish',
        'lang_english': '🇺🇸 English',
        # Example questions
        'example_1': 'What is CBT?',
        'example_2': 'Social anxiety techniques',
        'example_3': 'How to handle resistance',
        'example_4': 'Panic protocol',
        'example_5': 'Homework assignments',
        # Response display labels
        'question_label': 'Question:',
        'response_label': 'Response:',
        # System headers
        'base_llama_header': '💬 Base Llama (No RAG)',
        'tfidf_header': '📊 TF-IDF RAG',
        'advanced_rag_header': '🚀 Advanced RAG (ChromaDB)',
        # Source labels
        'sources_tfidf': 'TF-IDF Sources',
        'sources_advanced': 'Advanced RAG Sources',
        'fragments': 'fragments'
    },
    'spanish': {
        'title': '🧠 Sistema RAG para Psicoterapeutas',
        'subtitle': 'Asistente Inteligente para Consultas Clínicas en TCC y Trastornos de Ansiedad',
        'sidebar_header': '⚙️ Configuración del Sistema',
        'system_type_header': '🔧 Tipo de Sistema',
        'system_type_instruction': 'Selecciona uno o más sistemas para comparar:',
        'base_llama': '💬 Base Llama (Sin RAG)',
        'base_llama_help': 'Modelo Llama sin acceso a literatura clínica',
        'tfidf_rag': '📊 TF-IDF RAG',
        'tfidf_rag_help': 'Sistema RAG usando búsqueda TF-IDF',
        'advanced_rag': '🚀 RAG Avanzado',
        'advanced_rag_help': 'Sistema RAG con ChromaDB y embeddings semánticos',
        'initialize_button': '🔄 Inicializar Sistemas Seleccionados',
        'initializing': 'Inicializando sistemas...',
        'system_stats': '📊 Estadísticas de Sistemas',
        'language_header': '🌐 Configuración de Idioma',
        'response_language': 'Idioma de Respuesta:',
        'ui_language': 'Idioma de Interfaz:',
        'response_params': '📝 Parámetros de Respuesta',
        'response_style': 'Estilo de Respuesta:',
        'response_length': 'Longitud de Respuesta:',
        'complexity': 'Complejidad/Creatividad:',
        'complexity_help': '0.1 = Muy conservador, 1.5 = Muy creativo',
        'tone': 'Tono de Respuesta:',
        'advanced_options': '🔧 Opciones Avanzadas',
        'context_chunks': 'Fragmentos de Contexto:',
        'context_chunks_help': 'Número de fragmentos relevantes a usar',
        'content_filter': 'Filtrar por Tipo de Contenido:',
        'all_types': 'Todos los tipos',
        'no_systems_warning': '⚠️ Sistemas No Inicializados',
        'no_systems_message': 'Por favor, selecciona uno o más sistemas en la barra lateral y haz clic en "Inicializar Sistemas Seleccionados" para comenzar.',
        'clinical_query': '💬 Consulta Clínica',
        'query_label': 'Escribe tu pregunta clínica:',
        'query_placeholder': 'Ejemplo: ¿Cómo ayudo a un paciente con ataques de pánico que no responde a las técnicas de respiración?',
        'query_help': 'Puedes escribir en español o inglés',
        'examples_label': '**Ejemplos:**',
        'generate_button': '🔍 Generar Respuestas',
        'generating': 'Generando respuestas de múltiples sistemas...',
        'conversation_history': '📝 Historial de Conversación',
        'clear_history': '🗑️ Limpiar Historial',
        'question': 'Pregunta',
        'response': 'Respuesta',
        'sources': 'Fuentes',
        'config': 'Configuración',
        'systems_used': 'Sistemas Usados',
        'fragments_used': 'fragmentos utilizados',
        'footer_title': '🧠 Sistema RAG para Psicoterapeutas - Desarrollado para apoyar la práctica clínica basada en evidencia',
        'footer_warning': '⚠️ Esta herramienta es de apoyo educativo. Siempre consulta con supervisión clínica apropiada.',
        'select_system_warning': '⚠️ Selecciona al menos un sistema',
        'init_success': '✅ inicializado correctamente',
        'init_error': '❌ Error inicializando',
        'documents': 'Documentos',
        'fragments': 'Fragmentos',
        'content_types': 'Tipos de Contenido:',
        'relevance': 'Relevancia',
        'similarity': 'Similaridad',
        'confidence': 'Confianza',
        # Response style options
        'style_clinical': '👩‍⚕️ Clínico',
        'style_brief': '⚡ Breve',
        'style_detailed': '📚 Detallado',
        'style_practical': '🔨 Práctico',
        'style_educational': '🎓 Educativo',
        'style_patient_friendly': '😊 Para Pacientes',
        'style_protocol': '📋 Protocolo',
        'style_troubleshooting': '🔍 Resolución de Problemas',
        # Tone options
        'tone_professional': '💼 Profesional',
        'tone_empathetic': '❤️ Empático',
        'tone_authoritative': '📖 Autoritativo',
        'tone_collaborative': '🤝 Colaborativo',
        # Language options
        'lang_auto': '🔄 Automático',
        'lang_spanish': '🇪🇸 Español',
        'lang_english': '🇺🇸 English',
        # Example questions
        'example_1': '¿Qué es la TCC?',
        'example_2': 'Técnicas para ansiedad social',
        'example_3': 'Cómo manejar resistencia',
        'example_4': 'Protocolo para pánico',
        'example_5': 'Tareas para casa',
        # Response display labels
        'question_label': 'Pregunta:',
        'response_label': 'Respuesta:',
        # System headers
        'base_llama_header': '💬 Base Llama (Sin RAG)',
        'tfidf_header': '📊 TF-IDF RAG',
        'advanced_rag_header': '🚀 RAG Avanzado (ChromaDB)',
        # Source labels
        'sources_tfidf': 'Fuentes TF-IDF',
        'sources_advanced': 'Fuentes RAG Avanzado',
        'fragments': 'fragmentos'
    }
}

def get_text(key: str) -> str:
    """Get UI text in current language."""
    return UI_TEXT[st.session_state.ui_language].get(key, key)

def initialize_rag_system(system_type: str) -> bool:
    """Initialize the selected RAG system."""
    try:
        if system_type == "advanced":
            if st.session_state.advanced_rag_system is None:
                st.session_state.advanced_rag_system = TherapyRAG()
            st.session_state.systems_initialized['advanced'] = True
        elif system_type == "tfidf":
            if st.session_state.tfidf_rag_system is None:
                st.session_state.tfidf_rag_system = SimpleTherapyRAG()
            st.session_state.systems_initialized['tfidf'] = True
        else:
            st.error("Tipo de sistema no válido")
            return False
        
        return True
    except Exception as e:
        st.error(f"Error inicializando el sistema {system_type}: {e}")
        logger.error(f"RAG initialization error for {system_type}: {traceback.format_exc()}")
        return False

def get_system_stats(system_type: str) -> Dict:
    """Get statistics about the specified RAG system."""
    system = None
    if system_type == "advanced" and st.session_state.advanced_rag_system is not None:
        system = st.session_state.advanced_rag_system
    elif system_type == "tfidf" and st.session_state.tfidf_rag_system is not None:
        system = st.session_state.tfidf_rag_system
    
    if system is None:
        return {}
    
    try:
        if hasattr(system, 'get_collection_stats'):
            return system.get_collection_stats()
        elif hasattr(system, 'get_stats'):
            return system.get_stats()
        else:
            return {}
    except Exception as e:
        logger.error(f"Error getting stats for {system_type}: {e}")
        return {}

def generate_base_llama_response(question: str, complexity: float, response_length: int, 
                                custom_instructions: str, language: str) -> str:
    """Generate a response using only the base Llama model without RAG."""
    try:
        # Use any available system to access the generate_response method
        system = st.session_state.advanced_rag_system or st.session_state.tfidf_rag_system
        if system is None:
            return "Error: No hay sistemas inicializados para generar respuesta base"
        
        # Add length guidance based on token limit
        if response_length <= 128:
            length_instruction = "Provide a BRIEF, CONCISE response in 1-2 sentences maximum. Get straight to the point."
        elif response_length <= 256:
            length_instruction = "Provide a SHORT response in 2-3 sentences. Be direct and focused."
        elif response_length <= 512:
            length_instruction = "Provide a MODERATE response with key points. Be comprehensive but concise."
        else:
            length_instruction = "Provide a DETAILED, comprehensive response."
        
        # Force English prompt and strong English instruction when English is selected
        if language.lower() in ["english", "en"]:
            base_prompt = f"""You are an expert clinical psychologist specializing in CBT and anxiety disorders. 

IMPORTANT: You must respond in English only. Do not use Spanish.
{length_instruction}

Answer the following question based solely on your general knowledge, without using any external literature:

Question: {question}

Instructions:
{custom_instructions}
YOU MUST RESPOND IN ENGLISH ONLY.

Response in English:"""
        
        elif language.lower() in ["spanish", "es", "español"]:
            # Spanish length instructions
            if response_length <= 128:
                length_instruction_es = "Proporciona una respuesta BREVE y CONCISA en 1-2 oraciones máximo. Ve directo al punto."
            elif response_length <= 256:
                length_instruction_es = "Proporciona una respuesta CORTA en 2-3 oraciones. Sé directo y enfocado."
            elif response_length <= 512:
                length_instruction_es = "Proporciona una respuesta MODERADA con puntos clave. Sé completo pero conciso."
            else:
                length_instruction_es = "Proporciona una respuesta DETALLADA y completa."
            
            base_prompt = f"""Eres un psicólogo clínico experto en TCC y trastornos de ansiedad. 

IMPORTANTE: Debes responder únicamente en español.
{length_instruction_es}

Responde la siguiente pregunta basándote únicamente en tu conocimiento general, sin usar ninguna literatura externa:

Pregunta: {question}

Instrucciones:
{custom_instructions}
DEBES RESPONDER ÚNICAMENTE EN ESPAÑOL.

Respuesta en español:"""
        
        else:  # auto mode - detect from question
            if any(word in question.lower() for word in ["how", "what", "when", "where", "why", "can", "do", "does", "is", "are", "the", "and", "or", "but", "my", "patient", "has"]):
                base_prompt = f"""You are an expert clinical psychologist specializing in CBT and anxiety disorders. 
{length_instruction}

Answer the following question based solely on your general knowledge, without using any external literature:

Question: {question}

Instructions:
{custom_instructions}

Response:"""
            else:
                # Spanish length instructions for auto mode
                if response_length <= 128:
                    length_instruction_es = "Proporciona una respuesta BREVE y CONCISA en 1-2 oraciones máximo."
                elif response_length <= 256:
                    length_instruction_es = "Proporciona una respuesta CORTA en 2-3 oraciones."
                elif response_length <= 512:
                    length_instruction_es = "Proporciona una respuesta MODERADA con puntos clave."
                else:
                    length_instruction_es = "Proporciona una respuesta DETALLADA y completa."
                    
                base_prompt = f"""Eres un psicólogo clínico experto en TCC y trastornos de ansiedad.
{length_instruction_es}

Responde la siguiente pregunta basándote únicamente en tu conocimiento general, sin usar ninguna literatura externa:

Pregunta: {question}

Instrucciones:
{custom_instructions}

Respuesta:"""
        
        base_config = GenerationConfig(
            temperature=complexity,
            max_tokens=response_length
        )
        
        return system.generate_response(base_prompt, base_config)
    except Exception as e:
        logger.error(f"Error generating base response: {e}")
        return f"Error generando respuesta base: {e}"

def main():
    # Header with dynamic language
    st.markdown(f"""
    <div class="main-header">
        <h1>{get_text('title')}</h1>
        <p>{get_text('subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for system configuration
    with st.sidebar:
        st.header(get_text('sidebar_header'))
        
        # UI Language Selector - this changes the interface language
        st.subheader(get_text('language_header'))
        ui_language = st.selectbox(
            get_text('ui_language'),
            options=["spanish", "english"],
            format_func=lambda x: {
                "spanish": "🇪🇸 Español",
                "english": "🇺🇸 English"
            }[x],
            index=0 if st.session_state.ui_language == "spanish" else 1
        )
        
        # Update UI language if changed
        if ui_language != st.session_state.ui_language:
            st.session_state.ui_language = ui_language
            # Use compatibility approach for different Streamlit versions
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        
        st.divider()
        
        # RAG System Selection with checkboxes
        st.subheader(get_text('system_type_header'))
        st.write(get_text('system_type_instruction'))
        
        use_base_llama = st.checkbox(
            get_text('base_llama'),
            help=get_text('base_llama_help')
        )
        
        use_tfidf = st.checkbox(
            get_text('tfidf_rag'),
            help=get_text('tfidf_rag_help')
        )
        
        use_advanced = st.checkbox(
            get_text('advanced_rag'),
            help=get_text('advanced_rag_help')
        )
        
        # Initialize selected systems
        if st.button(get_text('initialize_button'), type="primary"):
            with st.spinner(get_text('initializing')):
                success_count = 0
                total_selected = sum([use_tfidf, use_advanced])
                
                if use_tfidf:
                    if initialize_rag_system("tfidf"):
                        success_count += 1
                        st.success(f"✅ TF-IDF RAG {get_text('init_success')}")
                    else:
                        st.error(f"❌ {get_text('init_error')} TF-IDF RAG")
                
                if use_advanced:
                    if initialize_rag_system("advanced"):
                        success_count += 1
                        st.success(f"✅ RAG Avanzado {get_text('init_success')}")
                    else:
                        st.error(f"❌ {get_text('init_error')} RAG Avanzado")
                
                if total_selected == 0 and not use_base_llama:
                    st.warning(get_text('select_system_warning'))
                elif success_count == total_selected and total_selected > 0:
                    st.success(f"✅ {success_count} sistema(s) {get_text('init_success')}")
        
        # Show system stats if initialized
        st.subheader(get_text('system_stats'))
        
        if st.session_state.systems_initialized['advanced']:
            with st.expander("🚀 RAG Avanzado"):
                stats = get_system_stats('advanced')
                if stats:
                    if 'total_documents' in stats:
                        st.metric(get_text('documents'), stats['total_documents'])
                    if 'content_types' in stats:
                        st.write(f"**{get_text('content_types')}**")
                        for ct, count in stats['content_types'].items():
                            st.write(f"• {ct}: {count}")
        
        if st.session_state.systems_initialized['tfidf']:
            with st.expander("📊 TF-IDF RAG"):
                stats = get_system_stats('tfidf')
                if stats:
                    if 'total_chunks' in stats:
                        st.metric(get_text('fragments'), stats['total_chunks'])
                    if 'content_types' in stats:
                        st.write(f"**{get_text('content_types')}**")
                        for ct, count in stats['content_types'].items():
                            st.write(f"• {ct}: {count}")
        
        st.divider()
        
        # Response Language Selection (separate from UI language)
        language = st.selectbox(
            get_text('response_language'),
            options=["auto", "spanish", "english"],
            format_func=lambda x: {
                "auto": get_text('lang_auto'),
                "spanish": get_text('lang_spanish'),
                "english": get_text('lang_english')
            }[x]
        )
        
        # Response Parameters
        st.subheader(get_text('response_params'))
        
        response_style = st.selectbox(
            get_text('response_style'),
            options=["clinical", "brief", "detailed", "practical", "educational", 
                    "patient_friendly", "protocol", "troubleshooting"],
            format_func=lambda x: {
                "clinical": get_text('style_clinical'),
                "brief": get_text('style_brief'),
                "detailed": get_text('style_detailed'),
                "practical": get_text('style_practical'),
                "educational": get_text('style_educational'),
                "patient_friendly": get_text('style_patient_friendly'),
                "protocol": get_text('style_protocol'),
                "troubleshooting": get_text('style_troubleshooting')
            }[x]
        )
        
        response_length = st.select_slider(
            get_text('response_length'),
            options=[128, 256, 512, 768, 1024],
            value=512,
            format_func=lambda x: f"{x} tokens"
        )
        
        complexity = st.slider(
            get_text('complexity'),
            min_value=0.1,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help=get_text('complexity_help')
        )
        
        tone = st.selectbox(
            get_text('tone'),
            options=["professional", "empathetic", "authoritative", "collaborative"],
            format_func=lambda x: {
                "professional": get_text('tone_professional'),
                "empathetic": get_text('tone_empathetic'),
                "authoritative": get_text('tone_authoritative'),
                "collaborative": get_text('tone_collaborative')
            }[x]
        )
        
        # Advanced Options
        with st.expander(get_text('advanced_options')):
            n_context_chunks = st.slider(
                get_text('context_chunks'),
                min_value=1,
                max_value=10,
                value=5,
                help=get_text('context_chunks_help')
            )
            
            content_type_filter = st.selectbox(
                get_text('content_filter'),
                options=[None, "dialogue", "procedure", "theory", "case_study"],
                format_func=lambda x: get_text('all_types') if x is None else x.title()
            )
    
    # Main content area
    any_system_ready = (use_base_llama or 
                       st.session_state.systems_initialized['advanced'] or 
                       st.session_state.systems_initialized['tfidf'])
    
    if not any_system_ready:
        st.markdown(f"""
        <div class="warning-box">
            <h3>{get_text('no_systems_warning')}</h3>
            <p>{get_text('no_systems_message')}</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Input section
    st.subheader(get_text('clinical_query'))
    
    # Question input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Handle example selection
        default_value = st.session_state.selected_example if st.session_state.selected_example else ""
        
        question = st.text_area(
            get_text('query_label'),
            value=default_value,
            placeholder=get_text('query_placeholder'),
            height=100,
            help=get_text('query_help')
        )
    
    with col2:
        st.write(get_text('examples_label'))
        example_questions = [
            get_text('example_1'),
            get_text('example_2'),
            get_text('example_3'),
            get_text('example_4'),
            get_text('example_5')
        ]
        
        for i, example in enumerate(example_questions):
            if st.button(example, key=f"example_{i}_{st.session_state.ui_language}"):
                st.session_state.selected_example = example
                st.rerun()
    
    # Generate response
    if st.button(get_text('generate_button'), type="primary", disabled=not question.strip()):
        if question.strip():
            # Clear selected example after generating response
            st.session_state.selected_example = ""
            
            with st.spinner(get_text('generating')):
                try:
                    # Map tone to custom instructions
                    tone_instructions = {
                        "professional": "Mantén un tono profesional y formal.",
                        "empathetic": "Usa un tono empático y comprensivo.",
                        "authoritative": "Presenta la información con autoridad y confianza.",
                        "collaborative": "Usa un tono colaborativo como si fueras un colega."
                    }
                    
                    custom_instructions = tone_instructions.get(tone, "")
                    responses = {}
                    
                    # Generate Base Llama response if selected
                    if use_base_llama:
                        st.markdown(f"### {get_text('base_llama_header')}")
                        base_response = generate_base_llama_response(
                            question, complexity, response_length, custom_instructions, language
                        )
                        responses['base'] = base_response
                        
                        st.markdown(f"""
                        <div class="response-container">
                            <strong>{get_text('question_label')}</strong> {question}<br><br>
                            <strong>{get_text('response_label')}</strong><br>
                            {base_response}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Generate TF-IDF RAG response if selected and initialized
                    if use_tfidf and st.session_state.systems_initialized['tfidf']:
                        st.markdown(f"### {get_text('tfidf_header')}")
                        tfidf_result = st.session_state.tfidf_rag_system.query(
                            question=question,
                            response_style=response_style,
                            temperature=complexity,
                            max_tokens=response_length,
                            n_context_chunks=n_context_chunks,
                            content_type_filter=content_type_filter,
                            language=language,
                            custom_instructions=custom_instructions
                        )
                        responses['tfidf'] = tfidf_result
                        
                        st.markdown(f"""
                        <div class="response-container">
                            <strong>{get_text('question_label')}</strong> {question}<br><br>
                            <strong>{get_text('response_label')}</strong><br>
                            {tfidf_result['response']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show TF-IDF sources
                        if 'context_sources' in tfidf_result:
                            with st.expander(f"📚 {get_text('sources_tfidf')} ({len(tfidf_result['context_sources'])} {get_text('fragments')})"):
                                for i, source in enumerate(tfidf_result['context_sources'], 1):
                                    confidence = source.get('similarity', 0)
                                    st.markdown(f"""
                                    <div class="source-info">
                                        <strong>Fuente {i}:</strong> {source.get('content_type', 'N/A')} - {source.get('section', 'N/A')}<br>
                                        <strong>Similaridad:</strong> {confidence:.3f}
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Generate Advanced RAG response if selected and initialized
                    if use_advanced and st.session_state.systems_initialized['advanced']:
                        st.markdown(f"### {get_text('advanced_rag_header')}")
                        advanced_result = st.session_state.advanced_rag_system.query(
                            question=question,
                            response_style=response_style,
                            temperature=complexity,
                            max_tokens=response_length,
                            n_context_chunks=n_context_chunks,
                            content_type_filter=content_type_filter,
                            language=language,
                            custom_instructions=custom_instructions
                        )
                        responses['advanced'] = advanced_result
                        
                        st.markdown(f"""
                        <div class="response-container">
                            <strong>{get_text('question_label')}</strong> {question}<br><br>
                            <strong>{get_text('response_label')}</strong><br>
                            {advanced_result['response']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show Advanced RAG sources
                        if 'context_sources' in advanced_result:
                            with st.expander(f"📚 {get_text('sources_advanced')} ({len(advanced_result['context_sources'])} {get_text('fragments')})"):
                                for i, source in enumerate(advanced_result['context_sources'], 1):
                                    confidence = source.get('confidence', 0)
                                    st.markdown(f"""
                                    <div class="source-info">
                                        <strong>Fuente {i}:</strong> {source.get('content_type', 'N/A')} - {source.get('section', 'N/A')}<br>
                                        <strong>Confianza:</strong> {confidence:.3f}
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Add to conversation history with all responses
                    if responses:
                        st.session_state.conversation_history.append({
                            'question': question,
                            'responses': responses,
                            'systems_used': list(responses.keys()),
                            'config': {
                                'temperature': complexity,
                                'max_tokens': response_length,
                                'style': response_style
                            }
                        })
                    
                except Exception as e:
                    st.error(f"Error generando respuestas: {e}")
                    logger.error(f"Query error: {traceback.format_exc()}")
    
    # Conversation History
    if st.session_state.conversation_history:
        st.divider()
        st.subheader(get_text('conversation_history'))
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(get_text('clear_history')):
                st.session_state.conversation_history = []
                # Use compatibility approach for different Streamlit versions
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
            with st.expander(f"Consulta {len(st.session_state.conversation_history) - i + 1}: {conv['question'][:50]}..."):
                st.write(f"**{get_text('question')}:** {conv['question']}")
                
                # Handle new multi-system format
                if 'responses' in conv:
                    st.write(f"**Sistemas Usados:** {', '.join(conv['systems_used'])}")
                    for system, response in conv['responses'].items():
                        system_names = {
                            'base': '💬 Base Llama',
                            'tfidf': '📊 TF-IDF RAG',
                            'advanced': '🚀 RAG Avanzado'
                        }
                        st.write(f"**{system_names.get(system, system)}:**")
                        if isinstance(response, dict):
                            st.write(f"  {response.get('response', response)}")
                            if 'context_sources' in response:
                                st.write(f"  Fuentes: {len(response['context_sources'])} fragmentos")
                        else:
                            st.write(f"  {response}")
                # Handle old single-system format
                else:
                    st.write(f"**{get_text('response')}:** {conv.get('response', 'N/A')}")
                    st.write(f"**Fuentes:** {conv.get('sources_count', 0)} fragmentos utilizados")
                
                if conv.get('config'):
                    st.write(f"**Configuración:** Temp: {conv['config'].get('temperature', 'N/A')}, "
                           f"Tokens: {conv['config'].get('max_tokens', 'N/A')}, "
                           f"Estilo: {conv['config'].get('style', 'N/A')}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>{get_text('footer_title')}</p>
    <p>{get_text('footer_warning')}</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()