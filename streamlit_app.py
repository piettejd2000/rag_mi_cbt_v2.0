#!/usr/bin/env python3
"""
RAG v2.0 Production Deployment with Complete UI
Streamlit Cloud Entry Point for https://herramientasaludmental.streamlit.app  
Enhanced with v2.0 adaptive grounding system + rich v1.0 UI features
"""

import streamlit as st
import anthropic
import os
import logging
import time
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths for v2.0 components
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'v2.0'))
sys.path.append(str(current_dir / 'chunking_pys'))

# Import v2.0 components
try:
    from enhanced_therapy_rag_v2 import create_enhanced_rag_v2
    v2_available = True
    logger.info("✅ v2.0 backend components loaded successfully")
except ImportError as import_error:
    v2_available = False
    logger.warning(f"⚠️ v2.0 components not available: {import_error}")
    
    # Define a fallback function so the code doesn't crash
    def create_enhanced_rag_v2(**kwargs):
        raise ImportError(f"v2.0 system not available: {import_error}")

# v2.0 imports are already configured above

# Page configuration - detect if mobile for sidebar state  
import platform
is_mobile = False
try:
    # Simple mobile detection based on common patterns
    user_agent = st.get_query_params().get('mobile', ['false'])[0]
    is_mobile = user_agent.lower() == 'true'
except:
    pass

st.set_page_config(
    page_title="RAG v2.0 para Psicoterapeutas", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed" if is_mobile else "expanded"
)

# Initialize session state
if 'ui_language' not in st.session_state:
    st.session_state.ui_language = 'spanish'
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = ""
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'systems_initialized' not in st.session_state:
    st.session_state.systems_initialized = {'enhanced_v2': False, 'enhanced_v1': False, 'base_claude': False}
if 'enhanced_v2_system' not in st.session_state:
    st.session_state.enhanced_v2_system = None
if 'enhanced_v1_system' not in st.session_state:
    st.session_state.enhanced_v1_system = None
if 'base_claude_system' not in st.session_state:
    st.session_state.base_claude_system = None

# UI Text Dictionary for bilingual support
UI_TEXT = {
    'english': {
        'title': '🧠 RAG v2.0 System for Psychotherapists',
        'subtitle': 'Advanced Intelligent Assistant with Adaptive Grounding System',
        'sidebar_header': '⚙️ System Configuration',
        'system_type_header': '🔧 System Type',
        'system_type_instruction': 'Select one or more systems to compare:',
        'base_claude': '💬 Base Claude (No RAG)',
        'base_claude_help': 'Claude model without access to clinical literature',
        'enhanced_v1': '🚀 Enhanced RAG v1.0',
        'enhanced_v1_help': 'Enhanced RAG with intent detection (v1.0)',
        'enhanced_v2': '🎯 Enhanced RAG v2.0',
        'enhanced_v2_help': 'Advanced RAG v2.0 with adaptive grounding system',
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
        'intent_detection': '🎯 Intent Detection',
        'use_intent_detection': 'Auto-detect query intent',
        'show_intent_details': 'Show intent analysis',
        'override_content_filter': 'Override content type filter:',
        'auto_intent_detection': 'Auto (use intent detection)',
        'all_types_override': 'All types',
        'advanced_options': '🔧 Advanced Options',
        'context_chunks': 'Context Fragments:',
        'context_chunks_help': 'Number of relevant fragments to use',
        'content_filter': 'Filter by Content Type:',
        'all_types': 'All types',
        'clinical_query': '💬 Clinical Query',
        'query_label': 'Enter your clinical question:',
        'query_placeholder': 'Example: How do I help a patient with panic attacks who is not responding to breathing techniques?',
        'query_help': 'You can write in Spanish or English',
        'examples_label': '**Examples:**',
        'generate_button': '🔍 Generate Responses',
        'generating': 'Generating responses from multiple systems...',
        'init_success': 'initialized successfully!',
        'init_error': 'Failed to initialize',
        'select_system_warning': 'Please select at least one system to initialize.',
        'no_systems_warning': '⚠️ Systems Not Initialized',
        'no_systems_message': 'Please select one or more systems in the sidebar and click "Initialize Selected Systems" to begin.',
        # Style options
        'style_clinical': '👩‍⚕️ Clinical',
        'style_brief': '⚡ Brief',
        'style_comprehensive': '📚 Comprehensive',
        'style_conversational': '💭 Conversational',
        # Length options
        'length_concise': '📝 Concise',
        'length_medium': '📄 Medium',
        'length_detailed': '📖 Detailed',
        # Tone options
        'tone_professional': '🎩 Professional',
        'tone_supportive': '🤝 Supportive',
        'tone_direct': '🎯 Direct',
        'tone_empathetic': '💝 Empathetic',
        # Example queries
        'example_1': 'What is CBT?',
        'example_2': 'Social anxiety techniques',
        'example_3': 'How to handle resistance',
        'example_4': 'Panic protocol',
        'example_5': 'Homework assignments',
        # Response display labels
        'response_from': 'Response from',
        'processing_time': 'Processing time',
        'fragments': 'Fragments',
        'content_types': 'Content Types:',
    },
    'spanish': {
        'title': '🧠 Sistema RAG v2.0 para Psicoterapeutas',
        'subtitle': 'Asistente Inteligente Avanzado con Sistema de Fundamentación Adaptiva',
        'sidebar_header': '⚙️ Configuración del Sistema',
        'system_type_header': '🔧 Tipo de Sistema',
        'system_type_instruction': 'Selecciona uno o más sistemas para comparar:',
        'base_claude': '💬 Base Claude (Sin RAG)',
        'base_claude_help': 'Modelo Claude sin acceso a literatura clínica',
        'enhanced_v1': '🚀 RAG Mejorado v1.0',
        'enhanced_v1_help': 'RAG mejorado con detección de intención (v1.0)',
        'enhanced_v2': '🎯 RAG Mejorado v2.0',
        'enhanced_v2_help': 'RAG avanzado v2.0 con sistema de fundamentación adaptiva',
        'initialize_button': '🔄 Inicializar Sistemas Seleccionados',
        'initializing': 'Inicializando sistemas...',
        'system_stats': '📊 Estadísticas del Sistema',
        'language_header': '🌐 Configuración de Idioma',
        'response_language': 'Idioma de Respuesta:',
        'ui_language': 'Idioma de Interfaz:',
        'response_params': '📝 Parámetros de Respuesta',
        'response_style': 'Estilo de Respuesta:',
        'response_length': 'Longitud de Respuesta:',
        'complexity': 'Complejidad/Creatividad:',
        'complexity_help': '0.1 = Muy conservador, 1.5 = Muy creativo',
        'tone': 'Tono de Respuesta:',
        'intent_detection': '🎯 Detección de Intención',
        'use_intent_detection': 'Auto-detectar intención de consulta',
        'show_intent_details': 'Mostrar análisis de intención',
        'override_content_filter': 'Anular filtro de tipo de contenido:',
        'auto_intent_detection': 'Auto (usar detección de intención)',
        'all_types_override': 'Todos los tipos',
        'advanced_options': '🔧 Opciones Avanzadas',
        'context_chunks': 'Fragmentos de Contexto:',
        'context_chunks_help': 'Número de fragmentos relevantes a usar',
        'content_filter': 'Filtrar por Tipo de Contenido:',
        'all_types': 'Todos los tipos',
        'clinical_query': '💬 Consulta Clínica',
        'query_label': 'Ingresa tu pregunta clínica:',
        'query_placeholder': 'Ejemplo: ¿Cómo ayudo a un paciente con ataques de pánico que no responde a técnicas de respiración?',
        'query_help': 'Puedes escribir en español o inglés',
        'examples_label': '**Ejemplos:**',
        'generate_button': '🔍 Generar Respuestas',
        'generating': 'Generando respuestas desde múltiples sistemas...',
        'init_success': '¡inicializado exitosamente!',
        'init_error': 'Error al inicializar',
        'select_system_warning': 'Por favor selecciona al menos un sistema para inicializar.',
        'no_systems_warning': '⚠️ Sistemas No Inicializados',
        'no_systems_message': 'Por favor selecciona uno o más sistemas en la barra lateral y haz clic en "Inicializar Sistemas Seleccionados" para comenzar.',
        # Style options
        'style_clinical': '👩‍⚕️ Clínico',
        'style_brief': '⚡ Breve',
        'style_comprehensive': '📚 Comprensivo',
        'style_conversational': '💭 Conversacional',
        # Length options
        'length_concise': '📝 Conciso',
        'length_medium': '📄 Mediano',
        'length_detailed': '📖 Detallado',
        # Tone options
        'tone_professional': '🎩 Profesional',
        'tone_supportive': '🤝 Solidario',
        'tone_direct': '🎯 Directo',
        'tone_empathetic': '💝 Empático',
        # Example queries
        'example_1': '¿Qué es la TCC?',
        'example_2': 'Técnicas para ansiedad social',
        'example_3': 'Cómo manejar resistencia',
        'example_4': 'Protocolo para pánico',
        'example_5': 'Tareas para casa',
        # Response display labels
        'response_from': 'Respuesta de',
        'processing_time': 'Tiempo de procesamiento',
        'fragments': 'Fragmentos',
        'content_types': 'Tipos de Contenido:',
    }
}

def get_text(key):
    """Get text in current UI language."""
    return UI_TEXT.get(st.session_state.ui_language, UI_TEXT['spanish']).get(key, key)

# Custom CSS for v2.0 styling with version indicator and mobile responsiveness
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
    font-size: 12px;
    z-index: 1000;
}

.v2-badge {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
    margin-left: 8px;
}

.system-response {
    border-left: 4px solid #4ECDC4;
    padding: 15px;
    margin: 10px 0;
    background: rgba(78, 205, 196, 0.1);
    border-radius: 5px;
}

.response-header {
    font-weight: bold;
    color: #2E86AB;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.processing-time {
    font-size: 0.8em;
    color: #666;
    background: rgba(0,0,0,0.05);
    padding: 2px 6px;
    border-radius: 3px;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    [data-testid="column"] {
        width: 100% !important;
        flex: 100% !important;
        margin-bottom: 0.5rem !important;
    }
    .version-indicator {
        position: relative;
        top: auto;
        left: auto;
        margin-bottom: 10px;
    }
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    border: 1px solid #ddd;
    background: linear-gradient(145deg, #f8f9fa, #e9ecef);
    color: #495057;
    padding: 8px 12px;
    transition: all 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(145deg, #e9ecef, #dee2e6);
    border-color: #4ECDC4;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# Version indicator
st.markdown("""
<div class="version-indicator">
    RAG v2.0 🚀
</div>
""", unsafe_allow_html=True)

class SimpleClaudeRAG:
    """Simple wrapper for v2.0 compatibility in multi-system comparison"""
    def __init__(self, system_type="enhanced_v2"):
        self.system_type = system_type
        # Get API key from Streamlit secrets or environment
        self.api_key = None
        try:
            if 'ANTHROPIC_API_KEY' in st.secrets:
                self.api_key = st.secrets['ANTHROPIC_API_KEY']
        except:
            pass  # No secrets file, that's ok
        
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if self.api_key and system_type == "enhanced_v2":
            try:
                # Initialize v2.0 system without api_key parameter
                # The v2.0 system will get api_key from environment or config
                os.environ['ANTHROPIC_API_KEY'] = self.api_key  # Ensure it's available
                logger.info(f"Attempting to initialize v2.0 system with API key: {self.api_key[:10]}...")
                self.rag_system = create_enhanced_rag_v2()
                # Expose collection for knowledge base management
                if hasattr(self.rag_system, 'collection'):
                    self.collection = self.rag_system.collection
                self.available = True
                logger.info("✅ v2.0 RAG system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize v2.0: {e}")
                logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
                self.available = False
        elif self.api_key and system_type == "base_claude":
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.available = True
        else:
            self.available = False

    def query(self, question, **kwargs):
        """Query the system with timing"""
        start_time = time.time()
        
        if self.system_type == "enhanced_v2" and hasattr(self, 'rag_system'):
            try:
                response = self.rag_system.query(question, **kwargs)
                processing_time = time.time() - start_time
                # Handle response from therapy_rag which returns 'response' not 'answer'
                answer_text = response.get('response', response.get('answer', 'No response'))
                return {
                    'answer': answer_text,
                    'processing_time': processing_time,
                    'system_type': 'Enhanced RAG v2.0',
                    'context_chunks_used': response.get('context_chunks_used', 0),
                    'context_sources': response.get('context_sources', []),
                    'grounding_path': response.get('grounding_path', 'unknown')
                }
            except Exception as e:
                logger.error(f"v2.0 query failed: {e}")
                return {'answer': f'Error: {str(e)}', 'processing_time': time.time() - start_time}
        
        elif self.system_type == "base_claude" and hasattr(self, 'client'):
            try:
                message = self.client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    temperature=0.7,
                    messages=[{"role": "user", "content": question}]
                )
                processing_time = time.time() - start_time
                return {
                    'answer': message.content[0].text,
                    'processing_time': processing_time,
                    'system_type': 'Base Claude',
                    'context_used': [],
                    'grounding_path': 'none'
                }
            except Exception as e:
                logger.error(f"Base Claude query failed: {e}")
                return {'answer': f'Error: {str(e)}', 'processing_time': time.time() - start_time}
        
        return {'answer': 'System not available', 'processing_time': 0}

def initialize_rag_system(system_type):
    """Initialize a RAG system of the specified type"""
    try:
        if system_type == "enhanced_v2":
            st.session_state.enhanced_v2_system = SimpleClaudeRAG("enhanced_v2")
            st.session_state.systems_initialized['enhanced_v2'] = st.session_state.enhanced_v2_system.available
            return st.session_state.enhanced_v2_system.available
        
        elif system_type == "enhanced_v1":
            # Placeholder for v1.0 system
            st.session_state.enhanced_v1_system = None
            st.session_state.systems_initialized['enhanced_v1'] = False
            return False
        
        elif system_type == "base_claude":
            st.session_state.base_claude_system = SimpleClaudeRAG("base_claude")
            st.session_state.systems_initialized['base_claude'] = st.session_state.base_claude_system.available
            return st.session_state.base_claude_system.available
            
    except Exception as e:
        logger.error(f"Failed to initialize {system_type}: {e}")
        return False
    
    return False

def main():
    """Main application function"""
    
    # Title with v2.0 badge
    st.title(get_text('title'))
    st.markdown(f"*{get_text('subtitle')}* <span class='v2-badge'>v2.0</span>", unsafe_allow_html=True)
    
    # Auto-initialize Enhanced RAG v2.0 and Base Claude on first load
    if not any(st.session_state.systems_initialized.values()):
        with st.spinner("🚀 Auto-initializing systems..."):
            try:
                # Auto-initialize Enhanced RAG v2.0 
                st.session_state.enhanced_v2_system = SimpleClaudeRAG("enhanced_v2")
                st.session_state.systems_initialized['enhanced_v2'] = st.session_state.enhanced_v2_system.available
                
                # Auto-initialize Base Claude
                st.session_state.base_claude_system = SimpleClaudeRAG("base_claude")
                st.session_state.systems_initialized['base_claude'] = st.session_state.base_claude_system.available
                
                if st.session_state.systems_initialized['enhanced_v2']:
                    st.success("✅ RAG v2.0 system ready!")
                if st.session_state.systems_initialized['base_claude']:
                    st.success("✅ Base Claude ready!")
                    
                if not any(st.session_state.systems_initialized.values()):
                    st.error("⚠️ No systems could be initialized. Check API key.")
                    
                time.sleep(0.5)  # Brief pause to show success message
            except Exception as e:
                st.error(f"Auto-initialization failed: {e}")
                st.info("Please check your API key configuration.")
    
    # Comprehensive Sidebar
    with st.sidebar:
        # UI Language Selector
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
            st.rerun()
        
        st.divider()
        
        # RAG System Selection with checkboxes
        st.subheader(get_text('system_type_header'))
        st.write(get_text('system_type_instruction'))
        
        use_base_claude = st.checkbox(
            get_text('base_claude'),
            help=get_text('base_claude_help')
        )
        
        use_enhanced_v1 = st.checkbox(
            get_text('enhanced_v1'),
            help=get_text('enhanced_v1_help'),
            disabled=True  # Disable v1.0 for now
        )
        
        use_enhanced_v2 = st.checkbox(
            get_text('enhanced_v2'),
            help=get_text('enhanced_v2_help'),
            value=True  # Default to enhanced RAG v2.0
        )
        
        # Initialize selected systems
        if st.button(get_text('initialize_button'), type="primary"):
            with st.spinner(get_text('initializing')):
                success_count = 0
                total_selected = sum([use_enhanced_v1, use_enhanced_v2])
                
                # Base Claude doesn't need complex initialization
                base_claude_selected = use_base_claude
                if base_claude_selected:
                    if initialize_rag_system("base_claude"):
                        st.success(f"✅ {get_text('base_claude')} {get_text('init_success')}")
                
                if use_enhanced_v1:
                    st.info("Enhanced RAG v1.0 temporarily disabled")
                
                if use_enhanced_v2:
                    if initialize_rag_system("enhanced_v2"):
                        success_count += 1
                        st.success(f"✅ {get_text('enhanced_v2')} {get_text('init_success')}")
                    else:
                        st.error(f"❌ {get_text('init_error')} {get_text('enhanced_v2')}")
                
                total_systems = total_selected + (1 if base_claude_selected else 0)
                if total_systems == 0:
                    st.warning(get_text('select_system_warning'))
                elif success_count == total_selected and total_selected > 0:
                    if base_claude_selected:
                        st.success(f"✅ {total_systems} systems {get_text('init_success')}")
                    else:
                        st.success(f"✅ {success_count} systems {get_text('init_success')}")
        
        # Show system stats if initialized
        st.subheader(get_text('system_stats'))
        
        # Enhanced RAG v2.0 status
        if st.session_state.systems_initialized.get('enhanced_v2', False):
            st.success("🎯 Enhanced RAG v2.0: ✅ Active")
        else:
            st.error("🎯 Enhanced RAG v2.0: ❌ Inactive")
        
        # Base Claude status
        if st.session_state.systems_initialized.get('base_claude', False):
            st.success("💬 Base Claude: ✅ Active")
        else:
            st.error("💬 Base Claude: ❌ Inactive")
        
        st.divider()
        
        # Knowledge Base Management
        st.subheader("📚 Knowledge Base Management")
        
        # Show current knowledge base stats
        if 'enhanced_v2_system' in st.session_state and st.session_state.enhanced_v2_system:
            try:
                stats = get_knowledge_base_stats(st.session_state.enhanced_v2_system)
                st.info(f"📊 Total Documents: {stats.get('total_documents', 0)}")
                
                # Show content types if available
                if stats.get('content_types'):
                    st.write("**Content Types:**")
                    for ct, count in stats['content_types'].items():
                        st.write(f"• {ct}: {count}")
                
                # Show source files if available
                if stats.get('source_files') and len(stats['source_files']) < 10:  # Only show if manageable number
                    st.write("**Source Files:**")
                    for source, count in list(stats['source_files'].items())[:5]:  # Limit to 5
                        st.write(f"• {source}: {count} chunks")
                        
            except Exception as e:
                st.warning(f"Could not load knowledge base stats: {str(e)}")
        else:
            st.warning("Enhanced RAG v2.0 system not initialized. Initialize system first.")
        
        # Document Upload Section
        st.write("**Upload New Documents:**")
        uploaded_files = st.file_uploader(
            "Choose therapy knowledge files",
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            help="Upload documents to expand the knowledge base. Supported formats: TXT, PDF, DOCX, Markdown"
        )
        
        if uploaded_files:
            if st.button("🔄 Process and Add Documents", use_container_width=True):
                if 'enhanced_v2_system' in st.session_state and st.session_state.enhanced_v2_system:
                    with st.spinner("Processing uploaded documents..."):
                        success, message = process_uploaded_documents(uploaded_files, st.session_state.enhanced_v2_system)
                        if success:
                            st.success(message)
                            st.rerun()  # Refresh to show updated stats
                        else:
                            st.error(message)
                else:
                    st.error("Enhanced RAG v2.0 system not initialized. Please initialize first.")
        
        # Knowledge Base Management Actions
        st.write("**Management Actions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Knowledge Base", 
                        use_container_width=True, 
                        help="Remove all documents from the knowledge base"):
                if 'enhanced_v2_system' in st.session_state and st.session_state.enhanced_v2_system:
                    if st.session_state.get('confirm_clear', False):
                        with st.spinner("Clearing knowledge base..."):
                            success = clear_knowledge_base(st.session_state.enhanced_v2_system)
                            if success:
                                st.success("Knowledge base cleared successfully!")
                                st.session_state.confirm_clear = False
                                st.rerun()
                            else:
                                st.error("Failed to clear knowledge base")
                    else:
                        st.warning("⚠️ This will delete ALL documents. Click again to confirm.")
                        st.session_state.confirm_clear = True
                else:
                    st.error("System not initialized")
        
        with col2:
            if st.button("🔄 Refresh Stats", 
                        use_container_width=True, 
                        help="Reload knowledge base statistics"):
                st.rerun()
        
        with col3:
            if st.button("💾 Export Info", 
                        use_container_width=True, 
                        help="Export knowledge base information"):
                if 'enhanced_v2_system' in st.session_state and st.session_state.enhanced_v2_system:
                    try:
                        stats = get_knowledge_base_stats(st.session_state.enhanced_v2_system)
                        stats_json = json.dumps(stats, indent=2)
                        st.download_button(
                            "📥 Download KB Stats",
                            data=stats_json,
                            file_name=f"knowledge_base_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Failed to export stats: {str(e)}")
                else:
                    st.error("System not initialized")
        
        st.divider()
        
        # Response Configuration
        st.subheader(get_text('response_params'))
        
        response_style = st.selectbox(
            get_text('response_style'),
            options=['clinical', 'brief', 'comprehensive', 'conversational'],
            format_func=lambda x: get_text(f'style_{x}'),
            index=0
        )
        
        response_length = st.selectbox(
            get_text('response_length'),
            options=['concise', 'medium', 'detailed'],
            format_func=lambda x: get_text(f'length_{x}'),
            index=1
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
            options=['professional', 'supportive', 'direct', 'empathetic'],
            format_func=lambda x: get_text(f'tone_{x}'),
            index=1
        )
        
        st.divider()
        
        # Advanced Options
        st.subheader(get_text('advanced_options'))
        
        context_chunks = st.slider(
            get_text('context_chunks'),
            min_value=1,
            max_value=10,
            value=5,
            help=get_text('context_chunks_help')
        )
        
        # System Configuration (moved to bottom)
        st.divider()
        st.header(get_text('sidebar_header'))
        
        st.success("☁️ **Cloud Mode Active**")
        try:
            if 'ANTHROPIC_API_KEY' in st.secrets:
                st.success("✅ API Key Found (from secrets)")
        except:
            pass
        if os.getenv("ANTHROPIC_API_KEY"):
            st.success("✅ API Key Found: sk-ant-api03-fl...")
        elif not os.getenv("ANTHROPIC_API_KEY"):
            st.error("❌ No API Key Found")
            st.info("Add ANTHROPIC_API_KEY to Streamlit secrets")

    # Main content area
    st.subheader(get_text('clinical_query'))
    
    # Quick examples
    st.write(get_text('examples_label'))
    example_questions = [
        get_text('example_1'),
        get_text('example_2'),
        get_text('example_3'),
        get_text('example_4'),
        get_text('example_5')
    ]
    
    # Display examples - responsive layout
    # Use 1 column on mobile, 3 on desktop
    with st.container():
        # Create columns but they'll stack on mobile due to CSS
        cols = st.columns([1, 1, 1])
        for i, example in enumerate(example_questions[:3]):
            with cols[i % 3]:
                if st.button(example, key=f"example_{i}_{st.session_state.ui_language}", 
                           use_container_width=True):
                    st.session_state.selected_example = example
                    st.rerun()
        
        # Show remaining examples if any
        if len(example_questions) > 3:
            cols2 = st.columns([1, 1, 1])
            for i, example in enumerate(example_questions[3:]):
                with cols2[i % 3]:
                    if st.button(example, key=f"example_{i+3}_{st.session_state.ui_language}",
                               use_container_width=True):
                        st.session_state.selected_example = example
                        st.rerun()
    
    # Handle example selection - update the widget state directly
    if st.session_state.get('selected_example', ''):
        # Use st.session_state with the widget key to directly update the text area
        st.session_state['question_input'] = st.session_state.selected_example
        st.session_state.selected_example = ""
    
    # Question input
    question = st.text_area(
        get_text('query_label'),
        placeholder=get_text('query_placeholder'),
        help=get_text('query_help'),
        height=100,
        key='question_input'
    )
    
    # Generate responses button
    if st.button(get_text('generate_button'), type="primary", disabled=not question.strip()):
        if not any(st.session_state.systems_initialized.values()):
            st.warning(get_text('no_systems_warning'))
            st.info(get_text('no_systems_message'))
        else:
            with st.spinner(get_text('generating')):
                responses = []
                
                # Generate response from Enhanced RAG v2.0
                if st.session_state.systems_initialized.get('enhanced_v2', False):
                    try:
                        # Map UI parameters to RAG system parameters
                        length_to_tokens = {
                            'brief': 150,
                            'standard': 300,
                            'detailed': 600
                        }
                        
                        # Use only valid parameters for therapy_rag.query()
                        response = st.session_state.enhanced_v2_system.query(
                            question,
                            response_style=response_style,
                            temperature=complexity,
                            max_tokens=length_to_tokens.get(response_length, 300),
                            n_context_chunks=context_chunks,
                            custom_instructions=f"Tone: {tone}"
                        )
                        responses.append(('Enhanced RAG v2.0 🎯', response))
                    except Exception as e:
                        st.error(f"Enhanced RAG v2.0 error: {str(e)}")
                
                # Generate response from Base Claude
                if st.session_state.systems_initialized.get('base_claude', False):
                    try:
                        response = st.session_state.base_claude_system.query(question)
                        responses.append(('Base Claude 💬', response))
                    except Exception as e:
                        st.error(f"Base Claude error: {str(e)}")
                
                # Display responses
                if responses:
                    st.subheader("📋 Responses")
                    
                    for system_name, response in responses:
                        with st.container():
                            st.markdown(f"""
                            <div class="system-response">
                                <div class="response-header">
                                    <span>{get_text('response_from')} {system_name}</span>
                                    <span class="processing-time">{get_text('processing_time')}: {response.get('processing_time', 0):.2f}s</span>
                                </div>
                                <div>{response.get('answer', 'No response available')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show additional info for RAG systems
                            if response.get('context_chunks_used', 0) > 0:
                                with st.expander(f"📚 Context Information ({response['context_chunks_used']} {get_text('fragments')})"):
                                    if 'context_sources' in response and response['context_sources']:
                                        for i, source in enumerate(response['context_sources'][:3]):  # Show first 3
                                            st.write(f"**Source {i+1}:** {source.get('source', source.get('section', source.get('content_type', 'Unknown')))} (Confidence: {source.get('confidence', 0):.2%})")
                                    else:
                                        st.write(f"Used {response['context_chunks_used']} context chunks from knowledge base.")
                            
                            # Show grounding path for v2.0
                            if 'grounding_path' in response and response['grounding_path'] != 'unknown':
                                st.info(f"🎯 Grounding Path: {response['grounding_path']}")


def process_uploaded_documents(uploaded_files, rag_system):
    """Process uploaded documents and add them to the knowledge base."""
    if not uploaded_files:
        return False, "No files uploaded"
    
    success_count = 0
    error_messages = []
    
    for uploaded_file in uploaded_files:
        try:
            # Extract text from file
            text_content = extract_text_from_file(uploaded_file)
            if text_content:
                # Add to knowledge base
                success = add_document_to_knowledge_base(
                    rag_system, 
                    text_content, 
                    uploaded_file.name
                )
                if success:
                    success_count += 1
                else:
                    error_messages.append(f"Failed to add {uploaded_file.name} to knowledge base")
            else:
                error_messages.append(f"Could not extract text from {uploaded_file.name}")
                
        except Exception as e:
            error_messages.append(f"Error processing {uploaded_file.name}: {str(e)}")
    
    if success_count > 0:
        return True, f"Successfully processed {success_count} documents. Errors: {'; '.join(error_messages) if error_messages else 'None'}"
    else:
        return False, f"Failed to process any documents. Errors: {'; '.join(error_messages)}"


def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file."""
    try:
        file_type = uploaded_file.type
        file_content = uploaded_file.read()
        
        if file_type == "text/plain" or uploaded_file.name.endswith('.txt'):
            return file_content.decode('utf-8')
        elif file_type == "text/markdown" or uploaded_file.name.endswith('.md'):
            return file_content.decode('utf-8')
        elif file_type == "application/pdf" or uploaded_file.name.endswith('.pdf'):
            try:
                import PyPDF2
                import io
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                st.warning("PyPDF2 not available. PDF files cannot be processed.")
                return None
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or uploaded_file.name.endswith('.docx'):
            try:
                import python_docx
                import io
                doc = python_docx.Document(io.BytesIO(file_content))
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                st.warning("python-docx not available. DOCX files cannot be processed.")
                return None
        else:
            st.warning(f"Unsupported file type: {file_type}")
            return None
            
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None


def add_document_to_knowledge_base(rag_system, text_content, filename):
    """Add a document to the RAG system's knowledge base."""
    try:
        # Chunk the document
        chunks = chunk_document(text_content, filename)
        
        if not chunks:
            return False
        
        # Add chunks to ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk['text'])
            metadatas.append({
                'source_file': filename,
                'chunk_id': i,
                'content_type': 'uploaded_document',
                'section_title': f"{filename} - Chunk {i+1}",
                'language': 'auto'
            })
            ids.append(f"{filename}_chunk_{i}")
        
        # Add to collection
        rag_system.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error adding document to knowledge base: {str(e)}")
        return False


def chunk_document(text_content, filename, chunk_size=500, chunk_overlap=50):
    """Split document into chunks for vector storage."""
    try:
        # Simple text chunking by sentences
        sentences = text_content.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, start new chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': filename
                })
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-chunk_overlap:]) if len(words) > chunk_overlap else current_chunk
                current_chunk = overlap_text + ". " + sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'source': filename
            })
        
        return chunks
        
    except Exception as e:
        st.error(f"Error chunking document: {str(e)}")
        return []


def clear_knowledge_base(rag_system):
    """Clear all documents from the knowledge base."""
    try:
        # Delete the collection and recreate it
        collection_name = rag_system.collection.name
        rag_system.client.delete_collection(name=collection_name)
        rag_system.collection = rag_system.client.create_collection(name=collection_name)
        return True
    except Exception as e:
        st.error(f"Error clearing knowledge base: {str(e)}")
        return False


def get_knowledge_base_stats(rag_system):
    """Get statistics about the current knowledge base."""
    try:
        total_docs = rag_system.collection.count()
        
        if total_docs == 0:
            return {
                'total_documents': 0,
                'content_types': {},
                'source_files': {}
            }
        
        # Get sample of documents to analyze
        sample_batch = rag_system.collection.get(limit=min(100, total_docs))
        
        content_types = {}
        source_files = {}
        
        for metadata in sample_batch['metadatas']:
            content_type = metadata.get('content_type', 'unknown')
            source_file = metadata.get('source_file', 'unknown')
            
            content_types[content_type] = content_types.get(content_type, 0) + 1
            source_files[source_file] = source_files.get(source_file, 0) + 1
        
        return {
            'total_documents': total_docs,
            'content_types': content_types,
            'source_files': source_files
        }
        
    except Exception as e:
        st.error(f"Error getting knowledge base stats: {str(e)}")
        return {
            'total_documents': 0,
            'content_types': {},
            'source_files': {}
        }


if __name__ == "__main__":
    main()# ChromaDB 0.3.2 Fix: Cache refresh Wed Jun 10 11:18:26 CST 2026
