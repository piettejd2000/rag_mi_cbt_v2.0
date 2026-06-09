# RAG v2.0 para Psicoterapeutas / RAG v2.0 for Psychotherapists

## 🚀 Version 2.0 - Adaptive Grounding System

This is the enhanced RAG (Retrieval-Augmented Generation) system v2.0 with adaptive grounding for mental health professionals. The system provides evidence-based responses using CBT and MI therapy knowledge.

### Key Features

- **Adaptive 3-Path Grounding System**:
  - PATH_A: Strong grounding for highly relevant retrieval
  - PATH_B: Soft blending of retrieval with clinical knowledge  
  - PATH_C: Knowledge-first approach when retrieval is not relevant

- **Bilingual Support**: Full support for English and Spanish
- **Multiple Response Styles**: Clinical, practical, educational, and more
- **Advanced Intent Detection**: Smart content type filtering
- **Comprehensive Knowledge Base**: CBT and MI therapy resources

### Deployment

This application is deployed on Streamlit Cloud:
- **Production URL**: https://rag-mi-cbt-v2-0.streamlit.app

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/rag_mi_cbt_v2.0.git
cd rag_mi_cbt_v2.0
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API key:
Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your-api-key-here"
```

4. Run the application:
```bash
streamlit run streamlit_app.py
```

### Technologies

- **Frontend**: Streamlit
- **LLM**: Anthropic Claude API
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers
- **Framework**: Python 3.9+

### Version History

- **v2.0** (June 2026): Adaptive grounding system with 3-path routing
- **v1.0** (May 2025): Initial RAG implementation with forced grounding

### License

This project is proprietary software for mental health professional use.

### Support

For questions or issues, please contact the development team.