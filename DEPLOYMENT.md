# Deployment Instructions for RAG v2.0

## GitHub Repository Setup

1. **Push to GitHub Repository**:
   ```bash
   cd deployment_package_v2
   git init
   git add .
   git commit -m "Initial commit: RAG v2.0 with adaptive grounding"
   git branch -M main
   git remote add origin https://github.com/[your-username]/rag_mi_cbt_v2.0.git
   git push -u origin main
   ```

## Streamlit Cloud Deployment

1. **Connect Repository to Streamlit Cloud**:
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Connect to GitHub repository: `rag_mi_cbt_v2.0`
   - Select branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: `rag-mi-cbt-v2-0` (will create https://rag-mi-cbt-v2-0.streamlit.app)

2. **Configure Secrets**:
   In Streamlit Cloud dashboard, go to Settings → Secrets and add:
   ```toml
   ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
   ```

3. **Advanced Settings** (if needed):
   - Python version: 3.9
   - Advanced → Install command:
     ```bash
     pip install -r requirements.txt
     ```

## Important Notes

- **ChromaDB**: The vector database will be created automatically on first run
- **Memory**: The app requires at least 1GB RAM due to embedding models
- **API Key**: Ensure the Anthropic API key has sufficient credits
- **Dependencies**: The requirements.txt includes PyTorch dependencies for sentence-transformers

## Monitoring

After deployment:
1. Check the app logs in Streamlit Cloud dashboard
2. Verify the app loads at https://rag-mi-cbt-v2-0.streamlit.app
3. Test both English and Spanish functionality
4. Confirm v2.0 branding appears in the UI

## Troubleshooting

**If deployment fails:**
- Check logs for missing dependencies
- Verify all files are present in the repository
- Ensure API key is correctly configured in Secrets
- Check that Python version matches requirements

**If ChromaDB issues occur:**
- The app will create the database from scratch if needed
- Ensure chunking_pys directory is included in deployment

**For performance issues:**
- Consider reducing the embedding model size if memory is limited
- Adjust the number of retrieved chunks in the settings