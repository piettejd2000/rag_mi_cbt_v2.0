#!/usr/bin/env python3
"""
Script to load chunked JSON files into ChromaDB with multilingual embeddings.
Can be run after chunking is complete.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingLoader:
    """
    Loads chunked documents into ChromaDB with multilingual embeddings.
    """
    
    def __init__(self, 
                 chunks_dir: Path,
                 chroma_path: Path = None,
                 embedding_model: str = 'intfloat/multilingual-e5-base',
                 collection_name: str = 'therapy_knowledge'):
        """
        Initialize the embedding loader.
        
        Args:
            chunks_dir: Directory containing chunked JSON files
            chroma_path: Path for ChromaDB persistence
            embedding_model: Name of the sentence transformer model
            collection_name: Name of the ChromaDB collection
        """
        self.chunks_dir = Path(chunks_dir)
        
        # Initialize embedding model (try alternatives if multilingual-e5-base fails)
        logger.info(f"Loading embedding model: {embedding_model}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.warning(f"Failed to load {embedding_model}: {e}")
            logger.info("Trying alternative embedding model: all-MiniLM-L6-v2")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding_model = 'all-MiniLM-L6-v2'
            except Exception as e2:
                logger.warning(f"Failed to load all-MiniLM-L6-v2: {e2}")
                logger.info("Trying paraphrase-MiniLM-L6-v2")
                self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                embedding_model = 'paraphrase-MiniLM-L6-v2'
        
        logger.info(f"Successfully loaded embedding model: {embedding_model}")
        
        # Initialize ChromaDB
        if chroma_path is None:
            chroma_path = Path('/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chroma_db')
        
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "CBT and MI clinical knowledge base - bilingual"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def load_json_chunks(self, json_file: Path) -> List[Dict]:
        """
        Load chunks from a JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Returns:
            List of chunk dictionaries
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('chunks', [])
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")
            return []
    
    def load_jsonl_chunks(self, jsonl_file: Path) -> List[Dict]:
        """
        Load chunks from a JSONL file (for MI exchanges).
        
        Args:
            jsonl_file: Path to the JSONL file
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        chunk = json.loads(line.strip())
                        chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed line {line_num} in {jsonl_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to load {jsonl_file}: {e}")
        return chunks
    
    def prepare_text_for_embedding(self, chunk: Dict) -> str:
        """
        Prepare chunk text for embedding with metadata context.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            Enhanced text for embedding
        """
        # Handle MI exchange chunks (different structure)
        if 'embedding_text' in chunk:
            return chunk['embedding_text']
        
        # Handle regular chunks
        text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        
        # Add section title for context
        if metadata.get('section_title'):
            text = f"Section: {metadata['section_title']}\n\n{text}"
        
        # Add content type hint
        content_type = metadata.get('content_type', '')
        if content_type == 'dialogue':
            text = f"[Clinical Dialogue]\n{text}"
        elif content_type == 'procedure':
            text = f"[Procedure/Protocol]\n{text}"
        elif content_type == 'case_example':
            text = f"[Case Example]\n{text}"
        elif content_type == 'mi_exchange':
            text = f"[MI Exchange]\n{text}"
        
        return text
    
    def process_file(self, file_path: Path) -> int:
        """
        Process a single file (JSON or JSONL) and load chunks to ChromaDB.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of chunks added
        """
        logger.info(f"Processing: {file_path.name}")
        
        # Load chunks based on file extension
        if file_path.suffix == '.jsonl':
            chunks = self.load_jsonl_chunks(file_path)
        else:
            chunks = self.load_json_chunks(file_path)
            
        if not chunks:
            logger.warning(f"No chunks found in {file_path.name}")
            return 0
        
        # Prepare data for ChromaDB
        texts = []
        embeddings = []
        metadatas = []
        ids = []
        
        # Process chunks in batches for efficiency
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare texts for embedding
            batch_texts = [self.prepare_text_for_embedding(chunk) for chunk in batch]
            
            # Generate embeddings
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Collect data
            for chunk, embedding in zip(batch, batch_embeddings):
                # Handle different chunk structures
                if 'exchange' in chunk:  # MI exchange chunk
                    text = f"{chunk['exchange']['client_utterance']}\n{chunk['exchange']['clinician_response']}"
                else:  # Regular chunk
                    text = chunk.get('text', '')
                    
                texts.append(text)
                embeddings.append(embedding.tolist())
                
                # Prepare metadata
                metadata = chunk.get('metadata', {})
                
                # Add content type for MI exchanges
                if 'exchange' in chunk:
                    metadata['content_type'] = 'mi_exchange'
                
                # Flatten nested metadata for MI chunks
                if 'clinical_context' in metadata:
                    for key, value in metadata['clinical_context'].items():
                        metadata[f'clinical_{key}'] = value
                
                # Convert complex structures to JSON strings
                for key in ['bilingual_concepts', 'mi_behaviors', 'clinical_scenario_tags', 
                           'therapeutic_goals', 'search_keywords', 'behavior_descriptions',
                           'quality_indicators', 'change_dynamics', 'demographics']:
                    if key in metadata and isinstance(metadata[key], (dict, list)):
                        metadata[key] = json.dumps(metadata[key])
                
                # Ensure all metadata values are strings, numbers, or booleans
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = value
                    elif isinstance(value, (dict, list)):
                        clean_metadata[key] = json.dumps(value)
                    else:
                        clean_metadata[key] = str(value)
                
                metadatas.append(clean_metadata)
                ids.append(chunk['chunk_id'])
        
        # Add to ChromaDB
        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(chunks)} chunks from {file_path.name}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            return 0
    
    def process_all_files(self, pattern: str = "*_chunks.json") -> None:
        """
        Process all JSON chunk files and load to ChromaDB.
        
        Args:
            pattern: Glob pattern for chunk files
        """
        # Find all chunk files
        files = sorted(self.chunks_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return
        
        logger.info(f"Found {len(files)} chunk files to process")
        
        total_chunks = 0
        for json_file in tqdm(files, desc="Loading to ChromaDB"):
            chunks_added = self.process_file(json_file)
            total_chunks += chunks_added
        
        logger.info(f"Successfully loaded {total_chunks} chunks to ChromaDB")
        
        # Print collection info
        collection_count = self.collection.count()
        logger.info(f"Total documents in collection: {collection_count}")
    
    def load_mi_exchanges(self) -> int:
        """
        Load MI labeled exchange chunks from JSONL file.
        
        Returns:
            Number of MI chunks added
        """
        mi_chunks_file = Path('/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chunked_MI_labeled_utterances/mi_labeled_exchanges_chunks.jsonl')
        
        if not mi_chunks_file.exists():
            logger.warning(f"MI chunks file not found: {mi_chunks_file}")
            return 0
            
        logger.info("Loading MI labeled exchanges...")
        chunks_added = self.process_file(mi_chunks_file)
        logger.info(f"Added {chunks_added} MI exchange chunks")
        return chunks_added
    
    def test_retrieval(self, query: str, n_results: int = 5) -> None:
        """
        Test retrieval with a sample query.
        
        Args:
            query: Test query
            n_results: Number of results to retrieve
        """
        print(f"\nTest Query: {query}")
        print("-" * 50)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Display results
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"\n{i}. Distance: {distance:.3f}")
            print(f"   Content Type: {metadata.get('content_type', 'unknown')}")
            print(f"   Language: {metadata.get('language', 'unknown')}")
            print(f"   Section: {metadata.get('section_title', 'N/A')}")
            print(f"   Text: {doc[:200]}...")


def main():
    """
    Main entry point for the embedding loader.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load chunked documents into ChromaDB with embeddings"
    )
    parser.add_argument(
        '--chunks-dir',
        type=str,
        default='/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chunked_docs',
        help='Directory containing chunked JSON files'
    )
    parser.add_argument(
        '--chroma-path',
        type=str,
        default='/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chroma_db',
        help='Path for ChromaDB persistence'
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default='therapy_knowledge',
        help='Name of the ChromaDB collection'
    )
    parser.add_argument(
        '--test-query',
        type=str,
        help='Optional test query to run after loading'
    )
    parser.add_argument(
        '--load-mi',
        action='store_true',
        help='Load MI labeled exchange chunks'
    )
    parser.add_argument(
        '--mi-only',
        action='store_true',
        help='Load only MI chunks (skip regular chunks)'
    )
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = EmbeddingLoader(
        chunks_dir=Path(args.chunks_dir),
        chroma_path=Path(args.chroma_path) if args.chroma_path else None,
        collection_name=args.collection_name
    )
    
    # Process files based on options
    if args.mi_only:
        # Load only MI chunks
        loader.load_mi_exchanges()
    else:
        # Process regular files
        if not args.load_mi:
            loader.process_all_files()
        else:
            # Load both regular and MI chunks
            loader.process_all_files()
            loader.load_mi_exchanges()
    
    # Run test query if provided
    if args.test_query:
        loader.test_retrieval(args.test_query)
    else:
        # Run some default test queries including MI-specific ones
        test_queries = [
            "What is behavioral activation?",
            "¿Qué es la ansiedad?",
            "How to structure first therapy session?",
            "Patient won't do homework",
            "What should I say to a patient with poor medication adherence?",
            "How to respond to client resistance about diabetes management?"
        ]
        
        for query in test_queries:
            loader.test_retrieval(query, n_results=3)


if __name__ == "__main__":
    main()