#!/usr/bin/env python3
"""
Main script to chunk CBT/anxiety book chapters with intelligent content detection.
Saves chunks as JSON files and optionally loads to ChromaDB.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import re

from smart_chunker import SmartChunker
from claude_classifier import ClaudeClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Main processor for chunking documents and saving results.
    """
    
    def __init__(self, 
                 input_dir: Path,
                 output_dir: Path,
                 use_claude: bool = True,
                 claude_api_key: Optional[str] = None):
        """
        Initialize document processor.
        
        Args:
            input_dir: Directory containing processed text files
            output_dir: Directory to save chunked JSON files
            use_claude: Whether to use Claude API for uncertain cases
            claude_api_key: Optional API key for Claude
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunker = SmartChunker(use_claude=use_claude, claude_api_key=claude_api_key)
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'chunks_by_type': {},
            'detection_methods': {'pattern': 0, 'claude': 0, 'fallback': 0},
            'languages': {'es': 0, 'en': 0, 'unknown': 0}
        }
    
    def process_file(self, filepath: Path) -> Dict:
        """
        Process a single document file.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            Dictionary containing chunks and metadata
        """
        logger.info(f"Processing file: {filepath.name}")
        
        # Read the file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return None
        
        # Extract document metadata from header if present
        doc_metadata = self._extract_document_metadata(text, filepath)
        
        # Chunk the document
        chunks = self.chunker.chunk_document(
            text=text,
            doc_metadata=doc_metadata,
            source_filename=filepath.name
        )
        
        # Update statistics
        self._update_stats(chunks)
        
        # Prepare output structure
        output = {
            'source_file': filepath.name,
            'processing_timestamp': datetime.now().isoformat(),
            'document_metadata': doc_metadata,
            'total_chunks': len(chunks),
            'chunks': [chunk.to_dict() for chunk in chunks]
        }
        
        # Add chunk statistics
        content_type_counts = {}
        for chunk in chunks:
            ct = chunk.metadata.get('content_type', 'unknown')
            content_type_counts[ct] = content_type_counts.get(ct, 0) + 1
        
        output['content_type_distribution'] = content_type_counts
        
        return output
    
    def process_all_files(self, pattern: str = "book_ans_cap_*.txt") -> None:
        """
        Process all files matching the pattern.
        
        Args:
            pattern: Glob pattern for files to process
        """
        # Find all matching files
        files = sorted(self.input_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No files found matching pattern: {pattern} in {self.input_dir}")
            return
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process each file
        for filepath in files:
            result = self.process_file(filepath)
            
            if result:
                # Save to JSON
                output_filename = f"{filepath.stem}_chunks.json"
                output_path = self.output_dir / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved {result['total_chunks']} chunks to {output_filename}")
                self.stats['files_processed'] += 1
        
        # Print summary statistics
        self._print_summary()
    
    def _extract_document_metadata(self, text: str, filepath: Path) -> Dict:
        """
        Extract metadata from document header.
        """
        metadata = {
            'source_path': str(filepath),
            'filename': filepath.name,
            'file_stem': filepath.stem
        }
        
        # Try to extract metadata from document header
        lines = text.split('\n')[:20]  # Check first 20 lines
        
        for line in lines:
            # Extract book info
            if 'Authors:' in line or 'Autores:' in line:
                metadata['authors'] = line.split(':', 1)[1].strip()
            elif 'Book Pages:' in line or 'Páginas:' in line:
                metadata['page_range'] = line.split(':', 1)[1].strip()
            elif 'Word Count:' in line:
                try:
                    metadata['word_count'] = int(line.split(':', 1)[1].strip())
                except:
                    pass
            elif 'Language:' in line:
                metadata['primary_language'] = line.split(':', 1)[1].strip().lower()
        
        # Extract chapter number from filename
        match = re.search(r'cap_(\d+)', filepath.stem)
        if match:
            metadata['chapter_number'] = int(match.group(1))
        
        # Identify book type
        if 'ans' in filepath.stem.lower():
            metadata['book_type'] = 'anxiety_disorders'
        
        return metadata
    
    def _update_stats(self, chunks: List) -> None:
        """
        Update processing statistics.
        """
        self.stats['total_chunks'] += len(chunks)
        
        for chunk in chunks:
            # Content type
            ct = chunk.metadata.get('content_type', 'unknown')
            self.stats['chunks_by_type'][ct] = self.stats['chunks_by_type'].get(ct, 0) + 1
            
            # Detection method
            method = chunk.metadata.get('detection_method', 'unknown')
            if method in self.stats['detection_methods']:
                self.stats['detection_methods'][method] += 1
            
            # Language
            lang = chunk.metadata.get('language', 'unknown')
            if lang in self.stats['languages']:
                self.stats['languages'][lang] += 1
    
    def _print_summary(self) -> None:
        """
        Print processing summary.
        """
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Total chunks created: {self.stats['total_chunks']}")
        
        if self.stats['files_processed'] > 0:
            print(f"Average chunks per file: {self.stats['total_chunks'] / self.stats['files_processed']:.1f}")
        
        print("\nContent Type Distribution:")
        for ct, count in sorted(self.stats['chunks_by_type'].items()):
            percentage = (count / self.stats['total_chunks']) * 100 if self.stats['total_chunks'] > 0 else 0
            print(f"  {ct}: {count} ({percentage:.1f}%)")
        
        print("\nDetection Methods Used:")
        for method, count in self.stats['detection_methods'].items():
            if count > 0:
                percentage = (count / self.stats['total_chunks']) * 100 if self.stats['total_chunks'] > 0 else 0
                print(f"  {method}: {count} ({percentage:.1f}%)")
        
        print("\nLanguage Distribution:")
        for lang, count in self.stats['languages'].items():
            if count > 0:
                percentage = (count / self.stats['total_chunks']) * 100 if self.stats['total_chunks'] > 0 else 0
                print(f"  {lang}: {count} ({percentage:.1f}%)")
        
        print("="*60)


def main():
    """
    Main entry point for the chunking script.
    """
    parser = argparse.ArgumentParser(
        description="Chunk clinical psychology documents with intelligent content detection"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/pre_processing_docs/processed_for_rag_no_cites/processed_text_files',
        help='Directory containing processed text files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chunked_docs',
        help='Directory to save chunked JSON files'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='book_ans_cap_*.txt',
        help='File pattern to process (default: book_ans_cap_*.txt)'
    )
    parser.add_argument(
        '--use-claude',
        action='store_true',
        default=True,
        help='Use Claude API for uncertain content classification'
    )
    parser.add_argument(
        '--no-claude',
        action='store_true',
        help='Disable Claude API (pattern-only mode)'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        help='Process only a single test file'
    )
    
    args = parser.parse_args()
    
    # Determine Claude usage
    use_claude = args.use_claude and not args.no_claude
    
    # Initialize processor
    processor = DocumentProcessor(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        use_claude=use_claude
    )
    
    # Process files
    if args.test_file:
        # Process single test file
        test_path = Path(args.input_dir) / args.test_file
        if test_path.exists():
            result = processor.process_file(test_path)
            if result:
                print(f"\nTest file processed successfully:")
                print(f"  Total chunks: {result['total_chunks']}")
                print(f"  Content types: {result['content_type_distribution']}")
        else:
            print(f"Test file not found: {test_path}")
    else:
        # Process all matching files
        processor.process_all_files(pattern=args.pattern)


if __name__ == "__main__":
    main()