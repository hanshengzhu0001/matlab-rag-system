#!/usr/bin/env python3
"""
Optimized RAG Database Builder for MATLAB Documentation

This script processes the MATLAB documentation and creates a vector database
optimized for technical documentation retrieval and code generation.

Features:
- GPU-accelerated embeddings (BGE-base-en-v1.5)
- Hybrid retrieval (dense + sparse)
- Intelligent chunking for technical docs
- Progress tracking for large datasets
- Optimized for RTX 5090 GPU
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
# Note: EnsembleRetriever not available in current langchain version
# We'll implement a simple ensemble approach
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MATLABRAGBuilder:
    """Optimized RAG system builder for MATLAB documentation."""

    def __init__(self, docs_path: str = "matlab_documents/matlab",
                 persist_dir: str = "./chroma_db_matlab",
                 chunk_size: int = 600,
                 chunk_overlap: int = 100):
        """
        Initialize the RAG builder.

        Args:
            docs_path: Path to MATLAB documentation directory
            persist_dir: Directory to persist the vector database
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.docs_path = Path(docs_path)
        self.persist_dir = Path(persist_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Validate paths
        if not self.docs_path.exists():
            raise ValueError(f"Documentation path does not exist: {self.docs_path}")

        # Create persist directory if it doesn't exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RAG builder:")
        logger.info(f"  Docs path: {self.docs_path}")
        logger.info(f"  Persist dir: {self.persist_dir}")
        logger.info(f"  Chunk size: {self.chunk_size}")
        logger.info(f"  Chunk overlap: {self.chunk_overlap}")


    def load_matlab_docs(self, max_files: int = None) -> List[Document]:
        """
        Load MATLAB HTML documentation with optimized settings.

        Args:
            max_files: Maximum number of files to load (for testing)

        Returns:
            List of loaded documents
        """
        logger.info("Starting document loading...")

        # Configure loader with optimized settings
        loader = DirectoryLoader(
            str(self.docs_path),
            glob="**/*.html",  # Focus on HTML files only
            loader_cls=BSHTMLLoader,
            loader_kwargs={
                "open_encoding": "utf-8",
                "get_text_separator": "\n",  # Preserve some structure
            },
            show_progress=True,
            use_multithreading=True,  # Parallel loading
            max_concurrency=4,  # Limit concurrency to avoid memory issues
            silent_errors=True  # Continue on individual file errors
        )

        start_time = time.time()
        documents = loader.load()

        # Filter out very small or empty documents
        filtered_docs = []
        for doc in documents:
            if len(doc.page_content.strip()) > 100:  # Minimum content length
                filtered_docs.append(doc)

        load_time = time.time() - start_time

        logger.info(f"Document loading completed:")
        logger.info(f"  Raw documents: {len(documents)}")
        logger.info(f"  Filtered documents: {len(filtered_docs)}")
        logger.info(f"  Loading time: {load_time:.2f} seconds")
        logger.info(".2f")

        return filtered_docs


    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents with intelligent splitting for technical content.

        Args:
            documents: Raw documents to chunk

        Returns:
            List of document chunks
        """
        logger.info("Starting document chunking...")

        # Optimized text splitter for technical documentation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentences
                "; ",    # Semicolons (common in MATLAB)
                ": ",    # Colons
                " ",     # Words
                ""       # Characters
            ],
            length_function=len,
            keep_separator=True,  # Preserve separators for context
            add_start_index=True,  # Track chunk positions
        )

        start_time = time.time()
        chunks = text_splitter.split_documents(documents)
        chunk_time = time.time() - start_time

        # Analyze chunk statistics
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        max_length = max(chunk_lengths) if chunk_lengths else 0
        min_length = min(chunk_lengths) if chunk_lengths else 0

        logger.info(f"Document chunking completed:")
        logger.info(f"  Total chunks: {len(chunks)}")
        logger.info(f"  Average chunk length: {avg_length:.1f} characters")
        logger.info(f"  Max chunk length: {max_length} characters")
        logger.info(f"  Min chunk length: {min_length} characters")
        logger.info(f"  Chunking time: {chunk_time:.2f} seconds")
        logger.info(".2f")

        return chunks


    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """
        Create ChromaDB vector store with GPU-accelerated BGE embeddings.

        Args:
            chunks: Document chunks to embed and store

        Returns:
            Chroma vector store instance
        """
        logger.info("Creating vector database with BGE embeddings...")

        # Custom embedding class to avoid langchain wrapper issues
        from sentence_transformers import SentenceTransformer

        class CustomBGEEmbeddings:
            def __init__(self):
                self.model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')

            def embed_documents(self, texts):
                """Embed multiple documents"""
                embeddings = []
                for text in texts:
                    embedding = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
                    embeddings.append(embedding.tolist())
                return embeddings

            def embed_query(self, text):
                """Embed a single query"""
                return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()

        embedding_model = CustomBGEEmbeddings()

        # Test embedding model
        logger.info("Testing embedding model...")
        test_text = "This is a test for MATLAB documentation embedding."
        test_embedding = embedding_model.embed_query(test_text)
        logger.info(f"  Embedding dimension: {len(test_embedding)}")
        logger.info("  Embedding model ready!")

        # Create ChromaDB instance first
        vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            collection_name="matlab_docs",
            embedding_function=embedding_model,
            collection_metadata={
                "created_at": time.time(),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "total_chunks": len(chunks)
            }
        )

        # Check if database already has significant data (resume scenario)
        existing_docs = 0
        skip_processing = False

        try:
            existing_docs = vectorstore._collection.count() if hasattr(vectorstore, '_collection') else 0
            if existing_docs > len(chunks) * 0.5:  # If we have >50% of expected chunks
                logger.info(f"Database already contains {existing_docs} embeddings (>50% complete)")
                logger.info("Will continue processing remaining chunks...")
                start_chunk = existing_docs  # Resume from where we left off
            else:
                start_chunk = 0
        except (AttributeError, Exception):
            start_chunk = 0

        # Add documents in small batches to avoid memory issues
        start_time = time.time()
        batch_size = 100  # Small batches

        if start_chunk < len(chunks):
            logger.info(f"Processing chunks {start_chunk} to {len(chunks)} in batches of {batch_size}...")
            processed_count = start_chunk

            for i in range(start_chunk, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                try:
                    vectorstore.add_documents(batch)
                    processed_count += len(batch)
                    if (processed_count // batch_size) % 10 == 0:  # Progress every 10 batches
                        logger.info(f"  Processed {processed_count}/{len(chunks)} chunks...")
                except Exception as e:
                    logger.warning(f"  Batch {(i//batch_size)} failed: {e}")
                    continue
        else:
            logger.info("All chunks already processed!")

        creation_time = time.time() - start_time

        # Note: ChromaDB automatically persists data, no manual persist() call needed

        # Get final count
        final_count = vectorstore._collection.count() if hasattr(vectorstore, '_collection') else len(chunks)

        logger.info(f"Vector database updated successfully:")
        logger.info(f"  Total chunks available: {len(chunks)}")
        logger.info(f"  Embeddings in database: {final_count}")
        logger.info(f"  Completion: {final_count/len(chunks)*100:.1f}%")
        logger.info(f"  Persist directory: {self.persist_dir}")
        if creation_time > 0:
            logger.info(f"  Processing time: {creation_time:.2f} seconds")
        logger.info(".2f")

        return vectorstore


    def create_hybrid_retriever(self, vectorstore: Chroma, chunks: List[Document]):
        """
        Create hybrid retriever combining dense (vector) and sparse (keyword) search.
        Since EnsembleRetriever is not available, we'll return both retrievers separately.

        Args:
            vectorstore: ChromaDB vector store
            chunks: Document chunks for BM25

        Returns:
            Dictionary with both retrievers
        """
        logger.info("Creating hybrid retrievers...")

        # Dense retriever (semantic search)
        dense_retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5,  # Top 5 semantic matches
                "score_threshold": 0.1,  # Minimum relevance threshold
            }
        )

        # Sparse retriever (keyword search) - excellent for function names
        logger.info("Building BM25 index...")
        sparse_retriever = BM25Retriever.from_documents(
            documents=chunks,
            k=5,  # Top 5 keyword matches
        )

        logger.info("Retrievers created:")
        logger.info("  Dense retriever: BGE embeddings")
        logger.info("  Sparse retriever: BM25 keywords")
        logger.info("  Note: EnsembleRetriever not available - using separate retrievers")

        return {
            "dense": dense_retriever,
            "sparse": sparse_retriever,
            "hybrid": True  # Flag to indicate hybrid approach
        }

    def add_visual_knowledge_to_db(self, vectorstore, json_path="visual_knowledge.json"):
        """
        Add visual knowledge from processed images to the vector database.

        Args:
            vectorstore: ChromaDB vectorstore to add documents to
            json_path: Path to the visual knowledge JSON file
        """
        import json
        from pathlib import Path

        json_path = Path(json_path)
        if not json_path.exists():
            logger.warning(f"Visual knowledge file not found: {json_path}")
            logger.info("💡 Generate visual knowledge first by running:")
            logger.info("   1. matlab -batch extract_ocr")
            logger.info("   2. python generate_blip_descriptions.py")
            return

        logger.info(f"🖼️  Loading visual knowledge from {json_path}")

        # Load visual knowledge data
        with open(json_path, 'r', encoding='utf-8') as f:
            visual_data = json.load(f)

        logger.info(f"📊 Found {len(visual_data)} visual documents to process")

        # Convert to LangChain Document objects
        visual_docs = []
        processed_count = 0

        for item in visual_data:
            try:
                # Use combined_text if available, otherwise fall back to OCR or BLIP
                content = item.get('combined_text', '')
                if not content:
                    content = item.get('ocr_text', '') or item.get('blip_caption', '')

                if not content or len(content.strip()) < 10:
                    continue  # Skip empty or very short content

                # Create document with metadata
                metadata = {
                    "source": item['path'],
                    "type": "image",
                    "confidence": item.get('confidence', 0),
                    "has_ocr": bool(item.get('ocr_text', '').strip()),
                    "has_blip": bool(item.get('blip_caption', '').strip())
                }

                doc = Document(page_content=content, metadata=metadata)
                visual_docs.append(doc)
                processed_count += 1

            except Exception as e:
                logger.warning(f"Error processing visual item {item.get('path', 'unknown')}: {e}")
                continue

        if not visual_docs:
            logger.warning("No valid visual documents found to add")
            return

        # Add to vectorstore in batches to avoid memory issues
        batch_size = 100
        total_added = 0

        logger.info(f"📥 Adding {len(visual_docs)} visual documents to database...")

        for i in range(0, len(visual_docs), batch_size):
            batch = visual_docs[i:i + batch_size]

            try:
                vectorstore.add_documents(batch)
                total_added += len(batch)
                logger.info(f"   → Added batch {i//batch_size + 1}: {len(batch)} documents")

            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                continue

        logger.info(f"✅ Added {total_added} visual documents to database")
        logger.info("🔍 Database now contains both text and visual knowledge!")


    def build_database(self, max_files: int = None, include_visual: bool = False):
        """
        Complete pipeline to build the RAG database.

        Args:
            max_files: Maximum files to process (for testing)
            include_visual: Whether to include visual knowledge from images

        Returns:
            Configured hybrid retriever
        """
        logger.info("=" * 60)
        logger.info("STARTING MATLAB RAG DATABASE BUILD")
        logger.info("=" * 60)

        total_start_time = time.time()

        try:
            # Phase 1: Load documents
            logger.info("\n📄 PHASE 1: Loading Documents")
            raw_docs = self.load_matlab_docs(max_files)

            if not raw_docs:
                raise ValueError("No documents were loaded!")

            # Phase 2: Chunk documents
            logger.info("\n✂️  PHASE 2: Chunking Documents")
            chunks = self.chunk_documents(raw_docs)

            if not chunks:
                raise ValueError("No chunks were created!")

            # Phase 3: Create vector database
            logger.info("\n🗄️  PHASE 3: Creating Vector Database")
            vectorstore = self.create_vectorstore(chunks)

            # Phase 3.5: Add visual knowledge (optional)
            if include_visual:
                logger.info("\n🖼️  PHASE 3.5: Adding Visual Knowledge")
                self.add_visual_knowledge_to_db(vectorstore)

            # Phase 4: Create hybrid retriever
            logger.info("\n🔍 PHASE 4: Building Hybrid Retriever")
            retriever = self.create_hybrid_retriever(vectorstore, chunks)

            total_time = time.time() - total_start_time

            logger.info("\n" + "=" * 60)
            logger.info("✅ MATLAB RAG DATABASE BUILD COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(".2f")
            if include_visual:
                logger.info("🎨 Visual knowledge: INCLUDED")
            else:
                logger.info("📝 Text-only knowledge")
            logger.info(f"Database location: {self.persist_dir}")
            logger.info("Ready for query processing!")

            return retriever

        except Exception as e:
            logger.error(f"Build failed with error: {str(e)}")
            raise


def main():
    """Main function to run the RAG database build."""

    # Configuration
    DOCS_PATH = "matlab_documents/matlab"  # Focus on core MATLAB docs
    PERSIST_DIR = "./chroma_db_matlab"
    MAX_FILES = None  # Set to small number for testing (e.g., 100)

    print("🚀 MATLAB RAG Database Builder")
    print("=" * 50)
    print(f"Documentation path: {DOCS_PATH}")
    print(f"Persist directory: {PERSIST_DIR}")
    print(f"Max files (None = all): {MAX_FILES}")
    print()

    # Build the database (start with 500 files for testing)
    builder = MATLABRAGBuilder(
        docs_path=DOCS_PATH,
        persist_dir=PERSIST_DIR,
        chunk_size=600,    # Optimized for technical docs
        chunk_overlap=100  # Good context preservation
    )

    # Override MAX_FILES for testing
    MAX_FILES = 10  # Start with 10 files for quick testing

    try:
        retriever = builder.build_database(max_files=MAX_FILES)
        print("\n✅ Success! RAG database ready for MATLAB queries.")
        print("Next: Run query_rag.py to test the system.")

    except Exception as e:
        print(f"\n❌ Build failed: {str(e)}")
        print("Check build_rag.log for detailed error information.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
