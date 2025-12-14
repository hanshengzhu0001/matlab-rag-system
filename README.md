# MATLAB-RAG-System

An intelligent RAG (Retrieval-Augmented Generation) system for MATLAB documentation that provides contextual code assistance using CodeLlama.

## Features

- **Document Processing**: Automated parsing and chunking of MATLAB documentation
- **Vector Embeddings**: BGE-base-en-v1.5 embeddings for semantic search
- **Hybrid Retrieval**: Combines dense (semantic) and sparse (keyword) search
- **Code Generation**: CodeLlama integration for MATLAB code assistance
- **CPU/GPU Support**: Optimized for both laptop and workstation deployment

## Architecture

- `build_rag.py`: Database builder with optimized batch processing
- `query_rag.py`: Query interface with CodeLlama integration
- Custom embedding classes for compatibility across different hardware
- ChromaDB for efficient vector storage and retrieval

## Quick Start

```bash
# Set up environment
conda create -n matlab_rag python=3.11 -y
conda activate matlab_rag
pip install langchain langchain-community langchain-chroma pypdf beautifulsoup4 sentence-transformers chromadb ollama flask langchain-ollama lxml

# Build the database
python build_rag.py

# Query the system
python query_rag.py
```

## Performance

- Processes 12,000+ MATLAB documentation files
- Creates 170,000+ searchable chunks
- Sub-2-second query response times
- Optimized for both CPU and GPU workflows

## Use Cases

- MATLAB code assistance and debugging
- Function discovery and usage examples
- Technical documentation search
- Code generation from natural language descriptions

## Technology Stack

- **Embeddings**: BAAI/bge-base-en-v1.5 (768-dim)
- **LLM**: CodeLlama 7B/13B via Ollama
- **Vector DB**: ChromaDB with HNSW indexing
- **Framework**: LangChain for RAG orchestration
- **Processing**: Sentence Transformers + PyTorch

## Project Structure

```
matlab_rag/
├── build_rag.py          # Database builder
├── query_rag.py          # Query interface
├── test_rag_components.py # Component testing
├── scan_file_types.py     # Document analysis
├── simple_html_test.py    # HTML parsing tests
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── matlab_documents/    # MATLAB documentation (not in repo)
```

## Development Notes

- Custom BGE embedding class bypasses LangChain wrapper compatibility issues
- Batch processing (100 chunks/batch) prevents memory overflow
- CPU-optimized settings for laptop deployment
- Progress logging for long-running operations
