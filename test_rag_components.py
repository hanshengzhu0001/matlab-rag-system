#!/usr/bin/env python3
"""
Test script to validate RAG system components before full build.
Tests embeddings, chunking, and basic functionality.
"""

import sys
import time
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")

    try:
        # Test langchain imports
        from langchain_community.document_loaders import BSHTMLLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        print("✅ LangChain imports successful")

        # Test other imports
        import chromadb
        import bs4
        import sentence_transformers
        print("✅ Other package imports successful")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_embedding_model():
    """Test the BGE embedding model."""
    print("\n🧮 Testing BGE embedding model...")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        # Test CPU first (faster for testing)
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},  # Use CPU for testing
            encode_kwargs={"normalize_embeddings": True}
        )

        # Test single embedding
        test_text = "MATLAB matrix multiplication example"
        embedding = embedding_model.embed_query(test_text)

        print(f"✅ Embedding successful: {len(embedding)} dimensions")

        # Test batch embedding
        test_texts = [
            "How to plot data in MATLAB",
            "Matrix operations syntax",
            "File input/output functions"
        ]
        embeddings = embedding_model.embed_documents(test_texts)
        print(f"✅ Batch embedding successful: {len(embeddings)} texts")

        return True
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False


def test_html_loader():
    """Test HTML loading on sample files."""
    print("\n📄 Testing HTML document loading...")

    try:
        from langchain_community.document_loaders import BSHTMLLoader

        # Test on a sample file
        test_file = "matlab_documents/matlab/arithmetic-operators.html"

        if not Path(test_file).exists():
            print(f"⚠️  Test file not found: {test_file}")
            return False

        loader = BSHTMLLoader(test_file, open_encoding="utf-8", bs_kwargs={'features': 'html.parser'})
        documents = loader.load()

        if not documents:
            print("❌ No documents loaded")
            return False

        doc = documents[0]
        content_length = len(doc.page_content)
        title = doc.metadata.get('title', 'No title')

        print(f"✅ HTML loading successful:")
        print(f"   Title: {title}")
        print(f"   Content length: {content_length:,} characters")

        return True
    except Exception as e:
        print(f"❌ HTML loading test failed: {e}")
        return False


def test_chroma_db():
    """Test ChromaDB basic functionality."""
    print("\n🗄️  Testing ChromaDB...")

    try:
        import chromadb
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        # Create temporary directory for testing
        test_dir = Path("./test_chroma")
        test_dir.mkdir(exist_ok=True)

        # Create embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Create test documents
        test_docs = [
            Document(page_content="MATLAB plot function creates visualizations", metadata={"source": "test1"}),
            Document(page_content="Matrix multiplication using * operator", metadata={"source": "test2"}),
            Document(page_content="File I/O with load and save functions", metadata={"source": "test3"})
        ]

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=test_docs,
            embedding=embedding_model,
            persist_directory=str(test_dir)
        )

        # Test search
        results = vectorstore.similarity_search("plot data", k=2)
        print(f"✅ ChromaDB test successful: {len(results)} results found")

        # Cleanup
        import shutil
        shutil.rmtree(test_dir)

        return True
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False


def test_ollama_connection():
    """Test Ollama connection (if available)."""
    print("\n🤖 Testing Ollama connection...")

    try:
        from langchain.llms import Ollama

        llm = Ollama(
            model="codellama:13b",
            timeout=10  # Short timeout for testing
        )

        # Try a simple test
        response = llm.invoke("Hello")

        if response:
            print("✅ Ollama connection successful")
            return True
        else:
            print("⚠️  Ollama connected but no response")
            return False

    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        print("   (This is OK - Ollama will be needed for full system)")
        return False


def estimate_full_build():
    """Estimate time and resources for full build."""
    print("\n⏱️  Full Build Estimation:")

    # Based on our earlier analysis
    html_files = 12389
    avg_chunk_time = 0.05  # seconds per file (estimated)
    embedding_time_per_chunk = 0.2  # seconds (CPU estimate)

    estimated_chunks = html_files * 3  # ~3 chunks per file average

    processing_time = html_files * avg_chunk_time / 60  # minutes
    embedding_time = estimated_chunks * embedding_time_per_chunk / 60  # minutes

    total_time = processing_time + embedding_time

    print(f"   HTML files: {html_files:,}")
    print(f"   Estimated chunks: {estimated_chunks:,}")
    print(f"   Processing time: {processing_time:.1f} minutes")
    print(f"   Embedding time: {embedding_time:.1f} minutes")
    print(f"   Total estimated: {total_time:.1f} minutes")
    print("   Note: GPU acceleration will significantly reduce embedding time")


def main():
    """Run all component tests."""
    print("🧪 MATLAB RAG System Component Tests")
    print("=" * 50)

    tests = [
        ("Package Imports", test_imports),
        ("BGE Embeddings", test_embedding_model),
        ("HTML Loading", test_html_loader),
        ("ChromaDB", test_chroma_db),
        ("Ollama (Optional)", test_ollama_connection)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Result: {status}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}")

    print(f"\nPassed: {passed}/{total}")

    if passed >= 4:  # Core components (excluding optional Ollama)
        print("✅ Core components ready! You can proceed with building the RAG database.")
        estimate_full_build()
        print("\n🚀 Next step: python build_rag.py")
    else:
        print("❌ Some core components failed. Please check the errors above.")
        print("   You may need to install missing packages or fix configuration issues.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
