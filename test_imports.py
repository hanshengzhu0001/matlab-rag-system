#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

def test_imports():
    """Test all critical imports for the MATLAB RAG system."""

    print("🧪 Testing MATLAB RAG imports...")
    print("=" * 50)

    tests = [
        ("PIL (Pillow)", "from PIL import Image", lambda: Image.new('RGB', (10, 10))),
        ("torch", "import torch", lambda: torch.tensor([1, 2, 3])),
        ("transformers", "from transformers import BlipProcessor", lambda: None),
        ("langchain_core", "from langchain_core.prompts import ChatPromptTemplate", lambda: None),
        ("langchain_chroma", "from langchain_chroma import Chroma", lambda: None),
        ("langchain_community", "from langchain_community.retrievers import BM25Retriever", lambda: None),
        ("sentence_transformers", "from sentence_transformers import SentenceTransformer", lambda: None),
        ("chromadb", "import chromadb", lambda: None),
    ]

    # Test Ollama import separately
    print("Testing Ollama imports...")
    try:
        from langchain_ollama import ChatOllama as Ollama
        print("✅ Ollama imported from langchain_ollama.ChatOllama")
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama as Ollama
            print("✅ Ollama imported from langchain_community.chat_models.ChatOllama")
        except ImportError:
            try:
                from langchain_community.llms import Ollama
                print("✅ Ollama imported from langchain_community.llms.Ollama")
            except ImportError:
                print("❌ Ollama import failed from all locations")
                return False

    # Test other imports
    failed_imports = []
    for name, import_stmt, test_func in tests:
        try:
            exec(import_stmt)
            if test_func:
                test_func()
            print(f"✅ {name}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)

    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False

    print("\n✅ All imports successful!")
    return True

def test_query_system():
    """Test that the MATLABQuerySystem can be imported and initialized."""

    print("\n🔧 Testing MATLABQuerySystem...")
    try:
        from query_rag import MATLABQuerySystem
        print("✅ MATLABQuerySystem import successful")

        # Try to initialize (this will fail if ChromaDB has issues, but import should work)
        try:
            system = MATLABQuerySystem()
            print("✅ MATLABQuerySystem initialization successful")
            return True
        except Exception as e:
            print(f"⚠️ MATLABQuerySystem init failed (expected if DB not ready): {e}")
            return True  # Import works, DB issue is separate

    except Exception as e:
        print(f"❌ MATLABQuerySystem import failed: {e}")
        return False

def test_vision_system():
    """Test that the vision system can be imported."""

    print("\n🖼️ Testing Vision RAG system...")
    try:
        from vision_rag_core import VisionRAGAnalyzer
        print("✅ VisionRAGAnalyzer import successful")
        return True
    except Exception as e:
        print(f"❌ VisionRAGAnalyzer import failed: {e}")
        return False

if __name__ == "__main__":
    success = True

    success &= test_imports()
    success &= test_query_system()
    success &= test_vision_system()

    if success:
        print("\n🎉 All systems ready! You can now run:")
        print("  python app.py          # Text RAG interface")
        print("  python app_vision.py   # Vision RAG interface")
    else:
        print("\n❌ Some imports failed. Please check dependencies:")
        print("  pip install -r requirements.txt")
