#!/usr/bin/env python3
"""
MATLAB RAG Query System with CodeLlama Integration

This script loads the pre-built RAG database and provides a query interface
for answering MATLAB coding questions using CodeLlama for generation.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MATLABQuerySystem:
    """Query interface for MATLAB RAG system using CodeLlama."""

    def __init__(self, persist_dir: str = "./chroma_db_matlab"):
        """
        Initialize the query system.

        Args:
            persist_dir: Directory containing the persisted vector database
        """
        self.persist_dir = Path(persist_dir)

        if not self.persist_dir.exists():
            raise ValueError(f"Database directory not found: {self.persist_dir}")

        logger.info(f"Loading MATLAB RAG system from: {self.persist_dir}")

        # Load the existing vector database
        self.vectorstore = self._load_vectorstore()
        self.retriever = self._create_retriever()

        # Initialize the RAG chain
        self.chain = self._create_rag_chain()

        logger.info("✅ MATLAB RAG query system ready!")


    def _load_vectorstore(self) -> Chroma:
        """Load the persisted ChromaDB vector store."""
        logger.info("Loading vector database...")

        # Recreate embedding model with same configuration
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 8}
        )

        # Load existing database
        vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=embedding_model,
            collection_name="matlab_docs"
        )

        # Get collection info
        collection = vectorstore._collection
        metadata = collection.metadata or {}

        logger.info(f"Database loaded successfully:")
        logger.info(f"  Collection: {collection.name}")
        logger.info(f"  Total documents: {collection.count()}")
        logger.info(f"  Created: {metadata.get('created_at', 'Unknown')}")
        logger.info(f"  Embedding model: {metadata.get('embedding_model', 'Unknown')}")

        return vectorstore


    def _create_retriever(self):
        """Create the retrievers from the loaded database."""
        logger.info("Building retrievers...")

        # Get all documents from the collection for BM25
        collection = self.vectorstore._collection
        results = collection.get(include=['documents', 'metadatas'])

        # Convert to Document objects for BM25
        from langchain_core.documents import Document
        chunks = []
        for i, doc_content in enumerate(results['documents']):
            metadata = results['metadatas'][i] if results['metadatas'] else {}
            chunks.append(Document(page_content=doc_content, metadata=metadata))

        # Dense retriever
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # Sparse retriever (BM25)
        sparse_retriever = BM25Retriever.from_documents(chunks)
        sparse_retriever.k = 5

        logger.info(f"Retrievers ready with {len(chunks)} chunks")
        return {
            "dense": dense_retriever,
            "sparse": sparse_retriever,
            "hybrid": True
        }


    def _create_rag_chain(self):
        """Create the RAG chain with CodeLlama for generation."""
        logger.info("Initializing CodeLlama model...")

        # Initialize CodeLlama via Ollama (CPU-optimized)
        llm = OllamaLLM(
            model="codellama:7b",   # 7B model for CPU/laptop usage
            temperature=0.1,        # Low temperature for deterministic code
            num_gpu=0,              # Disable GPU usage
            num_ctx=2048,           # Smaller context window for CPU
            num_thread=4,           # CPU threads (adjust based on your CPU cores)
            timeout=120,            # Timeout for generation
        )

        # Test LLM connection
        try:
            test_response = llm.invoke("Hello, MATLAB!")
            logger.info("CodeLlama model ready!")
        except Exception as e:
            logger.warning(f"LLM test failed: {e}")
            logger.warning("Make sure Ollama is running with 'codellama:13b' model")
            logger.warning("Run: ollama serve & ollama pull codellama:13b")

        # Optimized system prompt for MATLAB assistance
        system_prompt = """You are an expert MATLAB assistant. Use ONLY the following context to answer.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Generate COMPLETE, runnable MATLAB code (.m file)
2. Include clear comments explaining key steps
3. If context doesn't contain enough information, say "I cannot generate code for this based on the documentation."
4. Format output as a single code block starting with ```matlab
5. Focus on syntactically correct, efficient MATLAB code
6. Include error handling where appropriate

ANSWER:"""

        prompt = ChatPromptTemplate.from_template(system_prompt)

        # Build the RAG chain using dense retriever (primary)
        chain = (
            {"context": self.retriever["dense"], "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("RAG chain created with CodeLlama integration")
        return chain


    def query(self, question: str, show_context: bool = False) -> Dict[str, Any]:
        """
        Query the MATLAB RAG system.

        Args:
            question: User's MATLAB coding question
            show_context: Whether to include retrieved context in response

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")

        start_time = time.time()

        try:
            # Get answer from RAG chain
            answer = self.chain.invoke(question)
            query_time = time.time() - start_time

            # Get retrieval context if requested
            context_docs = None
            if show_context:
                context_docs = self.retriever.get_relevant_documents(question)

            result = {
                "question": question,
                "answer": answer,
                "query_time": query_time,
                "success": True,
                "context_docs": context_docs if show_context else None
            }

            logger.info(".2f")
            return result

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Query failed: {str(e)}")

            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "query_time": error_time,
                "success": False,
                "context_docs": None
            }


def test_queries():
    """Run a series of test queries to validate the system."""
    print("🧪 Testing MATLAB RAG System")
    print("-" * 40)

    try:
        # Initialize system
        system = MATLABQuerySystem()

        # Test queries
        test_questions = [
            "How do I read a CSV file and plot the first two columns?",
            "How to create a matrix and calculate its determinant?",
            "How do I use the plot function to create a line graph?",
            "How to perform element-wise multiplication of matrices?"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n📋 Test Query {i}: {question}")
            print("-" * 50)

            result = system.query(question, show_context=False)

            print(f"⏱️  Query time: {result['query_time']:.2f}s")
            print(f"📝 Answer preview: {result['answer'][:200]}...")
            print("✅ Query successful!" if result['success'] else "❌ Query failed!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")


def interactive_mode():
    """Run in interactive mode for manual queries."""
    print("💬 MATLAB RAG Interactive Mode")
    print("Type your MATLAB questions (or 'quit' to exit)")
    print("-" * 50)

    try:
        system = MATLABQuerySystem()

        while True:
            question = input("\n🤔 Your MATLAB question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            print("\n🔍 Processing...")
            result = system.query(question)

            print(f"\n📝 Answer ({result['query_time']:.2f}s):")
            print("-" * 50)
            print(result['answer'])
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def main():
    """Main function with mode selection."""
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "interactive"

    print("🚀 MATLAB RAG Query System")
    print("=" * 40)

    if mode == "test":
        test_queries()
    elif mode == "interactive":
        interactive_mode()
    else:
        print("Usage: python query_rag.py [test|interactive]")
        print("  test: Run automated test queries")
        print("  interactive: Manual query mode (default)")


if __name__ == "__main__":
    main()
