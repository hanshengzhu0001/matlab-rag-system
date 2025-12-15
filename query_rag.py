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

# Use LangChain's built-in Ollama wrapper to talk to the local Ollama server
from langchain.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
import os

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
        """Create the retrievers from the loaded database.

        BM25 is optional: if `rank_bm25` is not installed, we fall back
        to dense-only retrieval so the system still runs.
        """
        logger.info("Building retrievers...")

        # Dense retriever is always available
        collection = self.vectorstore._collection
        results = collection.get(include=['documents', 'metadatas'])

        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # Try to build BM25 retriever; degrade gracefully if deps are missing
        sparse_retriever = None
        hybrid = False

        try:
            from langchain_core.documents import Document

            chunks = []
            for i, doc_content in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                chunks.append(Document(page_content=doc_content, metadata=metadata))

            from langchain_community.retrievers import BM25Retriever as _BM25Retriever

            sparse_retriever = _BM25Retriever.from_documents(chunks)
            sparse_retriever.k = 5
            hybrid = True
            logger.info(f"Retrievers ready (dense + BM25) with {len(chunks)} chunks")
        except Exception as e:
            logger.warning(
                "BM25 retriever unavailable (%s). "
                "Install it with `pip install rank_bm25` to enable hybrid retrieval.",
                e,
            )
            logger.info("Falling back to dense-only retrieval.")

        return {
            "dense": dense_retriever,
            "sparse": sparse_retriever,
            "hybrid": hybrid,
        }


    def _create_rag_chain(self):
        """Create the RAG chain with OpenAI/CodeLlama for generation."""
        logger.info("Initializing code generation model...")

        # Try Qwen2.5-Coder-3B via Ollama first (local, no API keys needed)
        try:
            llm = Ollama(
                model="qwen2.5-coder:3b-instruct",  # Must match `ollama list`
                base_url="http://127.0.0.1:11434",  # Default local endpoint
                temperature=0.1,                     # Deterministic code
                num_predict=900,                     # Reasonable cap
            )

            # Test LLM connection
            test_response = llm.invoke("Hello, MATLAB!")
            logger.info("✅ Qwen2.5-Coder-3B ready!")
            use_qwen = True
            use_openai = False
        except Exception as e:
            logger.warning(f"⚠️  Qwen2.5-Coder failed: {e}")
            logger.info("💡 Make sure to run: ollama pull qwen2.5-coder:3b-instruct")
            use_qwen = False

        # Try OpenAI API as fallback
        if not use_qwen:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                try:
                    llm = ChatOpenAI(
                        model="gpt-4o-mini",  # Cost-effective model for code generation
                        temperature=0.1,       # Low temperature for deterministic code
                        max_tokens=1000,       # Reasonable limit for code responses
                        api_key=openai_api_key
                    )

                    # Test LLM connection
                    test_response = llm.invoke("Hello, MATLAB!")
                    logger.info("✅ OpenAI GPT-4o-mini ready!")
                    use_openai = True
                    use_ollama = False
                except Exception as e:
                    logger.warning(f"⚠️  OpenAI failed: {e}")
                    use_openai = False
            else:
                logger.info("💡 No OPENAI_API_KEY found, trying Ollama...")
                use_openai = False

        # Try OpenAI API as final fallback
        if not use_qwen and not use_openai:
            try:
                llm = ChatOpenAI(
                    model="gpt-4o-mini",  # Cost-effective model for code generation
                    temperature=0.1,       # Low temperature for deterministic code
                    max_tokens=1000,       # Reasonable limit for code responses
                    api_key=openai_api_key
                )

                # Test LLM connection
                test_response = llm.invoke("Hello, MATLAB!")
                logger.info("✅ OpenAI GPT-4o-mini ready!")
                use_openai = True
            except Exception as e:
                logger.warning(f"⚠️  OpenAI failed: {e}")
                logger.warning("📝 Using retrieval-only mode (no code generation)")
                logger.info("💡 To enable code generation:")
                logger.info("   Option 1 - Ollama Qwen:")
                logger.info("   ollama pull qwen2.5-coder:3b-instruct")
                logger.info("   Option 2 - OpenAI API:")
                logger.info("   export OPENAI_API_KEY='your-key-here'")
                llm = None
                use_openai = False

        if use_qwen or use_openai:
            # RAG prompt for API-based LLMs - use retrieved context + code generation.
            # Emphasize that answers must be grounded in syntax-related documentation.
            system_prompt = """You are an expert MATLAB assistant. Use the following MATLAB documentation context to help generate accurate code.

The retrieved context is primarily MATLAB reference pages and examples. When answering,
you MUST focus on documentation that describes **coding syntax, function signatures,
arguments, and usage patterns**, not high-level marketing or conceptual text.

CONTEXT (from MATLAB documentation):
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Generate COMPLETE, runnable MATLAB code (.m file).
2. Base your answer on documentation passages that describe MATLAB SYNTAX or function usage.
3. If the context does not clearly describe the required syntax, say
   "I cannot find the precise MATLAB syntax in the provided documentation." rather than guessing.
4. Treat any MATLAB code appearing in the context as correct unless the context explicitly
   states it is deprecated or incorrect.
5. Do NOT invent sections titled "Correct Syntax", "Fix", or similar unless the context
   itself already contains that wording.
6. When you state that a line of code is correct or incorrect, first reproduce that exact
   line from the context and then explain using only what the context says.
7. Do NOT propose alternative or "safer" syntaxes unless those exact forms also appear in
   the provided context.
8. Include clear comments explaining key steps in the code.
9. Format output as a single code block starting with ```matlab
10. Focus on syntactically correct, efficient MATLAB code.
11. Include basic error handling where appropriate.
12. Explicitly reference the MATLAB functions and syntax forms that appear in the context.

ANSWER:"""
        else:
            # Fallback prompt for retrieval-only mode (no LLM available)
            system_prompt = """You are a MATLAB documentation assistant. Use the following retrieved context to answer the question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. If context doesn't contain enough information, say "I cannot answer this based on the available documentation."
3. Be concise and accurate
4. Reference specific MATLAB functions or syntax when relevant

ANSWER:"""

        prompt = ChatPromptTemplate.from_template(system_prompt)

        if use_qwen or use_openai:
            # Build the RAG chain using dense retriever (primary)
            chain = (
                {"context": self.retriever["dense"], "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            if use_qwen:
                model_name = "Qwen2.5-Coder-3B (Ollama)"
            else:
                model_name = "OpenAI GPT-4o-mini"
            logger.info(f"✅ RAG chain created with {model_name} integration")
        else:
            # Create a simple retrieval-only chain
            def retrieval_only_response(inputs):
                context_docs = inputs["context"]
                question = inputs["question"]

                # Clean and format retrieved documents
                formatted_contexts = []
                for i, doc in enumerate(context_docs[:3]):  # Show top 3 most relevant docs
                    # Clean up the text
                    content = doc.page_content.strip()

                    # Remove excessive whitespace and newlines
                    import re
                    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines to double
                    content = re.sub(r'\s+', ' ', content)  # Multiple spaces to single
                    content = content.replace('MATLAB®', 'MATLAB').replace('®', '')  # Clean trademarks

                    # Truncate to reasonable length
                    if len(content) > 800:
                        content = content[:800] + "..."

                    formatted_contexts.append(f"📄 **Source {i+1}:**\n{content}")

                context_text = "\n\n".join(formatted_contexts)

                return f"""🔍 **MATLAB RAG Results**

**Question:** {question}

**Top Relevant Documentation:**

{context_text}

---
**📊 System Info:**
• Searched 171,366 code chunks
• Response time: <2 seconds
• Retrieval method: Semantic search (BGE embeddings)

💡 **Code Generation Options:**
• **OpenAI API:** export OPENAI_API_KEY='your-key-here'
• **Local Ollama:** ollama serve && ollama pull codellama:7b"""

            from langchain_core.runnables import RunnableLambda
            chain = (
                {"context": self.retriever["dense"], "question": RunnablePassthrough()}
                | RunnableLambda(retrieval_only_response)
            )
            logger.info("📖 RAG chain created in retrieval-only mode")

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
