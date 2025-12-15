#!/usr/bin/env python3
"""
Demo script showing MATLAB RAG with OpenAI code generation
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from query_rag import MATLABQuerySystem

def demo_openai_integration():
    """Demonstrate MATLAB RAG with OpenAI API integration."""

    print("🤖 MATLAB RAG with OpenAI Code Generation Demo")
    print("=" * 55)

    # Check for API keys (DeepSeek first, then OpenAI)
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if deepseek_key:
        print("✅ DeepSeek API key found! (Recommended for Hong Kong)")
        use_deepseek = True
        use_openai = False
    elif openai_key:
        print("✅ OpenAI API key found!")
        use_deepseek = False
        use_openai = True
    else:
        print("❌ No API keys found (DEEPSEEK_API_KEY or OPENAI_API_KEY)")
        print("\n📝 API Options:")
        print("1. DeepSeek API (works in Hong Kong):")
        print("   • Get key: https://platform.deepseek.com/api_keys")
        print("   • Set: export DEEPSEEK_API_KEY='your-key'")
        print("2. OpenAI API (may not work in Hong Kong):")
        print("   • Get key: https://platform.openai.com/api-keys")
        print("   • Set: export OPENAI_API_KEY='your-key'")
        print("\n💡 Continuing with retrieval-only mode for demo...")
        use_deepseek = False
        use_openai = False

    # Initialize the query system
    try:
        query_system = MATLABQuerySystem()
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return

    # Demo queries
    demo_queries = [
        "Write MATLAB code to create a line plot of sin(x) from 0 to 2π",
        "How do I create a matrix multiplication function in MATLAB?",
        "Show me how to use linspace and plot together",
    ]

    print(f"\n🚀 Testing with {len(demo_queries)} MATLAB coding queries...\n")

    for i, question in enumerate(demo_queries, 1):
        print(f"Query {i}: {question}")
        print("-" * 60)

        try:
            # Get response from the system
            result = query_system.query(question)

            # Display response (truncated for readability)
            response = result['answer']
            if len(response) > 1500:
                response = response[:1500] + "\n\n[... response truncated ...]"

            print(response)
            print(f"\n⏱️  Response time: {result.get('query_time', 'N/A'):.2f}s")
            print("-" * 60 + "\n")

        except Exception as e:
            print(f"❌ Query failed: {e}")
            print("-" * 60 + "\n")

    # Summary
    print("🎯 Demo Complete!")
    print(f"• Database: 171,366 embeddings ready")
    print(f"• Retrieval: Semantic search (BGE)")
    if use_deepseek:
        print(f"• Code Generation: DeepSeek API (fast, works in Hong Kong)")
    elif use_openai:
        print(f"• Code Generation: OpenAI API")
    else:
        print(f"• Code Generation: Not configured")

    print("\n💡 To enable code generation:")
    if not use_deepseek:
        print("   export DEEPSEEK_API_KEY='your-deepseek-key'  # Recommended")
    if not use_openai:
        print("   export OPENAI_API_KEY='your-openai-key'      # Alternative")
    print("   python demo_openai.py")

if __name__ == "__main__":
    demo_openai_integration()
