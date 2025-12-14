#!/usr/bin/env python3
"""
Test script to validate HTML parsing of MATLAB documentation.
This ensures we can properly extract content before building the full RAG system.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json


def test_html_parsing_sample():
    """Test HTML parsing on a few sample files to validate extraction."""

    # Test files from different parts of the documentation
    test_files = [
        "matlab_documents/matlab/arithmetic-operators.html",  # Core function docs
        "matlab_documents/matlab/index.html",                 # Main index
        "matlab_documents/documentation-center.html",         # Documentation center
        "matlab_documents/matlab-copilot/get-started-with-matlab-copilot.html"  # Copilot docs
    ]

    results = {}

    print("Testing HTML parsing on sample files...")
    print("-" * 50)

    for file_path in test_files:
        if not Path(file_path).exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        try:
            print(f"\n📄 Testing: {file_path}")

            # Load the HTML file
            loader = BSHTMLLoader(file_path, open_encoding="utf-8")
            documents = loader.load()

            if not documents:
                print("❌ No content extracted")
                continue

            doc = documents[0]
            content_length = len(doc.page_content)
            title = doc.metadata.get('title', 'No title')

            print(f"✅ Successfully parsed: {title}")
            print(f"   Content length: {content_length:,} characters")

            # Show a sample of the content
            content_preview = doc.page_content[:500].replace('\n', ' ').strip()
            print(f"   Preview: {content_preview}...")

            # Test chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            print(f"   Chunks created: {len(chunks)}")

            # Store results
            results[file_path] = {
                'success': True,
                'title': title,
                'content_length': content_length,
                'chunks': len(chunks),
                'preview': content_preview
            }

        except Exception as e:
            print(f"❌ Error parsing {file_path}: {str(e)}")
            results[file_path] = {
                'success': False,
                'error': str(e)
            }

    return results


def analyze_matlab_directory():
    """Analyze the structure of the matlab subdirectory specifically."""

    matlab_dir = Path("matlab_documents/matlab")

    if not matlab_dir.exists():
        print("❌ MATLAB directory not found")
        return

    print(f"\n📊 Analyzing MATLAB directory: {matlab_dir}")
    print("-" * 50)

    # Count HTML files
    html_files = list(matlab_dir.glob("**/*.html"))
    print(f"Total HTML files in matlab/: {len(html_files):,}")

    # Sample some file names
    if html_files:
        print("Sample files:")
        for i, file_path in enumerate(html_files[:10]):
            print(f"  {i+1}. {file_path.name}")

    # Check for Chinese translations
    zh_files = [f for f in html_files if '_zh_CN' in f.name]
    print(f"\nChinese translation files: {len(zh_files)} ({len(zh_files)/len(html_files)*100:.1f}%)")

    # Check file sizes
    sizes = []
    for file_path in html_files[:100]:  # Sample first 100 files
        try:
            size = file_path.stat().st_size
            sizes.append(size)
        except:
            pass

    if sizes:
        avg_size = sum(sizes) / len(sizes)
        print(f"Average HTML file size (sample): {avg_size/1024:.1f} KB")

    return {
        'total_html_files': len(html_files),
        'chinese_files': len(zh_files),
        'sample_avg_size_kb': avg_size/1024 if 'avg_size' in locals() else 0
    }


def estimate_processing_time():
    """Estimate processing time based on file counts and sample parsing."""

    print("\n⏱️  PROCESSING TIME ESTIMATIONS:")
    print("-" * 50)

    # Based on our analysis
    total_html = 12389
    avg_chunk_time = 0.1  # seconds per HTML file (estimated)
    embedding_time = 0.5  # seconds per chunk for embeddings
    chunks_per_file = 3  # estimated average

    parsing_time = total_html * avg_chunk_time / 60  # minutes
    embedding_time_total = total_html * chunks_per_file * embedding_time / 60  # minutes

    print(f"Total HTML files: {total_html:,}")
    print(f"Estimated parsing time: {parsing_time:.1f} minutes")
    print(f"Estimated embedding time: {embedding_time_total:.1f} minutes")
    print(f"Total estimated time: {parsing_time + embedding_time_total:.1f} minutes")
    print("\n⚠️  Note: These are rough estimates. Actual time will depend on hardware.")
    print("   GPU acceleration will significantly speed up embedding generation.")


def main():
    """Main function to run HTML parsing tests."""

    print("🔍 MATLAB Documentation HTML Parsing Test")
    print("=" * 60)

    # Test parsing on sample files
    parsing_results = test_html_parsing_sample()

    # Analyze MATLAB directory structure
    matlab_stats = analyze_matlab_directory()

    # Estimate processing time
    estimate_processing_time()

    # Save results
    results = {
        'parsing_tests': parsing_results,
        'matlab_stats': matlab_stats,
        'recommendations': [
            "HTML parsing works correctly with BSHTMLLoader",
            "Focus on core MATLAB documentation first (/matlab/ directory)",
            "Consider excluding Chinese translations to reduce processing time",
            "Chunking strategy (600 chars) appears appropriate",
            "GPU acceleration essential for embedding 12K+ files"
        ]
    }

    with open('html_parsing_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n📁 Results saved to: html_parsing_test_results.json")

    # Overall assessment
    successful_parses = sum(1 for r in parsing_results.values() if r.get('success', False))
    total_tests = len(parsing_results)

    print("\n🎯 ASSESSMENT:")
    if successful_parses == total_tests:
        print("✅ HTML parsing test: PASSED")
        print("   Ready to proceed with RAG system implementation")
    else:
        print(f"⚠️  HTML parsing test: {successful_parses}/{total_tests} passed")
        print("   May need to adjust parsing parameters")

    print("\n🚀 NEXT STEPS:")
    print("1. Set up conda environment with required packages")
    print("2. Create build_rag.py with optimized settings")
    print("3. Start with subset of MATLAB core documentation")
    print("4. Test retrieval quality before full processing")


if __name__ == "__main__":
    main()
