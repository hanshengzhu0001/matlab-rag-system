#!/usr/bin/env python3
"""
Simple HTML parsing test without langchain dependencies.
Tests basic file reading and content extraction.
"""

import os
from pathlib import Path
from bs4 import BeautifulSoup


def test_basic_html_reading():
    """Test basic HTML file reading and parsing."""

    test_files = [
        "matlab_documents/matlab/arithmetic-operators.html",
        "matlab_documents/matlab/index.html",
        "matlab_documents/documentation-center.html"
    ]

    results = {}

    print("🔍 Basic HTML Reading Test")
    print("-" * 40)

    for file_path in test_files:
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            continue

        try:
            print(f"\n📄 Testing: {Path(file_path).name}")

            # Read the HTML file
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            content_length = len(html_content)
            print(f"   File size: {content_length:,} characters")

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "No title"

            # Extract main content (try different selectors)
            content_selectors = [
                'body',
                '.content',
                '#content',
                'main',
                '[role="main"]'
            ]

            main_content = None
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    main_content = element.get_text().strip()
                    break

            if not main_content:
                # Fallback: get all text
                main_content = soup.get_text().strip()

            content_lines = len([line for line in main_content.split('\n') if line.strip()])
            print(f"   Title: {title}")
            print(f"   Content lines: {content_lines}")

            # Show preview
            preview = main_content[:300].replace('\n', ' ').strip()
            print(f"   Preview: {preview}...")

            results[file_path] = {
                'success': True,
                'title': title,
                'file_size': content_length,
                'content_lines': content_lines,
                'has_content': bool(main_content.strip())
            }

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results[file_path] = {
                'success': False,
                'error': str(e)
            }

    return results


def analyze_matlab_structure():
    """Analyze the MATLAB documentation structure."""

    matlab_dir = Path("matlab_documents/matlab")

    print("\n📊 MATLAB Directory Analysis")
    print("-" * 40)

    if not matlab_dir.exists():
        print("❌ MATLAB directory not found")
        return

    # Count files
    html_files = list(matlab_dir.glob("**/*.html"))
    print(f"Total HTML files: {len(html_files):,}")

    # Check for patterns
    function_files = [f for f in html_files if 'ref/' in str(f)]
    guide_files = [f for f in html_files if not 'ref/' in str(f)]

    print(f"Function reference files: {len(function_files):,}")
    print(f"Guide/tutorial files: {len(guide_files):,}")

    # Sample file sizes
    sizes = []
    for file_path in html_files[:50]:  # Sample first 50
        try:
            size = file_path.stat().st_size
            sizes.append(size)
        except:
            pass

    if sizes:
        avg_size = sum(sizes) / len(sizes)
        print(f"Average file size (sample): {avg_size/1024:.1f} KB")
        print(f"Largest file: {max(sizes)/1024:.1f} KB")
        print(f"Smallest file: {min(sizes)/1024:.1f} KB")

    return {
        'total_html': len(html_files),
        'function_files': len(function_files),
        'guide_files': len(guide_files),
        'avg_size_kb': avg_size/1024 if sizes else 0
    }


def main():
    """Main test function."""

    # Test basic HTML reading
    results = test_basic_html_reading()

    # Analyze structure
    structure = analyze_matlab_structure()

    # Summary
    successful_tests = sum(1 for r in results.values() if r.get('success', False))
    total_tests = len(results)

    print("\n🎯 RESULTS SUMMARY:")
    print(f"HTML parsing tests: {successful_tests}/{total_tests} successful")

    if successful_tests == total_tests:
        print("✅ Basic HTML parsing works correctly")
        print("   Ready for langchain integration")
    else:
        print("⚠️  Some parsing issues detected")

    print("\n🚀 NEXT STEPS:")
    print("1. Set up conda environment with langchain packages")
    print("2. Test full langchain BSHTMLLoader")
    print("3. Create build_rag.py with optimized settings")
    print("4. Start processing MATLAB documentation")


if __name__ == "__main__":
    main()
