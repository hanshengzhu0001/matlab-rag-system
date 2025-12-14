#!/usr/bin/env python3
"""
Script to scan and analyze file types in MATLAB documentation.
This helps understand the documentation structure before building the RAG system.
"""

import os
from pathlib import Path
from collections import defaultdict, Counter
import json
from typing import Dict, List, Tuple


def scan_directory(directory: str, max_files: int = None) -> Tuple[Dict[str, int], Dict[str, List[str]], int]:
    """
    Scan directory recursively and collect file type statistics.

    Args:
        directory: Root directory to scan
        max_files: Maximum number of files to scan (for testing)

    Returns:
        Tuple of (extension_counts, sample_files, total_files)
    """
    extension_counts = defaultdict(int)
    sample_files = defaultdict(list)
    total_files = 0

    root_path = Path(directory)

    print(f"Scanning directory: {directory}")
    print("-" * 50)

    for file_path in root_path.rglob('*'):
        if not file_path.is_file():
            continue

        total_files += 1

        if max_files and total_files > max_files:
            print(f"Stopping at {max_files} files (limit reached)")
            break

        # Get file extension (lowercase)
        ext = file_path.suffix.lower()
        if not ext:
            ext = 'no_extension'

        extension_counts[ext] += 1

        # Collect sample files for each extension (max 3 per type)
        if len(sample_files[ext]) < 3:
            sample_files[ext].append(str(file_path.relative_to(root_path)))

        # Progress indicator
        if total_files % 1000 == 0:
            print(f"Scanned {total_files} files...")

    return extension_counts, sample_files, total_files


def analyze_matlab_content(extension_counts: Dict[str, int]) -> Dict[str, any]:
    """
    Analyze the file types specifically relevant for MATLAB RAG system.
    """
    analysis = {
        'primary_content': {},
        'supporting_content': {},
        'media_content': {},
        'other': {},
        'totals': {}
    }

    # Primary content (what we want to process)
    primary_extensions = ['.html', '.htm']
    analysis['primary_content'] = {
        ext: extension_counts.get(ext, 0)
        for ext in primary_extensions
    }

    # Supporting content
    supporting_extensions = ['.json', '.xml', '.map', '.txt', '.properties']
    analysis['supporting_content'] = {
        ext: extension_counts.get(ext, 0)
        for ext in supporting_extensions
    }

    # Media content
    media_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico']
    analysis['supporting_content'].update({
        ext: extension_counts.get(ext, 0)
        for ext in media_extensions
    })

    # Other content
    all_known = set(primary_extensions + supporting_extensions + media_extensions)
    analysis['other'] = {
        ext: count for ext, count in extension_counts.items()
        if ext not in all_known
    }

    # Calculate totals
    analysis['totals'] = {
        'primary_content': sum(analysis['primary_content'].values()),
        'supporting_content': sum(analysis['supporting_content'].values()),
        'other': sum(analysis['other'].values()),
        'grand_total': sum(extension_counts.values())
    }

    return analysis


def print_summary(extension_counts: Dict[str, int], analysis: Dict[str, any], sample_files: Dict[str, List[str]]):
    """Print formatted summary of findings."""

    print("\n" + "="*80)
    print("MATLAB DOCUMENTATION FILE TYPE ANALYSIS")
    print("="*80)

    print("\n📊 OVERALL STATISTICS:")
    print(f"Total files scanned: {analysis['totals']['grand_total']:,}")
    print(f"Primary content files: {analysis['totals']['primary_content']:,}")
    print(f"Supporting files: {analysis['totals']['supporting_content']:,}")
    print(f"Other files: {analysis['totals']['other']:,}")

    print("\n📄 PRIMARY CONTENT (HTML files to process):")
    for ext, count in analysis['primary_content'].items():
        if count > 0:
            print(f"  {ext}: {count:,} files")
            if sample_files.get(ext):
                print(f"    Sample files: {', '.join(sample_files[ext][:2])}")

    print("\n🎯 TOP FILE TYPES (by count):")
    # Sort extensions by count
    sorted_exts = sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)
    for ext, count in sorted_exts[:15]:  # Top 15
        percentage = (count / analysis['totals']['grand_total']) * 100
        print("15s")

    print("\n🗂️  SUPPORTING CONTENT:")
    for ext, count in sorted(extension_counts.items(), key=lambda x: x[1], reverse=True):
        if ext in analysis['supporting_content'] and count > 0:
            print("15s")

    print("\n⚠️  OTHER FILE TYPES:")
    for ext, count in sorted(extension_counts.items(), key=lambda x: x[1], reverse=True):
        if ext in analysis['other'] and count > 0:
            print("15s")

    print("\n💡 RECOMMENDATIONS FOR RAG SYSTEM:")
    html_files = analysis['primary_content'].get('.html', 0)
    if html_files > 0:
        print(f"• Process {html_files:,} HTML files as primary content")
        print("• Focus on core MATLAB documentation in /matlab/ directory first")
        print("• Consider excluding Chinese translations (_zh_CN.html) to reduce scope")
        print("• Test parsing on sample files before full processing")


def save_results(extension_counts: Dict[str, int], analysis: Dict[str, any], sample_files: Dict[str, List[str]], output_file: str = "file_scan_results.json"):
    """Save results to JSON file for later reference."""

    results = {
        'scan_timestamp': str(Path(__file__).parent / 'scan_results'),
        'extension_counts': dict(extension_counts),
        'analysis': analysis,
        'sample_files': dict(sample_files),
        'recommendations': [
            "Focus on HTML files (.html) as primary content",
            "Start with core MATLAB documentation (/matlab/ directory)",
            "Consider excluding Chinese translations to reduce processing time",
            "Test HTML parsing on sample files before full implementation"
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Results saved to: {output_file}")


def main():
    """Main function to run the file type scan."""

    docs_directory = "matlab_documents"

    if not Path(docs_directory).exists():
        print(f"Error: Directory '{docs_directory}' not found!")
        print("Make sure you're running this script from the correct directory.")
        return

    # Scan the directory
    extension_counts, sample_files, total_files = scan_directory(docs_directory)

    # Analyze specifically for MATLAB RAG system
    analysis = analyze_matlab_content(extension_counts)

    # Print summary
    print_summary(extension_counts, analysis, sample_files)

    # Save results
    save_results(extension_counts, analysis, sample_files)

    print("\n✅ File type scan completed!")
    print("Next steps:")
    print("1. Review the results above")
    print("2. Test HTML parsing on sample files")
    print("3. Start building the RAG system with a subset of documents")


if __name__ == "__main__":
    main()
