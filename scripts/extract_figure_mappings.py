#!/usr/bin/env python3
"""
Extract Figure Mappings from MATLAB HTML Documentation

This script extracts figure-to-caption mappings from MATLAB HTML documentation,
creating a comprehensive mapping between images, their generating code, and context.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FigureMappingExtractor:
    """Extract figure mappings from MATLAB HTML documentation."""

    def __init__(self, docs_root: str = "./matlab_documents"):
        self.docs_root = Path(docs_root)
        self.output_file = "figure_mappings.json"

    def extract_all_mappings(self) -> List[Dict[str, Any]]:
        """Extract figure mappings from all HTML files."""
        logger.info(f"🔍 Scanning for HTML files in {self.docs_root}")

        html_files = []
        for html_file in self.docs_root.rglob("*.html"):
            if html_file.stat().st_size > 1024:  # Skip very small files
                html_files.append(html_file)

        logger.info(f"📊 Found {len(html_files)} HTML files to process")

        all_mappings = []
        batch_size = 20

        for i in range(0, len(html_files), batch_size):
            batch = html_files[i:i + batch_size]
            logger.info(f"🔄 Processing batch {i//batch_size + 1} of {(len(html_files) + batch_size - 1)//batch_size}")

            batch_mappings = []
            for html_file in batch:
                try:
                    mappings = self.extract_mappings_from_html(html_file)
                    batch_mappings.extend(mappings)
                except Exception as e:
                    logger.warning(f"Error processing {html_file}: {e}")

            all_mappings.extend(batch_mappings)

            # Save intermediate results
            if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(html_files):
                self.save_mappings(all_mappings)
                logger.info(f"💾 Saved intermediate results ({len(all_mappings)} mappings)")

        return all_mappings

    def extract_mappings_from_html(self, html_file: Path) -> List[Dict[str, Any]]:
        """Extract figure mappings from a single HTML file."""
        try:
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {html_file}: {e}")
            return []

        mappings = []

        # Find all image tags with context
        # Pattern matches: <img src="..." alt="..." ...>
        img_pattern = r'<img[^>]*src="([^"]*\.(?:png|jpg|jpeg|gif))"[^>]*alt="([^"]*)"[^>]*>'
        img_matches = re.findall(img_pattern, content, re.IGNORECASE)

        for img_src, img_alt in img_matches:
            # Find the position of this image in the HTML
            img_tag = f'<img src="{img_src}" alt="{img_alt}"'
            img_pos = content.find(img_tag)

            if img_pos == -1:
                continue

            # Extract context before image (look for code and section info)
            context_start = max(0, img_pos - 2000)
            context_before = content[context_start:img_pos]

            # Extract context after image
            context_end = min(len(content), img_pos + 500)
            context_after = content[img_pos:context_end]

            # Find MATLAB code in context
            matlab_code = self.extract_matlab_code(context_before + " " + context_after)

            # Find section title
            section_title = self.extract_section_title(context_before)

            # Create mapping entry
            mapping = {
                'html_path': str(html_file.relative_to(self.docs_root)),
                'image_src': img_src,
                'image_alt': img_alt,
                'matlab_code': matlab_code,
                'section_title': section_title,
                'context_before': context_before[-500:] if len(context_before) > 500 else context_before,
                'context_after': context_after[:500] if len(context_after) > 500 else context_after
            }

            mappings.append(mapping)

        return mappings

    def extract_matlab_code(self, context: str) -> str:
        """Extract MATLAB code from HTML context."""
        # Look for <pre> tags containing code
        pre_pattern = r'<pre[^>]*>(.*?)</pre>'
        pre_matches = re.findall(pre_pattern, context, re.DOTALL | re.IGNORECASE)

        for match in reversed(pre_matches):  # Check most recent code first
            code = re.sub(r'<[^>]+>', '', match)  # Remove HTML tags
            code = code.strip()
            if len(code) > 10 and not code.startswith('%'):  # Skip comments
                return code

        return ""

    def extract_section_title(self, context: str) -> str:
        """Extract section title from HTML context."""
        # Look for heading tags
        heading_pattern = r'<h[1-6][^>]*>(.*?)</h[1-6]>'
        heading_matches = re.findall(heading_pattern, context, re.DOTALL | re.IGNORECASE)

        for match in reversed(heading_matches):  # Check most recent heading
            title = re.sub(r'<[^>]+>', '', match).strip()
            if title and len(title) > 3:
                return title

        return ""

    def save_mappings(self, mappings: List[Dict[str, Any]]):
        """Save figure mappings to JSON file."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save mappings: {e}")

    def create_image_to_mapping_index(self, mappings: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create an index mapping image paths to their mappings for quick lookup."""
        index = {}

        for mapping in mappings:
            # Create various key variations for the image
            img_src = mapping['image_src']

            # Remove path components and create different keys
            img_basename = Path(img_src).name

            # Store mapping under different possible keys
            for key in [img_src, img_basename, f"./{img_src}"]:
                index[key] = mapping

        return index

def main():
    """Main function."""
    extractor = FigureMappingExtractor()

    print("🚀 MATLAB Figure Mapping Extractor")
    print("=" * 50)

    try:
        mappings = extractor.extract_all_mappings()

        # Create and save index for quick lookup
        index = extractor.create_image_to_mapping_index(mappings)

        with open('figure_index.json', 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        print("\n✅ Extraction complete!")
        print(f"📊 Total mappings: {len(mappings)}")
        print(f"📁 Files saved: {extractor.output_file}, figure_index.json")

        # Show sample mapping
        if mappings:
            sample = mappings[0]
            print("\n📋 Sample mapping:")
            print(f"  HTML: {sample['html_path']}")
            print(f"  Image: {sample['image_src']}")
            print(f"  Code: {sample['matlab_code'][:50]}..." if sample['matlab_code'] else "  Code: (none)")
            print(f"  Section: {sample['section_title']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
