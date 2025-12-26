#!/usr/bin/env python3
"""
BLIP Image Captioning Script for MATLAB Documentation

This script processes OCR data from extract_ocr.m and generates rich descriptions
using Salesforce's BLIP image captioning model.
"""

import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BLIPImageDescriber:
    """Handles BLIP-based image description generation."""

    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device="auto"):
        """
        Initialize the BLIP model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = self._get_device(device)

        logger.info(f"Loading BLIP model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load model and processor
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

        # Move to device
        self.model = self.model.to(self.device)

        # Enable optimizations for better performance
        if self.device.type == 'cuda':
            self.model = self.model.half()  # Use FP16 for faster inference

        logger.info("✅ BLIP model loaded successfully")

    def _get_device(self, device):
        """Determine the appropriate device to use."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def generate_caption(self, image_path, max_length=75):
        """
        Generate a caption for a single image.

        Args:
            image_path: Path to the image file
            max_length: Maximum caption length

        Returns:
            str: Generated caption
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')

            # Prepare inputs
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            # Generate caption
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():  # Mixed precision for speed
                        outputs = self.model.generate(**inputs, max_length=max_length)
                else:
                    outputs = self.model.generate(**inputs, max_length=max_length)

            # Decode caption
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)

            return caption.strip()

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return f"Error generating caption: {str(e)}"

def main():
    """Main function to process OCR data and generate BLIP descriptions."""

    # Configuration
    ocr_input_file = "image_ocr_data.json"
    output_file = "visual_knowledge.json"
    docs_root = Path("./matlab_documents")

    # Check if OCR data exists
    if not Path(ocr_input_file).exists():
        logger.error(f"❌ OCR input file not found: {ocr_input_file}")
        logger.info("💡 Run extract_ocr.m first to generate OCR data")
        return

    # Load OCR data
    logger.info(f"📂 Loading OCR data from {ocr_input_file}")
    with open(ocr_input_file, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)

    logger.info(f"📊 Found {len(ocr_data)} images to process")

    # Initialize BLIP describer
    describer = BLIPImageDescriber()

    # Process images and add BLIP descriptions
    processed_count = 0
    error_count = 0

    for item in tqdm(ocr_data, desc="Generating BLIP descriptions"):
        try:
            # Construct full path
            image_path = docs_root / item['path']

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                item['blip_caption'] = "Image file not found"
                error_count += 1
                continue

            # Generate BLIP caption
            caption = describer.generate_caption(str(image_path))

            # Combine OCR text with BLIP caption
            ocr_text = item.get('ocr_text', '')
            combined_text = f"{ocr_text} | Visual Description: {caption}".strip()

            # Update item
            item['blip_caption'] = caption
            item['combined_text'] = combined_text

            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
            item['blip_caption'] = f"Error: {str(e)}"
            item['combined_text'] = item.get('ocr_text', '')
            error_count += 1

    # Save enhanced data
    logger.info(f"💾 Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ocr_data, f, indent=2, ensure_ascii=False)

    # Summary
    logger.info("✅ BLIP description generation complete!")
    logger.info(f"   📊 Processed: {processed_count} images")
    logger.info(f"   ⚠️  Errors: {error_count} images")
    logger.info(f"   💾 Output saved to: {output_file}")

    # Sample output
    if ocr_data:
        sample = ocr_data[0]
        logger.info("\n📝 Sample result:")
        logger.info(f"   Image: {sample['path']}")
        logger.info(f"   OCR Text: {sample.get('ocr_text', '')[:100]}...")
        logger.info(f"   BLIP Caption: {sample.get('blip_caption', '')}")
        logger.info(f"   Combined Text: {sample.get('combined_text', '')[:200]}...")

if __name__ == "__main__":
    main()
