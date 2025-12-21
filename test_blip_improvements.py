#!/usr/bin/env python3
"""
Test script to verify BLIP-2 improvements for mathematical function recognition.
"""

from PIL import Image
from vision_rag_core import VisionRAGAnalyzer
from query_rag import MATLABQuerySystem

def test_blip_improvements():
    """Test the improved BLIP-2 captioning with mathematical functions."""

    print("🧪 Testing BLIP-2 Mathematical Function Recognition")
    print("=" * 60)

    # Create a test system
    print("🔧 Initializing system...")
    try:
        query_system = MATLABQuerySystem()
        analyzer = VisionRAGAnalyzer(query_system)
        print("✅ System initialized")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return

    # Test with a simple colored square (should recognize it as a basic shape)
    print("\n📷 Testing with synthetic image...")
    test_image = Image.new('RGB', (200, 200), color='blue')

    try:
        result = analyzer.analyze_image(test_image, "What does this show?")
        description = result['image_description']

        print("📝 Generated description:")
        print(f"'{description}'")

        # Check for repetition
        words = description.lower().split()
        word_counts = {}
        for word in words:
            word = word.strip('.,')
            if len(word) > 2:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1

        repeated_words = [word for word, count in word_counts.items() if count > 3]
        if repeated_words:
            print(f"⚠️  Found repeated words: {repeated_words}")
        else:
            print("✅ No excessive repetition detected")

        # Check length
        if len(description) > 500:
            print(f"⚠️  Description too long ({len(description)} chars)")
        elif len(description) < 20:
            print(f"⚠️  Description too short ({len(description)} chars)")
        else:
            print(f"✅ Description length appropriate ({len(description)} chars)")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_blip_improvements()
