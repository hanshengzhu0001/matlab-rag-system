#!/usr/bin/env python3
"""
MATLAB Vision RAG Web Interface

Gradio-based web application for real-time image analysis with MATLAB RAG.
Users can upload images and ask questions about MATLAB plots, diagrams, and UI elements.
"""

import gradio as gr
import logging
from pathlib import Path

# Import our custom modules
from vision_rag_core import VisionRAGAnalyzer
from query_rag import MATLABQuerySystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the analyzer
analyzer = None

def initialize_analyzer():
    """Initialize the VisionRAGAnalyzer with the MATLAB RAG system."""
    global analyzer

    if analyzer is None:
        try:
            logger.info("🔧 Initializing MATLAB RAG system...")
            query_system = MATLABQuerySystem()
            analyzer = VisionRAGAnalyzer(query_system)  # Pass the full system, not just chain
            logger.info("✅ Vision RAG system ready!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            raise

    return analyzer

def gradio_interface(image, question):
    """
    Main Gradio interface function.

    Args:
        image: PIL Image uploaded by user
        question: User's question about the image

    Returns:
        tuple: (analysis_markdown, image_description)
    """
    if image is None:
        return "❌ **Error:** Please upload an image first.", ""

    if not question or question.strip() == "":
        return "❌ **Error:** Please enter a question about the image.", ""

    try:
        # Initialize analyzer if needed
        vision_analyzer = initialize_analyzer()

        # Analyze the image
        logger.info(f"🖼️  Processing image analysis request: {question[:50]}...")
        result = vision_analyzer.analyze_image(image, question.strip())

        if not result['success']:
            return f"❌ **Analysis Failed:** {result.get('error', 'Unknown error')}", ""

        # Format the response
        analysis_md = "## 🧠 MATLAB Analysis\n\n"

        # Add visual analysis summary
        if 'visual_analysis' in result:
            visual = result['visual_analysis']
            analysis_md += f"### 📊 Visual Analysis\n"
            analysis_md += f"- **Function Family:** {visual.get('function_family', 'unknown')}\n"
            analysis_md += f"- **Plot Type:** {visual.get('plot_type', 'unknown')}\n"
            analysis_md += f"- **Characteristics:** {', '.join(visual.get('characteristics', []))}\n\n"

        # Add generated code
        if 'generated_code' in result and result['generated_code']:
            analysis_md += f"### 💻 Generated MATLAB Code\n```matlab\n{result['generated_code']}\n```\n\n"

        # Add retrieved examples
        if 'retrieved_examples' in result and result['retrieved_examples']:
            analysis_md += f"### 📋 Similar Examples Used\n"
            for i, example in enumerate(result['retrieved_examples'][:2], 1):
                analysis_md += f"**Example {i}:**\n```matlab\n{example[:200]}{'...' if len(example) > 200 else ''}\n```\n\n"

        # Add RAG context
        if 'rag_context' in result and result['rag_context']:
            analysis_md += f"### 📚 Documentation Context\n{result['rag_context'][:500]}{'...' if len(result['rag_context']) > 500 else ''}\n\n"

        # Add sources if available
        if result.get('sources') and len(result['sources']) > 0:
            analysis_md += "### 📚 Sources Consulted\n"
            for i, doc in enumerate(result['sources'], 1):
                source_type = "📄" if doc.get('type') == 'text' else "🖼️"
                analysis_md += f"{i}. {source_type} **{doc.get('source', 'Unknown')}**\n"
                analysis_md += f"   *{doc.get('content', 'No content available')}*\n\n"

        # Create image description summary
        image_desc = f"Function: {result.get('visual_analysis', {}).get('function_family', 'unknown')} | "
        image_desc += f"Type: {result.get('visual_analysis', {}).get('plot_type', 'unknown')} | "
        image_desc += f"Characteristics: {', '.join(result.get('visual_analysis', {}).get('characteristics', []))}"

        return analysis_md, image_desc

    except Exception as e:
        logger.error(f"❌ Interface error: {str(e)}")
        return f"❌ **System Error:** {str(e)}\n\nPlease try again or check the server logs.", ""

def create_interface():
    """Create and configure the Gradio interface."""

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    .gradio-container .main {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 20px;
        padding: 30px;
    }

    .gradio-container .title {
        color: #2d3748;
        text-align: center;
        margin-bottom: 30px;
        font-size: 2.5em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .gradio-container .description {
        color: #4a5568;
        text-align: center;
        margin-bottom: 40px;
        font-size: 1.2em;
    }

    .example-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }

    .example-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    """

    # Create the Gradio Blocks interface
    with gr.Blocks(
        title="🖥️ MATLAB Vision RAG Assistant",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:

        gr.HTML("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #2d3748; font-size: 3em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
                🖥️ MATLAB Vision RAG Assistant
            </h1>
            <p style="color: #4a5568; font-size: 1.3em; margin: 0;">
                Upload images of MATLAB plots, diagrams, or UI elements and ask questions!
            </p>
        </div>
        """)

        gr.Markdown("""
        ### 🎯 How It Works
        1. **Upload** a MATLAB-related image (plot, diagram, screenshot)
        2. **Ask** a specific question about what you see
        3. **Get** detailed analysis powered by MATLAB documentation and AI vision

        ### 💡 Example Questions
        - "What function creates this type of plot?"
        - "How do I modify this code to change the colors?"
        - "What does this error message mean?"
        - "How can I recreate this visualization?"
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                image_input = gr.Image(
                    type="pil",
                    label="📤 Upload MATLAB Image",
                    height=400
                )

                question_input = gr.Textbox(
                    label="❓ Your Question",
                    placeholder="e.g., What function generates this 3D surface plot?",
                    lines=3
                )

                analyze_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                # Output section
                analysis_output = gr.Markdown(
                    label="🤖 Analysis & Answer",
                    value="Upload an image and ask a question to get started!",
                    height=500
                )

                description_output = gr.Textbox(
                    label="📝 Image Description",
                    interactive=False,
                    lines=3
                )

        # Connect the interface
        analyze_btn.click(
            fn=gradio_interface,
            inputs=[image_input, question_input],
            outputs=[analysis_output, description_output],
            api_name="analyze_image"
        )

        # Example gallery
        gr.Examples(
            examples=[
                ["What plotting function creates this visualization?", "example_plot.png"],
                ["How do I modify the colors in this MATLAB plot?", "color_plot.png"],
                ["What does this MATLAB error message mean?", "error_screenshot.png"],
                ["How can I recreate this 3D surface plot?", "surface_plot.png"],
            ],
            inputs=[question_input, image_input],
            label="🚀 Quick Examples - Click to try!",
            examples_per_page=4
        )

        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
            <p>Powered by MATLAB Documentation RAG • BLIP Vision Model • LangChain</p>
            <p>Database: 171,366+ text chunks + 9,000+ visual knowledge items</p>
        </div>
        """)

    return demo

def main():
    """Main function to run the Vision RAG web interface."""

    print("🚀 Starting MATLAB Vision RAG Assistant...")
    print("=" * 60)

    try:
        # Test initialization
        print("🔧 Testing system initialization...")
        initialize_analyzer()
        print("✅ System ready!")

        # Create and launch interface
        demo = create_interface()

        print("\n🌐 Launching web interface...")
        print("📱 Interface will be available at: http://localhost:7860 (or next available port)")
        print("🔗 Share link will be generated for external access")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60)

        # Launch with optimal settings - let Gradio find available port
        demo.launch(
            server_name="0.0.0.0",
            server_port=None,  # Let Gradio find available port automatically
            share=True,  # Creates public link
            show_error=True,
            max_threads=4  # Limit threads for stability
        )

    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Failed to start: {str(e)}")
        print("💡 Check that:")
        print("   1. MATLAB RAG database exists (run build_rag.py)")
        print("   2. Ollama is running (if using Qwen)")
        print("   3. All dependencies are installed")
        raise

if __name__ == "__main__":
    main()
