#!/usr/bin/env python3
"""
Vision RAG Core Module

This module provides real-time image analysis capabilities for the MATLAB RAG system.
It combines BLIP image captioning with the existing text-based RAG pipeline.
"""

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class VisionRAGAnalyzer:
    """
    Real-time image analyzer that combines BLIP-2 (advanced) captioning with RAG retrieval.

    This class handles:
    1. BLIP-2-based image description generation (much more detailed)
    2. Visual Question Answering (VQA) capabilities
    3. Enhanced query formulation
    4. Integration with existing RAG pipeline
    """

    def __init__(self, rag_system_or_chain, model_name="Salesforce/blip2-opt-2.7b", device="auto"):
        """
        Initialize the Vision RAG Analyzer with BLIP-2 (properly using VQA capabilities).

        BLIP-2 is designed for Visual Question Answering and should provide much better
        specific mathematical function recognition than BLIP-base.

        Args:
            rag_system_or_chain: Pre-loaded LangChain RAG chain OR MATLABQuerySystem instance
            model_name: BLIP-2 model to use (default: more powerful 2.7B model)
            device: Device for inference ('auto', 'cpu', 'cuda')
        """
        # Check if it's a MATLABQuerySystem instance or a chain
        if hasattr(rag_system_or_chain, 'query'):
            # It's a MATLABQuerySystem instance
            self.rag_system = rag_system_or_chain
            self.rag_chain = None
            self.use_system_query = True
        else:
            # It's a chain
            self.rag_chain = rag_system_or_chain
            self.rag_system = None
            self.use_system_query = False

        # Determine device
        self.device = self._get_device(device)

        logger.info(f"🔧 Initializing VisionRAGAnalyzer with BLIP-2 on device: {self.device}")

        # Load BLIP-2 model and processor (using proper VQA approach)
        logger.info(f"🤖 Loading BLIP-2 model: {model_name}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)

        # Performance optimizations - BLIP-2 benefits from FP16
        if self.device.type == 'cuda':
            self.model = self.model.half()  # FP16 for speed
            logger.info("⚡ Enabled FP16 for faster inference")

        logger.info("✅ VisionRAGAnalyzer with BLIP-2 ready!")

    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate torch device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("CUDA requested but not available, using CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")

    def analyze_image(self, pil_image: Image.Image, user_question: str) -> Dict[str, Any]:
        """
        Analyze an uploaded image and answer the user's question.

        Args:
            pil_image: PIL Image object from user upload
            user_question: User's question about the image

        Returns:
            Dict containing analysis results
        """
        logger.info(f"🖼️  Analyzing image for question: {user_question[:50]}...")

        try:
            # Step 1: Generate BLIP description
            logger.info("🎨 Generating BLIP image description...")
            image_description = self._generate_blip_caption(pil_image)

            # Step 2: Formulate enriched query
            enriched_query = self._create_enriched_query(user_question, image_description)

            # Step 3: Query the RAG system
            logger.info("🔍 Querying RAG system with enriched context...")

            if self.use_system_query and self.rag_system:
                # Use the MATLABQuerySystem.query method (safer)
                logger.info("🔍 Using MATLABQuerySystem.query method")

                # Try the enriched query first
                query_result = self.rag_system.query(enriched_query, show_context=True)
                result = query_result['answer']
                source_docs = query_result.get('context_docs', [])

                # If the result is too generic (contains "cannot find"), try a simpler query
                if "cannot find" in result.lower() and "syntax" in result.lower():
                    logger.info("🔍 Enriched query failed, trying simpler query")
                    simple_query = f"How to create a plot in MATLAB? Provide example code."
                    simple_result = self.rag_system.query(simple_query, show_context=True)
                    result = simple_result['answer']
                    source_docs = simple_result.get('context_docs', [])
                    logger.info("🔍 Used fallback simple query")

                logger.info(f"🔍 Retrieved {len(source_docs) if source_docs else 0} source documents")
                if source_docs:
                    logger.info(f"🔍 First source doc preview: {source_docs[0].page_content[:200] if hasattr(source_docs[0], 'page_content') else str(source_docs[0])[:200]}...")
                if not source_docs:
                    logger.warning("🔍 No context documents retrieved - sources won't be shown")
            else:
                # Use the chain directly
                logger.info("🔍 Using RAG chain directly")
                result = self.rag_chain.invoke({"query": enriched_query})
                logger.info(f"🔍 RAG result type: {type(result)}")
                source_docs = []

            # The result is now already processed by MATLABQuerySystem.query()
            # It's a formatted string from the RAG system
            final_answer = result

            logger.info(f"🔍 Final answer type: {type(final_answer)}")
            logger.info(f"🔍 Final answer preview: {str(final_answer)[:100]}...")

            # Format sources
            formatted_sources = []
            if source_docs:
                for doc in source_docs[:3]:  # Top 3 sources
                    if hasattr(doc, 'page_content'):
                        # Document object
                        content = doc.page_content
                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                        source = metadata.get('source', 'Unknown')
                        doc_type = metadata.get('type', 'text')
                    elif isinstance(doc, dict):
                        # Dict format
                        content = doc.get('page_content', doc.get('content', ''))
                        source = doc.get('source', 'Unknown')
                        doc_type = doc.get('type', 'text')
                    else:
                        # Fallback
                        content = str(doc)
                        source = 'Unknown'
                        doc_type = 'text'

                    source_info = {
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'source': source,
                        'type': doc_type
                    }
                    formatted_sources.append(source_info)

            analysis_result = {
                "image_description": image_description,
                "enriched_query": enriched_query,
                "final_answer": final_answer,
                "source_documents": formatted_sources,
                "success": True
            }

            logger.info("✅ Image analysis complete!")
            return analysis_result

        except Exception as e:
            logger.error(f"❌ Error in image analysis: {str(e)}")
            return {
                "image_description": "Error generating description",
                "enriched_query": user_question,
                "final_answer": f"Error analyzing image: {str(e)}",
                "source_documents": [],
                "success": False,
                "error": str(e)
            }

    def _generate_blip_caption(self, pil_image: Image.Image, max_length: int = 50) -> str:
        """
        Generate a highly specific BLIP-2 caption using Visual Question Answering (VQA).

        BLIP-2 is designed for VQA and should provide much more specific mathematical
        function recognition than BLIP-base's generic captioning approach.

        Args:
            pil_image: PIL Image to analyze
            max_length: Maximum answer length

        Returns:
            str: Generated specific mathematical function description
        """
        try:
            # Ensure image is in RGB format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            answers = []

            # BLIP-2 VQA Questions - designed for specific mathematical recognition
            vqa_questions = [
                "What mathematical function is shown in this plot?",
                "What type of mathematical relationship or function is plotted?",
                "Is this a trigonometric function? If so, which one?",
                "What kind of curve or function does this plot represent?",
                "Identify the mathematical pattern: polynomial, exponential, trigonometric, or other?"
            ]

            for question in vqa_questions[:3]:  # Try 3 questions for efficiency
                try:
                    # BLIP-2 VQA format: image + question
                    inputs = self.processor(pil_image, text=question, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        if self.device.type == 'cuda':
                            with torch.cuda.amp.autocast():
                                outputs = self.model.generate(**inputs, max_length=max_length, num_beams=3, do_sample=False)
                        else:
                            outputs = self.model.generate(**inputs, max_length=max_length, num_beams=3)

                    answer = self.processor.decode(outputs[0], skip_special_tokens=True)

                    # Clean up the answer - remove the question if it appears
                    answer = answer.replace(question, "").strip()
                    if answer.startswith(("The plot shows", "This is", "It shows")):
                        answer = answer.split("shows", 1)[-1].strip()
                    if answer.startswith(("a ", "an ")):
                        answer = answer[2:].strip()

                    # Only keep meaningful answers
                    if len(answer) > 3 and not answer.lower().startswith(("i don't", "i cannot", "unknown")):
                        answers.append(answer.strip())

                except Exception as e:
                    logger.warning(f"VQA question failed: {e}")
                    continue

            # If we got good answers, find the most specific one
            if answers:
                # Prioritize answers that mention specific functions
                trigonometric_keywords = ['sine', 'cosine', 'sin', 'cos', 'tangent', 'tan', 'trigonometric']
                polynomial_keywords = ['linear', 'quadratic', 'polynomial', 'cubic', 'parabolic']
                exponential_keywords = ['exponential', 'logarithmic', 'log']

                best_answer = None
                best_score = 0

                for answer in answers:
                    score = len(answer.split())  # Prefer longer, more detailed answers

                    # Boost score for specific function mentions
                    answer_lower = answer.lower()
                    if any(word in answer_lower for word in trigonometric_keywords):
                        score += 10
                    elif any(word in answer_lower for word in polynomial_keywords + exponential_keywords):
                        score += 5

                    if score > best_score:
                        best_score = score
                        best_answer = answer

                if best_answer:
                    # Final cleanup and enhancement
                    result = best_answer.strip()

                    # Convert generic terms to specific mathematical descriptions
                    if "trigonometric" in result.lower() and "sine" not in result.lower() and "cosine" not in result.lower():
                        result = "trigonometric functions (sine/cosine combination)"

                    return result

            # Fallback to basic captioning if VQA fails
            logger.warning("VQA failed, falling back to basic captioning")
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(**inputs, max_length=max_length, num_beams=5)
                else:
                    outputs = self.model.generate(**inputs, max_length=max_length, num_beams=5)

            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return f"mathematical visualization: {caption.strip()}"

        except Exception as e:
            logger.error(f"BLIP-2 caption generation failed: {e}")
            return "mathematical visualization"

    def _create_enriched_query(self, user_question: str, image_description: str) -> str:
        """
        Create an enriched query that combines the user's question with image context.

        Args:
            user_question: Original user question
            image_description: BLIP-generated description

        Returns:
            str: Enriched query for RAG system
        """
        # Comprehensive keyword extraction for all plot types and functions
        detected_elements = []
        desc_lower = image_description.lower()

        # Mathematical functions
        math_functions = {
            'sine': ['sin', 'sine'],
            'cosine': ['cos', 'cosine'],
            'tangent': ['tan', 'tangent'],
            'exponential': ['exp', 'exponential'],
            'logarithmic': ['log', 'ln', 'logarithmic'],
            'linear': ['linear', 'straight line'],
            'quadratic': ['quadratic', 'parabolic'],
            'polynomial': ['polynomial', 'curve'],
            'power': ['power', 'x^2', 'x^3']
        }

        # Plot types
        plot_types = {
            'scatter': ['scatter', 'points', 'dots'],
            'bar': ['bar', 'bars', 'bar chart'],
            'histogram': ['histogram', 'distribution', 'frequency'],
            'surface': ['surface', '3d', 'mesh'],
            'contour': ['contour', 'level curves'],
            'heatmap': ['heatmap', 'heat map', 'color map'],
            'boxplot': ['box plot', 'boxplot', 'whisker'],
            'line': ['line', 'plot', 'curve', 'graph']
        }

        # Statistical indicators
        if any(word in desc_lower for word in ['distribution', 'histogram', 'boxplot', 'violin', 'error bars', 'confidence', 'statistics', 'statistical']):
            detected_elements.append('statistical visualization')

        # Detect functions
        for func_name, keywords in math_functions.items():
            if any(keyword in desc_lower for keyword in keywords):
                detected_elements.append(func_name)

        # Detect plot types
        for plot_name, keywords in plot_types.items():
            if any(keyword in desc_lower for keyword in keywords):
                detected_elements.append(f"{plot_name} plot")

        # Remove duplicates and create enriched query
        detected_elements = list(set(detected_elements))

        if detected_elements:
            # Create a descriptive summary
            if len(detected_elements) == 1:
                element_desc = detected_elements[0]
            elif len(detected_elements) == 2:
                element_desc = f"{detected_elements[0]} and {detected_elements[1]}"
            else:
                element_desc = f"{', '.join(detected_elements[:-1])}, and {detected_elements[-1]}"

            enriched_query = f"""
Based on an uploaded image showing a {element_desc}: {image_description}

{user_question}

Please provide MATLAB code that recreates this visualization. Focus on {element_desc} and show the appropriate MATLAB plotting functions and syntax.
"""
        else:
            # Generic fallback
            enriched_query = f"""
Based on an uploaded image showing: {image_description}

{user_question}

Please provide MATLAB code and explanation for creating this type of data visualization.
"""

        return enriched_query.strip()

def create_vision_rag_demo():
    """
    Create a simple demo function for testing the VisionRAGAnalyzer.
    This can be used for testing without the full Gradio interface.
    """
    from query_rag import MATLABQuerySystem
    import gradio as gr

    # Initialize the RAG system
    print("🔧 Initializing MATLAB RAG system...")
    query_system = MATLABQuerySystem()
    analyzer = VisionRAGAnalyzer(query_system.chain)

    def demo_analyze(image, question):
        if image is None:
            return "Please upload an image.", ""

        result = analyzer.analyze_image(image, question)

        answer = f"**Image Description:** {result['image_description']}\n\n"
        answer += f"**Analysis:** {result['final_answer']}\n\n"

        if result['source_documents']:
            answer += "**Sources:**\n"
            for i, doc in enumerate(result['source_documents'], 1):
                answer += f"{i}. {doc['source']} ({doc['type']})\n"

        return answer, result['image_description']

    # Create simple Gradio interface
    with gr.Blocks(title="Vision RAG Demo") as demo:
        gr.Markdown("# MATLAB Vision RAG Demo")
        gr.Markdown("Upload an image and ask a question!")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
            question_input = gr.Textbox(label="Your Question", placeholder="What does this plot show?")

        output = gr.Markdown(label="Analysis")

        btn = gr.Button("Analyze", variant="primary")
        btn.click(demo_analyze, inputs=[image_input, question_input], outputs=output)

    return demo

if __name__ == "__main__":
    # Simple test
    print("🧪 Testing VisionRAGAnalyzer...")

    # This would normally load the full RAG system
    # For testing, we'll just show that the module loads
    print("✅ Vision RAG core module loaded successfully!")
    print("💡 To use:")
    print("   1. Initialize MATLABQuerySystem")
    print("   2. analyzer = VisionRAGAnalyzer(rag_system.chain)")
    print("   3. result = analyzer.analyze_image(pil_image, question)")
