#!/usr/bin/env python3
"""
Vision RAG Core Module - SAFE MODE

This is a safe version that completely avoids problematic imports
and uses only pure Python implementations.
"""

import os
import re
import math
import logging
from typing import Dict, Any, List
from pathlib import Path

# No external library imports - pure Python only
logger = logging.getLogger(__name__)

class PurePythonFallbacks:
    """Pure Python implementations for all mathematical operations."""

    @staticmethod
    def array(data):
        """Return data as-is for array operations."""
        return data

    @staticmethod
    def linspace(start, stop, num):
        """Pure Python linspace implementation."""
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]

    @staticmethod
    def sin(x):
        """Pure Python sin for scalars and lists."""
        if hasattr(x, '__iter__') and not isinstance(x, str):
            return [math.sin(val) for val in x]
        return math.sin(x)

    @staticmethod
    def cos(x):
        """Pure Python cos for scalars and lists."""
        if hasattr(x, '__iter__') and not isinstance(x, str):
            return [math.cos(val) for val in x]
        return math.cos(x)

    @staticmethod
    def tan(x):
        """Pure Python tan for scalars and lists."""
        if hasattr(x, '__iter__') and not isinstance(x, str):
            return [math.tan(val) for val in x]
        return math.tan(x)

    @staticmethod
    def exp(x):
        """Pure Python exp for scalars and lists."""
        if hasattr(x, '__iter__') and not isinstance(x, str):
            return [math.exp(val) for val in x]
        return math.exp(x)

    @staticmethod
    def log(x):
        """Pure Python log for scalars and lists."""
        if hasattr(x, '__iter__') and not isinstance(x, str):
            return [math.log(max(val, 1e-10)) for val in x]
        return math.log(max(x, 1e-10))

    @staticmethod
    def mean(data):
        """Calculate mean of data."""
        return sum(data) / len(data)

    @staticmethod
    def std(data):
        """Calculate standard deviation of data."""
        mean_val = PurePythonFallbacks.mean(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

    @staticmethod
    def corrcoef(x, y):
        """Simple correlation coefficient."""
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(a * b for a, b in zip(x, y))
        sum_x2 = sum(a * a for a in x)
        sum_y2 = sum(b * b for b in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

        if denominator == 0:
            return 0.0
        corr = numerator / denominator
        return [[1.0, corr], [corr, 1.0]]

    @staticmethod
    def where(condition):
        """Simple where function."""
        if isinstance(condition, list):
            return [i for i, val in enumerate(condition) if val]
        return []

    @staticmethod
    def unique(arr):
        """Simple unique function."""
        seen = set()
        result = []
        for item in arr:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

# Use pure Python fallbacks for all operations
np = PurePythonFallbacks()

class MockImage:
    """Mock image class for fallback operations."""
    def __init__(self, path=None, mode='RGB', size=(100, 100), color=255):
        self.path = path
        self.mode = mode
        self.size = size
        self.color = color

    def convert(self, mode):
        self.mode = mode
        return self

    def getdata(self):
        """Return mock pixel data."""
        width, height = self.size
        return [self.color] * (width * height)

    def __array__(self):
        """Return 2D array representation."""
        width, height = self.size
        return [[self.color for _ in range(width)] for _ in range(height)]

# Mock Image module
class MockImageModule:
    @staticmethod
    def open(path):
        return MockImage(path=path)

    @staticmethod
    def new(mode, size, color=0):
        return MockImage(mode=mode, size=size, color=color)

Image = MockImageModule

logger = logging.getLogger(__name__)

class VisionRAGAnalyzer:
    """
    SAFE MODE: Vision RAG Analyzer with pure Python fallbacks.
    Completely avoids external library dependencies that cause segmentation faults.
    """

    def __init__(self, rag_system_or_chain=None, model_name="mock", device="cpu"):
        """
        Initialize the Vision RAG Analyzer in safe mode.
        """
        # Store RAG system
        if hasattr(rag_system_or_chain, 'query'):
            self.rag_system = rag_system_or_chain
            self.use_system_query = True
        else:
            self.rag_system = None
            self.use_system_query = False

        # Mock BLIP components
        self.processor = None
        self.model = None
        self.device = "cpu"

        # Load figure mappings (if available)
        self.figure_mappings = self._load_figure_mappings()

        logger.info("✅ VisionRAGAnalyzer initialized in SAFE MODE (no external libraries)")

    def _load_figure_mappings(self):
        """Load figure-to-code mappings."""
        try:
            mapping_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'figure_index.json')
            if os.path.exists(mapping_file):
                import json
                with open(mapping_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Figure mappings file not found")
                return {}
        except Exception as e:
            logger.warning(f"Failed to load figure mappings: {e}")
            return {}

    def analyze_image(self, pil_image, user_question: str) -> Dict[str, Any]:
        """
        Analyze an image in safe mode - provides basic functionality without external libraries.
        """
        logger.info(f"🖼️  Analyzing image in SAFE MODE for question: {user_question[:50]}...")

        try:
            # Basic vision analysis (mock)
            visual_description = {
                'function_family': 'mathematical',
                'plot_type': '2d_line',
                'characteristics': ['unknown'],
                'description': 'Analysis in safe mode - external libraries not available'
            }

            # No parameter extraction in safe mode
            extracted_parameters = {}

            # Enhance description
            enhanced_description = self._enhance_description_with_parameters(
                visual_description, extracted_parameters, pil_image
            )

            # RAG query FIRST (to get context for code generation)
            rag_answer = ""
            source_docs = []
            if self.use_system_query and self.rag_system:
                try:
                    function_family = enhanced_description.get('function_family', 'mathematical')
                    rag_query = f"{user_question} Create MATLAB code for a {function_family} plot."

                    # Pass detected functions from image if available (for demo purposes)
                    detected_functions = getattr(pil_image, '_detected_functions', None)
                    rag_result = self.rag_system.query(rag_query, show_context=True, detected_functions=detected_functions)
                    rag_answer = rag_result['answer']
                    source_docs = rag_result.get('context_docs', [])
                except Exception as e:
                    logger.warning(f"RAG query failed: {e}")

            # Retrieve examples (now using RAG context too)
            relevant_examples = self._retrieve_code_examples_by_vision(enhanced_description, user_question)

            # Generate code using BOTH vision analysis AND RAG context
            generated_code = self._generate_code_with_few_shot(
                user_question, enhanced_description, relevant_examples, rag_answer, source_docs
            )

            return {
                "visual_analysis": enhanced_description['description'],
                "function_family": enhanced_description.get('function_family', 'unknown'),
                "generated_code": generated_code,
                "rag_answer": rag_answer,
                "source_docs": source_docs,
                "relevant_examples": relevant_examples,
                "safe_mode": True
            }

        except Exception as e:
            logger.error(f"Analysis failed in safe mode: {e}")
            return {
                "visual_analysis": "Analysis failed",
                "function_family": "unknown",
                "generated_code": "% Error: Analysis failed",
                "rag_answer": "",
                "source_docs": [],
                "relevant_examples": [],
                "safe_mode": True,
                "error": str(e)
            }

    def _enhance_description_with_parameters(self, visual_description, extracted_parameters, pil_image):
        """Enhance description with parameters (safe mode)."""
        return visual_description

    def _retrieve_code_examples_by_vision(self, enhanced_description, user_question):
        """Retrieve code examples based on visual features."""
        # Return some mock examples
        return [
            "plot(x, sin(x));",
            "x = linspace(0, 2*pi, 100); y = cos(x); plot(x, y);",
            "x = 0:0.1:10; y = exp(-x); plot(x, y);"
        ]

    def _generate_code_with_few_shot(self, user_question, enhanced_description, relevant_examples, rag_answer="", source_docs=None):
        """Generate code using few-shot learning with RAG context (safe mode)."""
        if source_docs is None:
            source_docs = []

        function_family = enhanced_description.get('function_family', 'mathematical')

        # Analyze question and RAG context to determine what to plot
        question_lower = user_question.lower()
        rag_lower = rag_answer.lower()

        # Extract detected functions from RAG context and examples
        detected_functions = []
        if 'sin' in rag_lower or any('sin' in str(doc.get('page_content', '')) for doc in source_docs):
            detected_functions.append('sin')
        if 'cos' in rag_lower or any('cos' in str(doc.get('page_content', '')) for doc in source_docs):
            detected_functions.append('cos')
        if 'tan' in rag_lower or any('tan' in str(doc.get('page_content', '')) for doc in source_docs):
            detected_functions.append('tan')

        # Check for composite functions (sin + cos)
        has_composite = ('sin' in detected_functions and 'cos' in detected_functions and
                        ('+' in rag_lower or 'sum' in rag_lower or 'composite' in rag_lower))

        # Generate appropriate code based on detected functions
        if has_composite and len(detected_functions) >= 2:
            # Generate code for sin(x) + cos(x) composite
            return f"""% Generated trigonometric composite plot
x = linspace(0, 4*pi, 200);
y1 = sin(x);
y2 = cos(x);
y3 = sin(x) + cos(x);

figure;
plot(x, y1, 'b-', 'LineWidth', 1.5);
hold on;
plot(x, y2, 'r--', 'LineWidth', 1.5);
plot(x, y3, 'g:', 'LineWidth', 2);
hold off;

xlabel('x');
ylabel('y');
title('Trigonometric Functions: sin(x), cos(x), sin(x)+cos(x)');
legend('sin(x)', 'cos(x)', 'sin(x)+cos(x)');
grid on;"""

        elif len(detected_functions) >= 2:
            # Generate code for multiple trigonometric functions
            plot_commands = []
            legend_items = []

            if 'sin' in detected_functions:
                plot_commands.append("plot(x, sin(x), 'b-', 'LineWidth', 1.5);")
                legend_items.append("'sin(x)'")
            if 'cos' in detected_functions:
                plot_commands.append("plot(x, cos(x), 'r--', 'LineWidth', 1.5);")
                legend_items.append("'cos(x)'")
            if 'tan' in detected_functions:
                plot_commands.append("plot(x, tan(x), 'g:', 'LineWidth', 1.5);")
                legend_items.append("'tan(x)'")

            return f"""% Generated multi-function trigonometric plot
x = linspace(0, 4*pi, 200);
figure;
{chr(10).join(f"plot(x, {func}(x), 'LineWidth', 1.5);" if i == 0 else f"hold on; plot(x, {func}(x), 'LineWidth', 1.5);" for i, func in enumerate(detected_functions))}

xlabel('x');
ylabel('y');
title('Trigonometric Functions');
legend({', '.join([f"'{func}(x)'" for func in detected_functions])});
grid on;"""

        elif 'sin' in detected_functions:
            return f"""% Generated sine wave plot
x = linspace(0, 4*pi, 200);
y = sin(x);
figure;
plot(x, y, 'b-', 'LineWidth', 1.5);
xlabel('x');
ylabel('sin(x)');
title('Sine Function');
grid on;"""

        elif 'cos' in detected_functions:
            return f"""% Generated cosine wave plot
x = linspace(0, 4*pi, 200);
y = cos(x);
figure;
plot(x, y, 'r-', 'LineWidth', 1.5);
xlabel('x');
ylabel('cos(x)');
title('Cosine Function');
grid on;"""

        elif 'tan' in detected_functions:
            return f"""% Generated tangent plot
x = linspace(-pi/2 + 0.1, pi/2 - 0.1, 200);  % Avoid asymptotes
y = tan(x);
figure;
plot(x, y, 'g-', 'LineWidth', 1.5);
xlabel('x');
ylabel('tan(x)');
title('Tangent Function');
grid on;"""

        elif 'trigonometric' in function_family.lower():
            return f"""% Generated trigonometric plot
x = linspace(0, 2*pi, 100);
y = sin(x);
figure;
plot(x, y);
xlabel('x');
ylabel('sin(x)');
title('Trigonometric Function');
grid on;"""

        elif 'polynomial' in function_family.lower():
            return f"""% Generated polynomial plot
x = linspace(-5, 5, 100);
y = x.^2;
figure;
plot(x, y);
xlabel('x');
ylabel('x^2');
title('Polynomial Function');
grid on;"""

        else:
            # Default fallback
            return f"""% Generated mathematical plot
x = linspace(0, 10, 100);
y = x;
figure;
plot(x, y);
xlabel('x');
ylabel('y');
title('Mathematical Function');
grid on;"""

    def _get_structured_vision_description(self, pil_image):
        """Get structured vision description (safe mode)."""
        return {
            'function_family': 'mathematical',
            'plot_type': '2d_line',
            'characteristics': ['unknown'],
            'description': 'Safe mode analysis'
        }

    def _extract_parameters_from_image(self, pil_image):
        """Extract parameters (safe mode - returns empty)."""
        return {}

    def _analyze_curve_characteristics(self, pil_image):
        """Analyze curve characteristics (safe mode)."""
        return "Safe mode curve analysis - limited functionality"

    def _extract_curve_from_plot(self, img_array):
        """Extract curve from plot (safe mode)."""
        # Return mock curve points
        return [(i, 50 + 25 * math.sin(i * 0.1)) for i in range(100)]

    def _multi_scale_function_analysis(self, x_norm, y_norm):
        """Multi-scale analysis (safe mode)."""
        return {
            'sin(x)': 0.8,
            'cos(x)': 0.6,
            'linear': 0.3
        }

    def _calculate_scaled_correlation(self, y_norm, x_norm, func_type, amp_scale, freq_scale):
        """Calculate scaled correlation (safe mode)."""
        return 0.5  # Mock correlation

    def _advanced_curve_segmentation(self, img_array):
        """Advanced curve segmentation (safe mode)."""
        return [self._extract_curve_from_plot(img_array)]

    def _analyze_multi_function_patterns(self, curve_points):
        """Analyze multi-function patterns (safe mode)."""
        return "Safe mode pattern analysis - basic trigonometric detection"
