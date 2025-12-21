#!/usr/bin/env python3
"""
Demo of general-purpose plot and function recognition.
"""

def demo_general_recognition():
    """Show how the system now recognizes various plot types and functions."""

    print("🎯 MATLAB Vision RAG - General-Purpose Recognition Demo")
    print("=" * 60)
    print()

    examples = [
        {
            "plot_type": "Trigonometric Combination",
            "description": "A MATLAB plot showing a combination of sine and cosine functions. Mathematical functions: combination of sine and cosine functions.",
            "code": """% Trigonometric combination
t = 0:0.01:4*pi;
y = sin(t) + 0.5*cos(2*t);
plot(t, y, 'LineWidth', 2);
xlabel('Time'); ylabel('Amplitude');
title('Sin + Cos Combination');"""
        },
        {
            "plot_type": "Polynomial Function",
            "description": "A MATLAB plot showing a quadratic polynomial function. Mathematical functions: quadratic function.",
            "code": """% Quadratic polynomial
x = -5:0.1:5;
y = x.^2 - 3*x + 2;
plot(x, y, 'r-', 'LineWidth', 2);
xlabel('x'); ylabel('y = x² - 3x + 2');
title('Quadratic Function');
grid on;"""
        },
        {
            "plot_type": "Exponential Growth",
            "description": "A MATLAB plot showing exponential growth function. Mathematical functions: exponential function.",
            "code": """% Exponential growth
x = 0:0.1:5;
y = exp(x);
semilogy(x, y, 'g-', 'LineWidth', 2);
xlabel('x'); ylabel('y = e^x');
title('Exponential Growth');
grid on;"""
        },
        {
            "plot_type": "Scatter Plot",
            "description": "A MATLAB scatter plot showing data point relationships. Visualization: scatter plot.",
            "code": """% Scatter plot with random data
x = randn(100, 1);
y = 2*x + randn(100, 1);
scatter(x, y, 50, 'b', 'filled');
xlabel('X values'); ylabel('Y values');
title('Scatter Plot Example');
grid on;"""
        },
        {
            "plot_type": "Bar Chart",
            "description": "A MATLAB bar chart showing categorical data. Visualization: bar chart.",
            "code": """% Bar chart
categories = {'A', 'B', 'C', 'D', 'E'};
values = [23, 45, 12, 67, 34];
bar(values);
set(gca, 'XTickLabel', categories);
xlabel('Categories'); ylabel('Values');
title('Bar Chart Example');"""
        },
        {
            "plot_type": "Histogram",
            "description": "A MATLAB histogram showing data distribution. Visualization: histogram, statistical plot.",
            "code": """% Histogram
data = randn(1000, 1);
histogram(data, 30);
xlabel('Values'); ylabel('Frequency');
title('Data Distribution Histogram');
grid on;"""
        },
        {
            "plot_type": "3D Surface Plot",
            "description": "A MATLAB 3D surface plot showing functional relationships. Visualization: surface plot.",
            "code": """% 3D surface plot
[X, Y] = meshgrid(-2:0.1:2, -2:0.1:2);
Z = X.^2 + Y.^2;
surf(X, Y, Z);
xlabel('X'); ylabel('Y'); zlabel('Z = X² + Y²');
title('3D Surface Plot');
colorbar;"""
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"{i}. 🖼️  {example['plot_type']}")
        print(f"   📝 Description: {example['description']}")
        print("   💻 Generated Code:")
        print("   ```matlab")
        for line in example['code'].strip().split('\n'):
            print(f"   {line}")
        print("   ```")
        print()

    print("✅ SYSTEM NOW RECOGNIZES:")
    print("• Trigonometric functions (sin, cos, tan)")
    print("• Polynomial functions (linear, quadratic, cubic)")
    print("• Exponential & logarithmic functions")
    print("• Plot types (scatter, bar, histogram, surface)")
    print("• Statistical visualizations")
    print("• 2D and 3D plots")
    print("• Time series and temporal data")
    print()
    print("🎯 Upload any MATLAB plot - the system will:")
    print("   1. Identify the plot type and functions")
    print("   2. Generate appropriate MATLAB recreation code")
    print("   3. Provide accurate explanations and syntax")

if __name__ == "__main__":
    demo_general_recognition()
