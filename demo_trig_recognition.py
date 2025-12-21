#!/usr/bin/env python3
"""
Demo of improved trigonometric function recognition in BLIP-2.
"""

def demo_trig_recognition():
    """Show what the improved trigonometric recognition can achieve."""

    print("🎯 MATLAB Vision RAG - Trigonometric Function Recognition Demo")
    print("=" * 65)
    print()

    print("🖼️  INPUT: User uploads image of sin(t) + cos(2t) combination plot")
    print("❓ QUESTION: 'What function creates this type of plot?'")
    print()

    print("📝 IMAGE DESCRIPTION (BLIP-2 Enhanced):")
    print('"A MATLAB plot showing a combination of sine and cosine functions. Mathematical functions: combination of sine and cosine functions. This appears to be a MATLAB-generated visualization."')
    print()

    print("🔍 DETECTED FUNCTIONS:")
    print("• sine function ✓")
    print("• cosine function ✓")
    print("• trigonometric combination ✓")
    print()

    print("🤖 ENRICHED QUERY:")
    print('"Based on an uploaded image showing a combination of sine and cosine functions: ..."')
    print('"Please provide MATLAB code that recreates this plot. Include the mathematical functions sine, cosine and show how to combine them in MATLAB."')
    print()

    print("💻 GENERATED MATLAB CODE:")
    print("```matlab")
    print("% Recreate the trigonometric function combination")
    print("t = 0:0.01:4*pi;  % Time vector")
    print("y1 = sin(t);       % Sine function")
    print("y2 = cos(2*t);     % Cosine with different frequency")
    print("y = y1 + 0.5*y2;   % Linear combination")
    print()
    print("% Create the plot")
    print("figure('Position', [100, 100, 800, 400]);")
    print("plot(t, y, 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8]);")
    print("xlabel('Time (t)');")
    print("ylabel('Amplitude');")
    print("title('Combination of sin(t) + 0.5*cos(2t)');")
    print("grid on;")
    print("axis tight;")
    print("```")
    print()

    print("📚 EXPLANATION:")
    print("This code recreates the exact trigonometric function combination shown in your plot,")
    print("using sin(t) and cos(2*t) functions with a linear combination.")
    print()

    print("✅ KEY IMPROVEMENTS:")
    print("• Explicit trigonometric function recognition (sin, cos)")
    print("• Function-specific query enrichment")
    print("• Accurate mathematical recreation")
    print("• Proper MATLAB syntax for function combinations")

if __name__ == "__main__":
    demo_trig_recognition()
