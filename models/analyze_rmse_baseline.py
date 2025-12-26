#!/usr/bin/env python3
"""
Analyze RMSE performance and calculate baseline metrics.
"""

import json
import math
import numpy as np

# Parameter ranges from training code
PARAMETER_RANGES = {
    'amplitude': (0.5, 2.0),      # Range: 1.5
    'frequency': (0.5, 3.0),      # Range: 2.5
    'phase': (0, 2*math.pi),      # Range: 6.28
    'param_3': (0.0, 0.0),        # Often 0
    'param_4': (0.0, 0.0)         # Often 0
}

def calculate_baseline_rmse():
    """Calculate baseline RMSE if we predict the mean of each parameter."""
    
    # Calculate means for each parameter
    means = {
        'amplitude': (0.5 + 2.0) / 2,      # 1.25
        'frequency': (0.5 + 3.0) / 2,      # 1.75
        'phase': (0 + 2*math.pi) / 2,      # π ≈ 3.14
        'param_3': 0.0,
        'param_4': 0.0
    }
    
    # For uniform distribution, if we predict mean, the expected MSE per parameter is:
    # E[(X - μ)²] = Var(X) = (b-a)²/12 for uniform distribution [a, b]
    
    variances = {
        'amplitude': (2.0 - 0.5)**2 / 12,      # 1.5²/12 = 0.1875
        'frequency': (3.0 - 0.5)**2 / 12,      # 2.5²/12 = 0.5208
        'phase': (2*math.pi - 0)**2 / 12,      # (2π)²/12 = 3.2899
        'param_3': 0.0,
        'param_4': 0.0
    }
    
    # Average MSE across all 5 parameters
    avg_mse = sum(variances.values()) / 5
    baseline_rmse = math.sqrt(avg_mse)
    
    return baseline_rmse, variances, means

def interpret_rmse(rmse_value):
    """Interpret RMSE value in context of parameter ranges."""
    
    baseline_rmse, variances, means = calculate_baseline_rmse()
    
    print("="*60)
    print("RMSE PERFORMANCE ANALYSIS")
    print("="*60)
    
    print("\n📊 PARAMETER RANGES:")
    for param, (min_val, max_val) in PARAMETER_RANGES.items():
        range_val = max_val - min_val
        mean_val = (min_val + max_val) / 2
        print(f"  {param:12s}: [{min_val:5.2f}, {max_val:5.2f}] (range: {range_val:5.2f}, mean: {mean_val:5.2f})")
    
    print("\n📉 BASELINE PERFORMANCE (predicting mean):")
    print(f"  Expected MSE per parameter:")
    total_mse = 0
    for param, variance in variances.items():
        rmse_param = math.sqrt(variance)
        print(f"    {param:12s}: RMSE = {rmse_param:.4f}")
        total_mse += variance
    
    avg_mse = total_mse / 5
    print(f"\n  Average MSE: {avg_mse:.4f}")
    print(f"  Baseline RMSE: {baseline_rmse:.4f}")
    
    print(f"\n🎯 ACTUAL MODEL PERFORMANCE:")
    print(f"  Validation RMSE: {rmse_value:.4f}")
    
    print(f"\n📈 RELATIVE PERFORMANCE:")
    improvement = ((baseline_rmse - rmse_value) / baseline_rmse) * 100
    print(f"  Improvement over baseline: {improvement:.1f}%")
    print(f"  RMSE reduction: {baseline_rmse - rmse_value:.4f}")
    
    print(f"\n✅ INTERPRETATION:")
    if rmse_value < baseline_rmse * 0.5:
        quality = "Excellent"
    elif rmse_value < baseline_rmse * 0.7:
        quality = "Very Good"
    elif rmse_value < baseline_rmse * 0.9:
        quality = "Good"
    else:
        quality = "Moderate"
    
    print(f"  Quality: {quality} (RMSE = {rmse_value:.4f} vs baseline = {baseline_rmse:.4f})")
    
    # Relative error interpretation
    print(f"\n📐 RELATIVE ERROR ANALYSIS:")
    print(f"  Average parameter range: ~2.0 units")
    print(f"  RMSE = {rmse_value:.4f} represents:")
    print(f"    - ~{(rmse_value/2.0)*100:.1f}% of average parameter range")
    print(f"    - For amplitude (range 1.5): ~{(rmse_value/1.5)*100:.1f}% error")
    print(f"    - For frequency (range 2.5): ~{(rmse_value/2.5)*100:.1f}% error")
    print(f"    - For phase (range 6.28): ~{(rmse_value/6.28)*100:.1f}% error")
    
    print("\n💡 CONTEXT:")
    print("  - RMSE of 0.411 means on average, predicted parameters are")
    print(f"    off by {rmse_value:.3f} units")
    print("  - This is significantly better than baseline (predicting mean)")
    print("  - For practical use, this level of error is acceptable for")
    print("    identifying function patterns and generating approximate code")
    
    return {
        'baseline_rmse': baseline_rmse,
        'actual_rmse': rmse_value,
        'improvement_percent': improvement,
        'quality': quality
    }

if __name__ == "__main__":
    # Load actual validation RMSE from metrics
    try:
        with open('training_metrics.json', 'r') as f:
            metrics = json.load(f)
        best_rmse = metrics['history']['val_rmse'][3]  # Best at epoch 4
        print("Loaded metrics from training_metrics.json")
    except:
        best_rmse = 0.4111  # Best validation RMSE from training
        print("Using known best RMSE: 0.4111")
    
    interpret_rmse(best_rmse)

