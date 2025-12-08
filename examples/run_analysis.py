"""
Example usage of the HMM process analyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import analyze_process

# Example 1: Basic analysis with default settings
def example_basic():
    """Basic example with temperature and flow signals."""
    feature_plan = {
        'statistical': ['T', 'Q_in', 'Q_out'],
        'temporal': ['T', 'Q_in', 'Q_out'],
        'stability': ['T', 'Q_in', 'Q_out'],
        'interaction': [['T', 'Q_in'], ['T', 'Q_out']],
        'event': [
            '(T > 70) & (T_stable_flag == 1)',  # High stable temperature
            '(Q_in > 0.3) & (Q_out < 0.1)',     # Filling phase
            '(T_diff_smooth < -0.5)',           # Cooling phase
            '(Q_out > 0.2) & (T < 40)',         # Discharge phase
        ],
        'contextual': []
    }
    
    results = analyze_process(
        data_path="synthetic_pasteurization_with_cip_signals.csv",
        feature_plan=feature_plan,
        mode="unsupervised"
    )
    
    return results

# Example 2: Supervised learning with custom states
def example_supervised():
    """Supervised learning example with known states."""
    feature_plan = {
        'statistical': ['T', 'Q_in'],
        'temporal': ['T', 'Q_in'],
        'stability': ['T', 'Q_in'],
        'interaction': [['T', 'Q_in']],
        'event': [
            {'fill_start': '(Q_in > 0.25) & (T_diff < 0.1)'},
            {'heat_start': '(T_diff_smooth > 0.5) & (T > 30)'},
            {'cool_start': '(T_diff_smooth < -0.5) & (T < 80)'},
            {'discharge_start': '(Q_out > 0.2) & (T < 40)'}
        ],
        'contextual': ['batch_position']
    }
    
    results = analyze_process(
        data_path="synthetic_pasteurization_with_cip_signals.csv",
        feature_plan=feature_plan,
        mode="supervised"
    )
    
    return results

# Example 3: Custom feature extraction
def example_custom_features():
    """Example with custom feature parameters."""
    from utils.feature_library import ModularFeatureLibrary
    
    # Create custom feature library
    feature_lib = ModularFeatureLibrary(
        window_sizes=[3, 5, 10],  # Multiple window sizes
        stability_eps=0.5,         # Tighter stability threshold
        peak_threshold=0.05        # More sensitive peak detection
    )
    
    # Custom feature plan
    feature_plan = {
        'statistical': ['T', 'Q_in', 'Q_out'],
        'temporal': ['T', 'Q_in', 'Q_out'],
        'stability': ['T'],  # Only temperature stability
        'interaction': [['T', 'Q_in'], ['Q_in', 'Q_out']],
        'event': [
            'T_roll_mean_5 > 65',
            'T_roll_mean_10 - T_roll_mean_3 > 10',
            'Q_in_roll_mean_5 > 0.3 AND Q_out_roll_mean_5 < 0.1'
        ],
        'contextual': []
    }
    
    # Load data
    import pandas as pd
    df = pd.read_csv("synthetic_pasteurization_with_cip_signals.csv")
    
    # Compute features
    features = feature_lib.compute_features(df, feature_plan)
    print(f"Generated {features.shape[1]} features")
    
    return features

if __name__ == "__main__":
    print("Running example 1: Basic analysis...")
    results1 = example_basic()
    
    print("\n" + "="*50 + "\n")
    
    print("Running example 2: Supervised learning...")
    results2 = example_supervised()
    
    print("\n" + "="*50 + "\n")
    
    print("Running example 3: Custom feature extraction...")
    features = example_custom_features()
    
    print("\nAll examples completed successfully!")