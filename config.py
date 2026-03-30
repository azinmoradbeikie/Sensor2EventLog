"""
Configuration parameters for HMM process analyzer
"""

# Process state definitions
PROCESS_STATES = {
    "production": ["Idle", "Fill", "HeatUp", "Hold", "Cool", "Discharge"],
    "cip": ["PreRinse", "Caustic", "InterRinse", "Acid", "FinalRinse", 
            "Sanitize", "Verification", "Standby"]
}

# Feature extraction parameters
FEATURE_CONFIG = {
    "window_sizes": [5],
    "stability_eps": 1,
    "peak_threshold": 0.1
}

# HMM parameters
HMM_CONFIG = {
    "covariance_type": "diag",
    "n_iter": 100,
    "random_seed": 42,
    "tol": 1e-6
}

# Diagnostic thresholds
DIAGNOSTIC_CONFIG = {
    "coverage_threshold": 0.6,
    "precision_threshold": 0.7,
    "explainability_threshold": 0.3
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "gantt_figsize": (14, 8),
    "colors": "Set3",
    "min_duration_for_label": 0.1  # hours
}

# File paths
PATHS = {
    "event_log": "pasteurization_event_log.csv",
    "filtered_log": "pasteurization_cleaned_event_log.csv",
    "gantt_chart": "process_gantt_chart.png"
}