"""
Main analysis pipeline for HMM process analyzer
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

from utils.feature_library import ModularFeatureLibrary
from utils.hmm_utils import (
    empirical_start_trans, emissions_from_labels, viterbi_decode, 
    print_evaluation, normalize_timestamps, create_interval_event_log_normalized,
    filter_brief_states, create_gantt_chart
)
import config


def analyze_process(data_path, feature_plan, mode="unsupervised", 
                   use_cip=False, n_unsup=None, random_seed=42):
    """
    Main analysis pipeline for process data.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV data file
    feature_plan : dict
        Feature extraction plan
    mode : str
        "supervised" or "unsupervised"
    use_cip : bool
        Whether to include CIP states
    n_unsup : int
        Number of states for unsupervised mode
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict with analysis results
    """
    # Load and prepare data
    df = load_and_prepare_data(data_path, use_cip)
    
    # Initialize feature library
    feature_lib = ModularFeatureLibrary(
        window_sizes=config.FEATURE_CONFIG["window_sizes"],
        stability_eps=config.FEATURE_CONFIG["stability_eps"],
        peak_threshold=config.FEATURE_CONFIG["peak_threshold"]
    )
    
    # Compute features and analyze rule performance
    print("Computing features and analyzing rule performance...")
    result = feature_lib.analyze_rule_performance(df, feature_plan)
    all_features = result['features']
    diagnostics = result['diagnostics']
    
    # Print diagnostic report
    if diagnostics:
        feature_lib.diagnostic_analyzer.print_diagnostic_report(diagnostics)
    
    # Prepare features for HMM
    event_features = all_features[[col for col in all_features.columns if col.startswith('event_')]]
    important_raw_features = ['T_roll_mean_5', 'Q_in_roll_mean_5', 'Q_out_roll_mean_5', 'T_diff']
    
    # Combine features
    features = pd.concat([all_features[important_raw_features], event_features], axis=1)
    
    # Split data into train/test
    train_features, test_features, df_train, df_test = split_train_test(df, features)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_test_scaled = scaler.transform(test_features)
    
    # Pack sequences for HMM
    X_train_np, lengths_train = pack_sequences(df_train, X_train_scaled)
    X_test_np, lengths_test = pack_sequences(df_test, X_test_scaled)
    
    # Create state mappings
    state_list = get_state_list(use_cip, n_unsup, mode)
    state_to_idx, idx_to_state = create_state_mappings(state_list)
    
    # Train and evaluate HMM
    if mode.lower() == "supervised":
        results = train_supervised_hmm(
            X_train_np, lengths_train, X_test_np, lengths_test,
            df_train, df_test, state_to_idx, idx_to_state, state_list
        )
    else:
        results = train_unsupervised_hmm(
            X_train_np, lengths_train, X_test_np, lengths_test,
            df_train, df_test, state_to_idx, idx_to_state, state_list, n_unsup
        )
    
    # Generate event log
    print("\nGenerating event log...")
    event_log = generate_event_log(df, results['model'], results['mapping'], 
                                 results['test_predictions'], scaler, features)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(event_log)
    
    return {
        'model': results['model'],
        'features': features,
        'event_log': event_log,
        'diagnostics': diagnostics,
        'predictions': results['test_predictions']
    }


def load_and_prepare_data(data_path, use_cip):
    """Load and prepare data for analysis."""
    df = pd.read_csv(data_path)
    
    # Determine state list
    state_list = config.PROCESS_STATES["production"]
    if use_cip:
        state_list += config.PROCESS_STATES["cip"]
    
    # Filter to relevant states and sort
    df = df[df["state"].isin(state_list)].copy()
    df.sort_values(["batch_id", "timestamp"], inplace=True)
    
    return df


def split_train_test(df, features, train_ratio=0.6):
    """Split data into training and testing sets."""
    batch_ids = df["batch_id"].unique()
    n_train = max(1, int(train_ratio * len(batch_ids)))
    
    train_batch_ids = set(batch_ids[:n_train])
    test_batch_ids = set(batch_ids[n_train:])
    
    # Split features
    X_train = features[df["batch_id"].isin(train_batch_ids)]
    X_test = features[df["batch_id"].isin(test_batch_ids)]
    
    # Split labels
    df_train = df[df["batch_id"].isin(train_batch_ids)]
    df_test = df[df["batch_id"].isin(test_batch_ids)]
    
    return X_train, X_test, df_train, df_test


def pack_sequences(df_subset, X_subset):
    """Pack sequences for HMM training."""
    lengths = df_subset.groupby("batch_id").size().tolist()
    if isinstance(X_subset, pd.DataFrame):
        X_subset = X_subset.values
    return X_subset, lengths


def get_state_list(use_cip, n_unsup, mode):
    """Get the list of states based on configuration."""
    state_list = config.PROCESS_STATES["production"]
    if use_cip:
        state_list += config.PROCESS_STATES["cip"]
    
    if mode.lower() == "unsupervised" and n_unsup is not None:
        n_states = n_unsup
    else:
        n_states = len(state_list)
    
    return state_list


def create_state_mappings(state_list):
    """Create mappings between state names and indices."""
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}
    return state_to_idx, idx_to_state


def train_supervised_hmm(X_train_np, lengths_train, X_test_np, lengths_test,
                        df_train, df_test, state_to_idx, idx_to_state, state_list):
    """Train and evaluate supervised HMM."""
    print("\nTraining supervised HMM...")
    
    # Convert labels to indices
    y_train_idx = df_train["state"].map(state_to_idx).values
    y_test_idx = df_test["state"].map(state_to_idx).values
    
    # Initialize from labels
    startprob_, transmat_ = empirical_start_trans(y_train_idx, lengths_train, len(state_list))
    means_, covars_ = emissions_from_labels(X_train_np, y_train_idx, len(state_list))
    
    # Build and fit HMM
    hmm = GaussianHMM(
        n_components=len(state_list),
        covariance_type="full",
        n_iter=30,
        init_params="",
        random_state=config.HMM_CONFIG["random_seed"],
        tol=config.HMM_CONFIG["tol"],
        verbose=False
    )
    hmm.startprob_ = startprob_
    hmm.transmat_ = transmat_
    hmm.means_ = means_
    hmm.covars_ = covars_
    
    hmm.fit(X_train_np, lengths_train)
    
    # Decode and evaluate
    y_pred_test = viterbi_decode(hmm, X_test_np, lengths_test)
    print_evaluation(y_test_idx, y_pred_test, idx_to_state, state_list, 
                    title="Supervised HMM (Test)")
    
    return {
        'model': hmm,
        'mapping': idx_to_state,
        'test_predictions': y_pred_test
    }


def train_unsupervised_hmm(X_train_np, lengths_train, X_test_np, lengths_test,
                          df_train, df_test, state_to_idx, idx_to_state, state_list, n_unsup):
    """Train and evaluate unsupervised HMM with state mapping."""
    print("\nTraining unsupervised HMM...")
    
    n_states = n_unsup if n_unsup is not None else len(state_list)
    
    # Train unsupervised HMM
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type=config.HMM_CONFIG["covariance_type"],
        n_iter=config.HMM_CONFIG["n_iter"],
        random_state=config.HMM_CONFIG["random_seed"],
        tol=config.HMM_CONFIG["tol"],
        init_params="stmc",
        params="stmc"
    )
    hmm.fit(X_train_np, lengths_train)
    
    # Predict and map states
    y_train_true = df_train["state"].map(state_to_idx).values
    y_train_hat = viterbi_decode(hmm, X_train_np, lengths_train)
    
    # Build contingency matrix for mapping
    K = len(state_list)
    cont = np.zeros((K, K), dtype=int)
    for t, p in zip(y_train_true, y_train_hat):
        if t < K and p < K:
            cont[t, p] += 1
    
    # Optimal mapping using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cont.max() - cont)
    mapping = {pred: true for true, pred in zip(row_ind, col_ind)}
    
    # Decode test set and map states
    y_test_hat = viterbi_decode(hmm, X_test_np, lengths_test)
    y_test_mapped = np.array([mapping.get(s, 0) for s in y_test_hat], dtype=int)
    y_test_true = df_test["state"].map(state_to_idx).values
    
    print_evaluation(y_test_true, y_test_mapped, idx_to_state, state_list,
                    title="Unsupervised HMM (mapped) — Test")
    
    # Create full mapping for event log
    state_mapping = {pred: idx_to_state[true] for pred, true in mapping.items() if true in idx_to_state}
    
    return {
        'model': hmm,
        'mapping': state_mapping,
        'test_predictions': y_test_hat
    }


def generate_event_log(df, hmm, state_mapping, predictions, scaler, features):
    """Generate event log from HMM predictions."""
    # Normalize timestamps
    df_normalized = normalize_timestamps(df)
    
    # Prepare test features
    X_test_scaled = scaler.transform(features)
    X_test_np, lengths_test = pack_sequences(df, X_test_scaled)
    
    # Get predictions for full dataset
    y_pred_full = viterbi_decode(hmm, X_test_np, lengths_test)
    
    # Create event log
    event_log = create_interval_event_log_normalized(
        df_normalized, y_pred_full, state_mapping
    )
    
    # Filter brief states
    filtered_log = filter_brief_states(
        event_log, 
        min_duration_seconds=2.0
    )
    
    # Save logs
    event_log.to_csv(config.PATHS["event_log"], index=False)
    filtered_log.to_csv(config.PATHS["filtered_log"], index=False)
    
    print(f"Event log saved to: {config.PATHS['event_log']}")
    print(f"Filtered event log saved to: {config.PATHS['filtered_log']}")
    
    return filtered_log


def create_visualizations(event_log):
    """Create visualization of process execution."""
    # Create Gantt chart
    gantt_chart = create_gantt_chart(
        event_log,
        max_cases=10,
        figsize=config.VISUALIZATION_CONFIG["gantt_figsize"],
        color_map=config.VISUALIZATION_CONFIG["colors"]
    )
    
    # Save and show
    gantt_chart.savefig(config.PATHS["gantt_chart"], dpi=300, bbox_inches='tight')
    gantt_chart.show()
    
    print(f"Gantt chart saved to: {config.PATHS['gantt_chart']}")


if __name__ == "__main__":
    # Example feature plan
    feature_plan = {
        'statistical': ['T', 'Q_in', 'Q_out'],
        'temporal': ['T', 'Q_in', 'Q_out'],
        'stability': ['T', 'Q_in', 'Q_out'],
        'interaction': [['T', 'Q_in', 'Q_out']],
        'event': [
            '(T_diff_smooth > 1)', 
            '(T_diff_smooth < -1)',
            '(Q_out > 0.3)',
            '(T > 70) & (T_stable_flag == 1)',
            '(Q_in > 0.3) AND (T_diff < 0.2)'
        ],
        'contextual': []
    }
    
    # Run analysis
    results = analyze_process(
        data_path="synthetic_pasteurization_with_cip_signals.csv",
        feature_plan=feature_plan,
        mode="unsupervised",
        use_cip=False,
        n_unsup=None,
        random_seed=42
    )