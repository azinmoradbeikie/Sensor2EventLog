"""
HMM utility functions for training, evaluation, and event log generation
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def empirical_start_trans(labels, lengths, n_states):
    """Estimate startprob_ and transmat_ from labeled sequences."""
    start = np.zeros(n_states)
    trans = np.zeros((n_states, n_states))
    idx = 0
    for L in lengths:
        seq = labels[idx:idx+L]
        start[seq[0]] += 1
        for i in range(L-1):
            trans[seq[i], seq[i+1]] += 1
        idx += L
    # normalize with small epsilon to avoid zeros
    start = (start + 1e-6) / (start.sum() + 1e-6*n_states)
    trans = (trans + 1e-6)
    trans /= trans.sum(axis=1, keepdims=True)
    return start, trans


def emissions_from_labels(X_np, labels_np, n_states):
    """Compute means and covariances per labeled state."""
    D = X_np.shape[1]
    means = np.zeros((n_states, D))
    covars = np.zeros((n_states, D, D))
    for s in range(n_states):
        sel = (labels_np == s)
        Xi = X_np[sel]
        if len(Xi) < 2:
            # fallback tiny variance
            means[s] = 0.0
            covars[s] = np.eye(D)*1e-2
        else:
            means[s] = Xi.mean(axis=0)
            covars[s] = np.cov(Xi.T) + np.eye(D)*1e-6
    return means, covars


def viterbi_decode(model, X_np, lengths):
    """Wrapper for HMM Viterbi decoding."""
    return model.predict(X_np, lengths)


def print_evaluation(y_true_idx, y_pred_idx, idx_to_state, state_list, title=""):
    """Print classification report and confusion matrix."""
    labs_true = [idx_to_state[i] for i in y_true_idx]
    labs_pred = [idx_to_state.get(i, f"UNK{i}") for i in y_pred_idx]
    print(f"\n== {title} ==")
    print(classification_report(labs_true, labs_pred, labels=state_list, zero_division=0))
    cm = confusion_matrix(labs_true, labs_pred, labels=state_list)
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=state_list, columns=state_list))


def normalize_timestamps(df, timestamp_col="timestamp", case_id_col="batch_id", base_date="2023-01-01"):
    """
    Normalize timestamps by handling different time units properly.
    """
    df_normalized = df.copy()
    
    # Check timestamp format
    print(f"Original timestamp sample: {df[timestamp_col].iloc[:5].tolist()}")
    
    # Convert numeric seconds to datetime or parse as datetime
    if np.issubdtype(df[timestamp_col].dtype, np.number):
        print("Timestamps are numeric - assuming they represent seconds")
        base_datetime = pd.to_datetime(base_date)
        df_normalized[timestamp_col] = base_datetime + pd.to_timedelta(df[timestamp_col], unit='s')
    else:
        try:
            df_normalized[timestamp_col] = pd.to_datetime(df[timestamp_col])
            print("Timestamps successfully parsed as datetime")
        except:
            print("Could not parse timestamps. Please check the format.")
            return df
    
    # Normalize each case to start at base_date
    case_groups = df_normalized.groupby(case_id_col)
    
    for case_id, case_data in case_groups:
        case_start = case_data[timestamp_col].min()
        time_deltas = case_data[timestamp_col] - case_start
        df_normalized.loc[case_data.index, timestamp_col] = pd.to_datetime(base_date) + time_deltas
    
    return df_normalized


def create_interval_event_log_normalized(df, y_pred, state_mapping, 
                                        case_id_col="batch_id", timestamp_col="timestamp"):
    """
    Create interval-based event log using normalized timestamps.
    """
    df_with_pred = df.copy()
    df_with_pred['predicted_state'] = [state_mapping.get(i, f"Unknown_{i}") for i in y_pred]
    
    event_log_segments = []
    
    for case_id in df_with_pred[case_id_col].unique():
        case_data = df_with_pred[df_with_pred[case_id_col] == case_id].copy()
        case_data = case_data.sort_values(timestamp_col)
        
        current_state = None
        segment_start = None
        segment_indices = []
        
        for idx, row in case_data.iterrows():
            if current_state is None:
                current_state = row['predicted_state']
                segment_start = row[timestamp_col]
                segment_indices = [idx]
            elif row['predicted_state'] == current_state:
                segment_indices.append(idx)
            else:
                segment_end = case_data.loc[segment_indices[-1], timestamp_col]
                duration = (pd.to_datetime(segment_end) - pd.to_datetime(segment_start)).total_seconds()
                event_log_segments.append({
                    'case_id': case_id,
                    'activity': current_state,
                    'start_timestamp': segment_start,
                    'end_timestamp': segment_end,
                    'duration_seconds': duration,
                    'event_count': len(segment_indices)
                })
                current_state = row['predicted_state']
                segment_start = row[timestamp_col]
                segment_indices = [idx]
        
        # Add the last segment
        if current_state is not None and segment_start is not None:
            segment_end = case_data.loc[segment_indices[-1], timestamp_col]
            duration = (pd.to_datetime(segment_end) - pd.to_datetime(segment_start)).total_seconds()
            
            event_log_segments.append({
                'case_id': case_id,
                'activity': current_state,
                'start_timestamp': segment_start,
                'end_timestamp': segment_end,
                'duration_seconds': duration,
                'event_count': len(segment_indices)
            })
    
    event_log = pd.DataFrame(event_log_segments)
    event_log['activity_sequence'] = event_log.groupby('case_id').cumcount() + 1
    
    event_log = event_log[['case_id', 'activity_sequence', 'activity', 
                          'start_timestamp', 'end_timestamp', 
                          'duration_seconds', 'event_count']]
    
    return event_log


def filter_brief_states(event_log, min_duration_seconds=5.0):
    """
    Remove state segments that are too brief by merging them with adjacent states.
    """
    filtered_segments = []
    
    for case_id in event_log['case_id'].unique():
        case_data = event_log[event_log['case_id'] == case_id].copy()
        
        i = 0
        while i < len(case_data):
            current_segment = case_data.iloc[i]
            
            # If segment is too brief, merge with previous or next
            if current_segment['duration_seconds'] < min_duration_seconds and len(case_data) > 1:
                
                if i == 0:  # First segment - merge with next
                    next_segment = case_data.iloc[i + 1]
                    merged_segment = {
                        'case_id': case_id,
                        'activity': next_segment['activity'],
                        'start_timestamp': current_segment['start_timestamp'],
                        'end_timestamp': next_segment['end_timestamp'],
                        'duration_seconds': current_segment['duration_seconds'] + next_segment['duration_seconds'],
                        'event_count': current_segment['event_count'] + next_segment['event_count']
                    }
                    filtered_segments.append(merged_segment)
                    i += 2  # Skip next segment since we merged it
                    
                elif i == len(case_data) - 1:  # Last segment - merge with previous
                    prev_segment = case_data.iloc[i - 1]
                    merged_segment = {
                        'case_id': case_id,
                        'activity': prev_segment['activity'],
                        'start_timestamp': prev_segment['start_timestamp'],
                        'end_timestamp': current_segment['end_timestamp'],
                        'duration_seconds': prev_segment['duration_seconds'] + current_segment['duration_seconds'],
                        'event_count': prev_segment['event_count'] + current_segment['event_count']
                    }
                    # Replace the last segment we added
                    filtered_segments = filtered_segments[:-1]
                    filtered_segments.append(merged_segment)
                    i += 1
                    
                else:  # Middle segment - merge with previous
                    prev_segment = case_data.iloc[i - 1]
                    merged_segment = {
                        'case_id': case_id,
                        'activity': prev_segment['activity'],
                        'start_timestamp': prev_segment['start_timestamp'],
                        'end_timestamp': current_segment['end_timestamp'],
                        'duration_seconds': prev_segment['duration_seconds'] + current_segment['duration_seconds'],
                        'event_count': prev_segment['event_count'] + current_segment['event_count']
                    }
                    # Replace the last segment we added
                    filtered_segments = filtered_segments[:-1]
                    filtered_segments.append(merged_segment)
                    i += 1
            else:
                # Keep segments that are long enough
                filtered_segments.append(current_segment.to_dict())
                i += 1
    
    # Create new event log
    filtered_log = pd.DataFrame(filtered_segments)
    
    # Recalculate activity sequence
    filtered_log['activity_sequence'] = filtered_log.groupby('case_id').cumcount() + 1
    
    return filtered_log


def create_gantt_chart(event_log, max_cases=10, figsize=(14, 8), color_map='Set3'):
    """
    Create Gantt chart visualization of process execution.
    """
    plt.figure(figsize=figsize)
    
    activities = event_log['activity'].unique()
    colors = plt.cm.get_cmap(color_map)(np.linspace(0, 1, len(activities)))
    color_dict = dict(zip(activities, colors))
    
    case_ids = event_log['case_id'].unique()[:max_cases]
    
    for i, case_id in enumerate(case_ids):
        case_data = event_log[event_log['case_id'] == case_id]
        
        for _, activity_row in case_data.iterrows():
            start = pd.to_datetime(activity_row['start_timestamp'])
            end = pd.to_datetime(activity_row['end_timestamp'])
            duration = (end - start).total_seconds() / 3600  # Convert to hours
            
            plt.barh(y=i, width=duration, left=start, 
                    color=color_dict[activity_row['activity']], 
                    edgecolor='black', alpha=0.7)
            
            # Add activity label for longer segments
            if duration > 0.1:  # Only label segments longer than 6 minutes
                plt.text(start + pd.Timedelta(seconds=duration*3600/2), i, 
                        activity_row['activity'], ha='center', va='center', 
                        fontsize=8, fontweight='bold')
    
    plt.yticks(range(len(case_ids)), case_ids)
    plt.xlabel('Time (from normalized start)')
    plt.ylabel('Case ID')
    plt.title(f'Process Execution Gantt Chart (First {len(case_ids)} Cases)')
    
    # Create legend
    legend_patches = [plt.Rectangle((0,0),1,1, color=color_dict[act]) for act in activities]
    plt.legend(legend_patches, activities, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt