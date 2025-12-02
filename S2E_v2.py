import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks
from typing import List, Dict, Union, Optional, Tuple
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from scipy.optimize import linear_sum_assignment
import re

class RuleDiagnosticAnalyzer:
    """
    Analyzes rule performance using coverage, precision, and explainability metrics
    """
    
    def __init__(self, coverage_threshold: float = 0.6, precision_threshold: float = 0.7, 
                 explainability_threshold: float = 0.3):
        self.c_low = coverage_threshold
        self.p_low = precision_threshold
        self.epsilon_unex = explainability_threshold
        
    def compute_rule_metrics(self, df: pd.DataFrame, event_features: pd.DataFrame, 
                           state_column: str = 'state') -> Dict:
        """
        Compute coverage, precision, and explainability metrics for all event features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original dataframe with state labels
        event_features : pd.DataFrame
            DataFrame containing event rule features (binary columns)
        state_column : str
            Column name containing state labels
            
        Returns:
        --------
        Dict with comprehensive diagnostic results
        """
        results = {
            'rule_metrics': {},
            'state_metrics': {},
            'recommendations': [],
            'unexplainable_states': []
        }
        
        states = df[state_column].unique()
        
        # Compute metrics for each rule-state combination
        rule_metrics = {}
        for rule_col in event_features.columns:
            if rule_col.startswith('event_'):
                rule_metrics[rule_col] = {}
                
                for state in states:
                    # Get timestamps for this state
                    state_mask = df[state_column] == state
                    state_timestamps = state_mask[state_mask].index
                    
                    if len(state_timestamps) == 0:
                        continue
                    
                    # Rule activations for this state
                    rule_activations_state = event_features.loc[state_timestamps, rule_col]
                    
                    # Total rule activations
                    total_rule_activations = event_features[rule_col].sum()
                    
                    # Compute coverage and precision
                    coverage = rule_activations_state.sum() / len(state_timestamps)
                    precision = (rule_activations_state.sum() / total_rule_activations 
                               if total_rule_activations > 0 else 0)
                    
                    rule_metrics[rule_col][state] = {
                        'coverage': coverage,
                        'precision': precision,
                        'effectiveness': np.sqrt(coverage * precision) if coverage > 0 and precision > 0 else 0
                    }
        
        results['rule_metrics'] = rule_metrics
        
        # Compute state-level metrics
        state_metrics = {}
        for state in states:
            state_mask = df[state_column] == state
            state_timestamps = state_mask[state_mask].index
            
            if len(state_timestamps) == 0:
                continue
                
            # Find best coverage across all rules for this state
            best_coverage = 0
            best_rule = None
            
            for rule_col, state_metrics_dict in rule_metrics.items():
                if state in state_metrics_dict:
                    coverage = state_metrics_dict[state]['coverage']
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_rule = rule_col
            
            explainability = best_coverage
            gap = 1 - explainability
            
            state_metrics[state] = {
                'explainability': explainability,
                'gap': gap,
                'best_rule': best_rule,
                'best_coverage': best_coverage,
                'state_frequency': len(state_timestamps) / len(df),
                'unexplainable': explainability < self.epsilon_unex
            }
            
            if explainability < self.epsilon_unex:
                results['unexplainable_states'].append(state)
        
        results['state_metrics'] = state_metrics
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(rule_metrics, state_metrics)
        
        return results
    
    def _generate_recommendations(self, rule_metrics: Dict, state_metrics: Dict) -> List[Dict]:
        """Generate actionable recommendations based on diagnostic metrics"""
        recommendations = []
        
        # Analyze each rule-state combination
        for rule_col, state_dict in rule_metrics.items():
            for state, metrics in state_dict.items():
                coverage = metrics['coverage']
                precision = metrics['precision']
                
                # Rule categorization and recommendations
                if coverage >= self.c_low and precision >= self.p_low:
                    # Optimal rule - no action needed
                    continue
                    
                elif coverage >= self.c_low and precision < self.p_low:
                    # Overly sensitive rule
                    recommendations.append({
                        'type': 'OVERLY_SENSITIVE_RULE',
                        'rule': rule_col,
                        'state': state,
                        'coverage': coverage,
                        'precision': precision,
                        'action': f"Rule '{rule_col}' for state '{state}' has good coverage ({coverage:.1%}) but low precision ({precision:.1%}). Add temporal stability constraints or interaction features to reduce false positives.",
                        'priority': 'HIGH',
                        'suggested_families': ['stability', 'interaction']
                    })
                    
                elif coverage < self.c_low and precision >= self.p_low:
                    # Overly specific rule
                    recommendations.append({
                        'type': 'OVERLY_SPECIFIC_RULE', 
                        'rule': rule_col,
                        'state': state,
                        'coverage': coverage,
                        'precision': precision,
                        'action': f"Rule '{rule_col}' for state '{state}' has high precision ({precision:.1%}) but low coverage ({coverage:.1%}). Relax thresholds or remove restrictive conditions.",
                        'priority': 'MEDIUM',
                        'suggested_families': ['temporal', 'statistical']
                    })
                    
                elif coverage < self.c_low and precision < self.p_low:
                    # Ineffective rule
                    recommendations.append({
                        'type': 'INEFFECTIVE_RULE',
                        'rule': rule_col,
                        'state': state,
                        'coverage': coverage,
                        'precision': precision,
                        'action': f"Rule '{rule_col}' for state '{state}' performs poorly (coverage: {coverage:.1%}, precision: {precision:.1%}). Consider complete redesign with alternative sensor combinations.",
                        'priority': 'HIGH',
                        'suggested_families': ['all']
                    })
        
        # Analyze unexplainable states
        for state, metrics in state_metrics.items():
            if metrics['unexplainable']:
                recommendations.append({
                    'type': 'UNEXPLAINABLE_STATE',
                    'state': state,
                    'explainability': metrics['explainability'],
                    'frequency': metrics['state_frequency'],
                    'action': f"State '{state}' is largely unexplained (explainability: {metrics['explainability']:.1%}). Consider state decomposition, feature space expansion, or probabilistic approaches.",
                    'priority': 'CRITICAL' if metrics['state_frequency'] > 0.1 else 'HIGH',
                    'suggested_approaches': ['state_decomposition', 'feature_expansion', 'probabilistic_modeling']
                })
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def print_diagnostic_report(self, diagnostic_results: Dict):
        """Print comprehensive diagnostic report"""
        print("=" * 80)
        print("RULE DIAGNOSTIC REPORT")
        print("=" * 80)
        
        # Rule performance summary
        print("\n1. RULE PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        rule_metrics = diagnostic_results['rule_metrics']
        for rule_col, state_dict in rule_metrics.items():
            print(f"\nRule: {rule_col}")
            for state, metrics in state_dict.items():
                print(f"  State: {state:15} | Coverage: {metrics['coverage']:6.1%} | "
                      f"Precision: {metrics['precision']:6.1%} | "
                      f"Effectiveness: {metrics['effectiveness']:6.1%}")
        
        # State explainability
        print("\n2. STATE EXPLAINABILITY ANALYSIS:")
        print("-" * 40)
        
        state_metrics = diagnostic_results['state_metrics']
        for state, metrics in state_metrics.items():
            unexplainable_flag = " ⚠ UNEXPLAINABLE" if metrics['unexplainable'] else ""
            print(f"State: {state:15} | Explainability: {metrics['explainability']:6.1%} | "
                  f"Best Rule: {metrics['best_rule'] or 'None'}{unexplainable_flag}")
        
        # Recommendations
        print("\n3. ACTIONABLE RECOMMENDATIONS:")
        print("-" * 40)
        
        for i, rec in enumerate(diagnostic_results['recommendations'], 1):
            print(f"\n{i}. [{rec['priority']}] {rec['type']}")
            print(f"   {rec['action']}")
            
            if 'suggested_families' in rec:
                print(f"   Suggested feature families: {', '.join(rec['suggested_families'])}")
            if 'suggested_approaches' in rec:
                print(f"   Suggested approaches: {', '.join(rec['suggested_approaches'])}")
        
        # Summary statistics
        print("\n4. SUMMARY STATISTICS:")
        print("-" * 40)
        
        total_states = len(state_metrics)
        unexplainable_states = len(diagnostic_results['unexplainable_states'])
        avg_explainability = np.mean([m['explainability'] for m in state_metrics.values()])
        
        print(f"Total states analyzed: {total_states}")
        print(f"Unexplainable states: {unexplainable_states} ({unexplainable_states/total_states:.1%})")
        print(f"Average explainability: {avg_explainability:.1%}")
        print(f"Recommendations generated: {len(diagnostic_results['recommendations'])}")

# Enhanced ModularFeatureLibrary with diagnostic capabilities
class ModularFeatureLibrary:
    def __init__(self, window_sizes=[5], stability_eps=1, peak_threshold=0.1):
        self.window_sizes = window_sizes
        self.stability_eps = stability_eps
        self.peak_threshold = peak_threshold
        self.feature_families = {
            'statistical': self._compute_statistical_features,
            'temporal': self._compute_temporal_features,
            'stability': self._compute_stability_features,
            'interaction': self._compute_interaction_features,
            'event': self._compute_event_features,
            'contextual': self._compute_contextual_features
        }
        self.diagnostic_analyzer = RuleDiagnosticAnalyzer()
        self._feature_cache = {}
    
    def _normalize_rule_expr(self, expr: str) -> str:
        """Convert human-friendly logical ops to pandas-style bitwise ops and normalize spacing."""
        s = expr.strip()
        s = re.sub(r'\bAND\b', '&', s, flags=re.I)
        s = re.sub(r'\bOR\b', '|', s, flags=re.I)
        s = re.sub(r'\bNOT\b', '~', s, flags=re.I)
        s = re.sub(r'\band\b', '&', s)
        s = re.sub(r'\bor\b', '|', s)
        s = re.sub(r'\bnot\b', '~', s)
        s = re.sub(r'\s*([&|~><=!]+)\s*', r' \1 ', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()
    
    def _evaluate_rule(self, rule_expr: str, available_features: pd.DataFrame) -> pd.Series:
        """Evaluate a rule expression using the available features."""
        normalized_expr = self._normalize_rule_expr(rule_expr)
        
        try:
            eval_env = {col: available_features[col] for col in available_features.columns}
            eval_env.update({
                'np': np, 'pd': pd, 'abs': np.abs, 'min': np.minimum, 'max': np.maximum
            })
            
            result = eval(normalized_expr, {"__builtins__": {}}, eval_env)
            
            if isinstance(result, pd.Series):
                return result.astype(bool)
            else:
                return pd.Series([bool(result)] * len(available_features), 
                               index=available_features.index)
                
        except Exception as e:
            print(f"Error evaluating rule '{rule_expr}': {e}")
            return pd.Series(False, index=available_features.index)
    
    def _safe_norm(self, series):
        """Z-score normalization"""
        return (series - series.mean()) / (series.std() + 1e-8)
    
    def _safe_ratio(self, a, b):
        """Safe ratio calculation"""
        a_safe = np.abs(a) + 1e-6
        b_safe = np.abs(b) + 1e-6
        ratio = np.log1p(a_safe) - np.log1p(b_safe)
        sign = np.sign(a * b)
        return ratio * sign
    
    def _compute_statistical_features(self, df, signals, **kwargs):
        """Statistical Features Family"""
        features = pd.DataFrame(index=df.index)
        
        for signal in signals:
            s = df[signal]
            for win in self.window_sizes:
                roll = s.rolling(win, min_periods=1)
                features[f"{signal}_roll_mean_{win}"] = roll.mean()
        
        return features
    
    def _compute_temporal_features(self, df, signals, **kwargs):
        """Temporal Dynamics Family"""
        features = pd.DataFrame(index=df.index)
        
        for signal in signals:
            s = df[signal]
            diff = s.diff().fillna(0)
            features[f"{signal}_diff"] = diff
            features[f"{signal}_diff_sign"] = np.sign(diff)
            features[f"{signal}_diff_smooth"] = diff.ewm(span=5).mean()
            features[f"{signal}_abs_diff"] = np.abs(diff)
        
        return features
    
    def _compute_stability_features(self, df, signals, **kwargs):
        """Stability Features Family"""
        features = pd.DataFrame(index=df.index)
        
        for signal in signals:
            s = df[signal]
            diff = s.diff().fillna(0)
            features[f"{signal}_stability"] = 1.0 / (1.0 + np.abs(diff))
            features[f"{signal}_stable_flag"] = (np.abs(diff) < self.stability_eps).astype(int)
            
            stable_periods = (np.abs(diff) < self.stability_eps)
            consecutive_stable = stable_periods.groupby((~stable_periods).cumsum()).cumsum()
            features[f"{signal}_consecutive_stable"] = consecutive_stable
        
        return features
    
    def _compute_interaction_features(self, df, signals, **kwargs):
        """Interaction Features Family"""
        features = pd.DataFrame(index=df.index)
        
        if len(signals) < 2:
            return features
        
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                sig1, sig2 = signals[i], signals[j]
                features[f"{sig1}_x_{sig2}"] = df[sig1] * df[sig2]
                features[f"{sig1}_ratio_{sig2}"] = self._safe_ratio(df[sig1], df[sig2])
        
        return features
    
    def _compute_event_features(self, df, signals, **kwargs):
        """Event/Regime Features Family with rule-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Create a comprehensive set of available features for rule evaluation
        available_features = df.copy()
        
        # Pre-compute all necessary derived features for ALL numeric columns
        for signal in df.columns:
            if pd.api.types.is_numeric_dtype(df[signal]):
                try:
                    # Compute temporal features
                    diff = df[signal].diff().fillna(0)
                    available_features[f"{signal}_diff"] = diff
                    available_features[f"{signal}_diff_smooth"] = diff.ewm(span=5).mean()
                    available_features[f"{signal}_abs_diff"] = np.abs(diff)
                    
                    # Compute stability features  
                    available_features[f"{signal}_stability"] = 1.0 / (1.0 + np.abs(diff))
                    available_features[f"{signal}_stable_flag"] = (np.abs(diff) < self.stability_eps).astype(int)
                    
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not compute derived features for {signal}: {e}")
        
        # Process event definitions
        rule_counter = 0
        for signal_def in signals:
            if isinstance(signal_def, str) and any(op in signal_def for op in ['>', '<', '==', '&', '|']):
                rule_counter += 1
                try:
                    fixed_expr = self._fix_rule_parentheses(signal_def)
                    rule_result = self._evaluate_rule(fixed_expr, available_features)
                    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', signal_def[:20])
                    feature_name = f"event_{clean_name}"
                    if feature_name in features.columns:
                        feature_name = f"event_{clean_name}_{rule_counter}"
                    features[feature_name] = rule_result.astype(int)
                    print(f"Created event feature: {feature_name} from rule: {signal_def}")
                except Exception as e:
                    print(f"Error processing rule '{signal_def}': {e}")
                    features[f"event_rule_error_{rule_counter}"] = 0
                    
            elif isinstance(signal_def, dict):
                for rule_name, rule_expr in signal_def.items():
                    try:
                        fixed_expr = self._fix_rule_parentheses(rule_expr)
                        rule_result = self._evaluate_rule(fixed_expr, available_features)
                        features[f"event_{rule_name}"] = rule_result.astype(int)
                        print(f"Created named event feature: event_{rule_name}")
                    except Exception as e:
                        print(f"Error processing named rule '{rule_name}': {e}")
                        features[f"event_{rule_name}_error"] = 0
        
        return features

    def _compute_contextual_features(self, df, signals, **kwargs):
        """Contextual Features Family"""
        features = pd.DataFrame(index=df.index)
        batch_id = kwargs.get('batch_id', 'batch_id')
        
        if batch_id in df.columns:
            batch_pos = df.groupby(batch_id).cumcount()
            features["batch_position"] = batch_pos / batch_pos.groupby(df[batch_id]).transform('max')
            features["is_batch_start"] = (batch_pos == 0).astype(int)
            features["is_batch_end"] = (batch_pos == batch_pos.groupby(df[batch_id]).transform('max')).astype(int)
        
        return features
    
    def _fix_rule_parentheses(self, expr: str) -> str:
        """Add parentheses around comparison operations to avoid ambiguous truth values."""
        normalized = self._normalize_rule_expr(expr)
        parts = re.split(r'(\s*[&|]\s*)', normalized)
        
        if len(parts) == 1:
            return normalized
        
        result_parts = []
        for part in parts:
            if part.strip() in ['&', '|']:
                result_parts.append(part)
            else:
                if any(op in part for op in ['>', '<', '==', '!=', '>=', '<=']):
                    result_parts.append(f'({part})')
                else:
                    result_parts.append(part)
        
        return ''.join(result_parts)

    def compute_features(self, df, feature_plan: Dict[str, List[str]]):
        """Compute features based on a feature plan"""
        all_features = pd.DataFrame(index=df.index)
        
        for family, signals in feature_plan.items():
            if family not in self.feature_families:
                print(f"Warning: Unknown feature family '{family}'")
                continue
            
            if family == 'interaction':
                for signal_pair in signals:
                    if len(signal_pair) == 2:
                        family_features = self.feature_families[family](df, signal_pair)
                        all_features = pd.concat([all_features, family_features], axis=1)
            else:
                family_features = self.feature_families[family](df, signals)
                all_features = pd.concat([all_features, family_features], axis=1)
        
        return all_features.fillna(0)
    
    def analyze_rule_performance(self, df: pd.DataFrame, feature_plan: Dict[str, List[str]]) -> Dict:
        """
        Compute features and analyze rule performance
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data with sensor signals and state labels
        feature_plan : dict
            Feature plan including event rules
            
        Returns:
        --------
        Dict with features and diagnostic results
        """
        # Compute features
        features = self.compute_features(df, feature_plan)
        
        # Extract event features for analysis
        event_features = features[[col for col in features.columns if col.startswith('event_')]]
        
        if event_features.empty:
            print("No event features found for analysis")
            return {'features': features, 'diagnostics': None}
        
        # Run diagnostic analysis
        diagnostic_results = self.diagnostic_analyzer.compute_rule_metrics(df, event_features)
        
        return {
            'features': features,
            'diagnostics': diagnostic_results
        }

# [Keep all your existing HMM functions here...]
# empirical_start_trans, emissions_from_labels, viterbi_decode, print_evaluation


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
    return model.predict(X_np, lengths)

def print_evaluation(y_true_idx, y_pred_idx, title=""):
    labs_true = [idx_to_state[i] for i in y_true_idx]
    labs_pred = [idx_to_state.get(i, f"UNK{i}") for i in y_pred_idx]
    print(f"\n== {title} ==")
    print(classification_report(labs_true, labs_pred, labels=state_list, zero_division=0))
    cm = confusion_matrix(labs_true, labs_pred, labels=state_list)
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=state_list, columns=state_list))

# Enhanced main section with diagnostic analysis
if __name__ == "__main__":
    CSV_PATH = "synthetic_pasteurization_with_cip_signals.csv"
    #CSV_PATH="SWat.csv"
    USE_CIP = False
    MODE = "unsupervised"
    ADD_DERIVS = True
    N_UNSUP = None
    RANDOM_SEED = 42

    states_prod = ["Idle","Fill","HeatUp","Hold","Cool","Discharge"]
    #states_prod = ["Filling", "Draining", "Hold"]
    states_cip = ["PreRinse","Caustic","InterRinse","Acid","FinalRinse","Sanitize","Verification","Standby"]

    state_list = states_prod + (states_cip if USE_CIP else [])
    n_states = len(state_list) if N_UNSUP is None else (N_UNSUP if MODE=="unsupervised" else len(state_list))

    # Load data
    df = pd.read_csv(CSV_PATH)
    df = df[df["state"].isin(state_list)].copy()
    df.sort_values(["batch_id","timestamp"], inplace=True)

    # Initialize feature library
    feature_lib = ModularFeatureLibrary()
    '''
    feature_plan = {
       'statistical' : ['FIT101', 'LIT101'],          
        'temporal': ['FIT101', 'LIT101'],                      
        'stability': ['FIT101', 'LIT101'],             
        'interaction': [['FIT101', 'LIT101']],         
        'event': [        ],                         
        'contextual': [] 
    }
    '''
    # Define feature plan with event rules
    
    feature_plan = {
        'statistical': ['T', 'Q_in','Q_out'],           
        'temporal': ['T', 'Q_in','Q_out'],                      
        'stability': ['T', 'Q_in','Q_out'],             
        'interaction': [['T', 'Q_in','Q_out']],         
        'event': [
            '(T_diff_smooth > 1)', '(T_diff_smooth < -1)',
            '(Q_out > 0.3)',

            #'(T < 20)',
            '(T > 70) & (T_stable_flag == 1)', #  
            '(Q_in > 0.3)  AND (T_diff < 0.2)'
            #'T > 75',  # Additional test rule
            #'Q_in < 0.2'  # Additional test rule
        ],                         
        'contextual': []                        
    }
    
    # Compute features and analyze rule performance
    result = feature_lib.analyze_rule_performance(df, feature_plan)
    all_features = result['features']
    diagnostics = result['diagnostics']
    
    print(f"Original data shape: {df.shape}")
    print(f"Computed features shape: {all_features.shape}")
    
    # Print diagnostic report
    if diagnostics:
        feature_lib.diagnostic_analyzer.print_diagnostic_report(diagnostics)
    
    # Continue with HMM training as before...
    all_features.to_csv("data.csv", index=False)
    #event_features = all_features[[col for col in all_features.columns if col.startswith('event_')]]
    #features = event_features

    important_raw_features = ['T_roll_mean_5',  'Q_in_roll_mean_5',  'Q_out_roll_mean_5', 'T_diff']
    event_features = all_features[[col for col in all_features.columns if col.startswith('event_')]]

    # Combine them
    features = pd.concat([all_features[important_raw_features], event_features], axis=1)
    # [Rest of your existing HMM code...]
    bids = df["batch_id"].unique()
    bids_train = set(bids[:max(1, int(0.6*len(bids)))])
    bids_test = set(bids) - bids_train

    def pack_sequences(df_subset, X_subset):
        lengths = df_subset.groupby("batch_id").size().tolist()
        return X_subset.values, lengths

    scaler = StandardScaler()
    X_train = features[df["batch_id"].isin(bids_train)]
    X_test = features[df["batch_id"].isin(bids_test)]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    df_train = df[df["batch_id"].isin(bids_train)]
    df_test = df[df["batch_id"].isin(bids_test)]

    X_train_np, lengths_train = pack_sequences(df_train, pd.DataFrame(X_train_scaled, index=X_train.index))
    X_test_np, lengths_test = pack_sequences(df_test, pd.DataFrame(X_test_scaled, index=X_test.index))

    state_to_idx = {s:i for i,s in enumerate(state_list)}
    idx_to_state = {i:s for s,i in state_to_idx.items()}

    # Your existing HMM code continues here...
    # ========= SUPERVISED HMM =========
    if MODE.lower() == "supervised":
        # Labeled indices (train/test)
        y_train_idx = df_train["state"].map(state_to_idx).values
        y_test_idx  = df_test["state"].map(state_to_idx).values

        # Initialize startprob/transmat from labels
        startprob_, transmat_ = empirical_start_trans(y_train_idx, lengths_train, n_states)

        # Initialize emissions from labels
        means_, covars_ = emissions_from_labels(X_train_np, y_train_idx, n_states)

        # Build and fit HMM (few EM iters to refine)
        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=30,
            init_params="",   # do not overwrite our inits
            random_state=RANDOM_SEED,
            tol=1e-3,
            verbose=False
        )
        hmm.startprob_ = startprob_
        hmm.transmat_  = transmat_
        hmm.means_     = means_
        hmm.covars_    = covars_

        hmm.fit(X_train_np, lengths_train)

        # Decode test and evaluate
        y_pred_test = viterbi_decode(hmm, X_test_np, lengths_test)
        print_evaluation(y_test_idx, y_pred_test, title="Supervised HMM (Test)")

    # ========= UNSUPERVISED HMM + label mapping =========
    else:
        # Unsupervised fit on TRAIN, then map discovered states to labels (Hungarian)
        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",  # BETTER FOR STATE SEPARATION
            n_iter=100,
            random_state=RANDOM_SEED,
            tol=1e-6,
            init_params="stmc",
            params="stmc"
        )
        hmm.fit(X_train_np, lengths_train)

        # Predict hidden labels on TRAIN to build contingency with ground truth
        y_train_true = df_train["state"].map(state_to_idx).values
        y_train_hat  = viterbi_decode(hmm, X_train_np, lengths_train)

        # Build contingency (true x pred)
        K = len(state_list)
        cont = np.zeros((K, K), dtype=int)
        for t, p in zip(y_train_true, y_train_hat):
            if t < K and p < K:
                cont[t, p] += 1

        # Optimal mapping: rows(true)->cols(pred)
        row_ind, col_ind = linear_sum_assignment(cont.max() - cont)
        mapping = {pred: true for true, pred in zip(row_ind, col_ind)}

        # Decode TEST, remap discovered states to labels
        y_test_hat = viterbi_decode(hmm, X_test_np, lengths_test)
        y_test_mapped = np.array([mapping.get(s, 0) for s in y_test_hat], dtype=int)

        y_test_true = df_test["state"].map(state_to_idx).values
        print_evaluation(y_test_true, y_test_mapped, title="Unsupervised HMM (mapped) — Test")


        #######################################################################################
        

        # ========= TIMESTAMP NORMALIZATION =========
        print("\n=== Normalizing Timestamps ===")

        def normalize_timestamps(df, timestamp_col="timestamp", case_id_col="batch_id", base_date="2023-01-01"):
            """
            Correctly normalize timestamps by handling different time units properly.
            """
            
            df_normalized = df.copy()
            
            # First, ensure we understand the timestamp format
            print(f"Original timestamp sample: {df[timestamp_col].iloc[:5].tolist()}")
            
            # Check if timestamps are numeric (seconds) or string/datetime
            if np.issubdtype(df[timestamp_col].dtype, np.number):
                print("Timestamps are numeric - assuming they represent seconds")
                # Convert numeric seconds to datetime
                base_datetime = pd.to_datetime(base_date)
                df_normalized[timestamp_col] = base_datetime + pd.to_timedelta(df[timestamp_col], unit='s')
            else:
                # Try to parse as datetime
                try:
                    df_normalized[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    print("Timestamps successfully parsed as datetime")
                except:
                    print("Could not parse timestamps. Please check the format.")
                    return df
            

            case_groups = df_normalized.groupby(case_id_col)
            
            for case_id, case_data in case_groups:
                case_start = case_data[timestamp_col].min()
                time_deltas = case_data[timestamp_col] - case_start
                df_normalized.loc[case_data.index, timestamp_col] = pd.to_datetime(base_date) + time_deltas
            
            return df_normalized


        df_normalized = normalize_timestamps(df, base_date="2023-01-01")


        def create_interval_event_log_normalized(df, y_pred, state_mapping, case_id_col="batch_id", timestamp_col="timestamp"):
            """
            Create interval-based event log using normalized timestamps
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


        X_test_full_np, lengths_test_full = pack_sequences(df_test, pd.DataFrame(X_test_scaled, index=X_test.index))
        y_test_full_hat = viterbi_decode(hmm, X_test_full_np, lengths_test_full)


        if MODE.lower() == "unsupervised":
            
            state_mapping = {pred: idx_to_state[true] for pred, true in mapping.items() if true in idx_to_state}
        else:
            
            state_mapping = idx_to_state
            

        df_test_normalized = df_normalized[df_normalized["batch_id"].isin(df_test["batch_id"].unique())]

        # Align lengths
        assert len(df_test_normalized) == len(y_test_full_hat), "Mismatch between test rows and predictions!"

        interval_event_log_normalized = create_interval_event_log_normalized(
            df_test_normalized, y_test_full_hat, state_mapping
        )


        # Save to CSV
        normalized_log_path = "pasteurization_normalized_event_log_MT.csv"
        interval_event_log_normalized.to_csv(normalized_log_path, index=False)
        print(f"Normalized event log saved to: {normalized_log_path}")

        # Show the beautiful result!
        print("\nSample of normalized event log:")
        print(interval_event_log_normalized.head(10))

        # ========= COMPARE BEFORE/AFTER =========
        print("\n=== Timestamp Normalization Comparison ===")


        sample_case = interval_event_log_normalized['case_id'].iloc[0]
        original_case_data = df[df['batch_id'] == sample_case].copy()
        normalized_case_data = df_normalized[df_normalized['batch_id'] == sample_case].copy()

        print(f"Sample Case: {sample_case}")
        print(f"Original start: {original_case_data['timestamp'].min()}")
        print(f"Normalized start: {normalized_case_data['timestamp'].min()}")
        print(f"Original duration: {(pd.to_datetime(original_case_data['timestamp'].max()) - pd.to_datetime(original_case_data['timestamp'].min())).total_seconds():.0f} seconds")
        print(f"Normalized duration: {(pd.to_datetime(normalized_case_data['timestamp'].max()) - pd.to_datetime(normalized_case_data['timestamp'].min())).total_seconds():.0f} seconds (same!)")


        print("\n=== Creating Enhanced Visualizations ===")


        plt.figure(figsize=(14, 8))


        activities = interval_event_log_normalized['activity'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(activities)))
        color_map = dict(zip(activities, colors))


        for i, case_id in enumerate(interval_event_log_normalized['case_id'].unique()[:10]):  # First 10 cases
            case_data = interval_event_log_normalized[interval_event_log_normalized['case_id'] == case_id]
            
            for _, activity_row in case_data.iterrows():
                start = pd.to_datetime(activity_row['start_timestamp'])
                end = pd.to_datetime(activity_row['end_timestamp'])
                duration = (end - start).total_seconds() / 3600  # Convert to hours for plotting
                
                plt.barh(y=i, width=duration, left=start, 
                        color=color_map[activity_row['activity']], 
                        edgecolor='black', alpha=0.7)
                
                # Add activity label for longer segments
                if duration > 0.1:  # Only label segments longer than 6 minutes
                    plt.text(start + pd.Timedelta(seconds=duration*3600/2), i, 
                            activity_row['activity'], ha='center', va='center', 
                            fontsize=8, fontweight='bold')

        plt.yticks(range(10), interval_event_log_normalized['case_id'].unique()[:10])
        plt.xlabel('Time (from normalized start)')
        plt.ylabel('Case ID')
        plt.title('Process Execution Gantt Chart (First 10 Cases)')
        plt.legend([plt.Rectangle((0,0),1,1, color=color_map[act]) for act in activities], 
                activities, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('process_gantt_chart.png', dpi=300, bbox_inches='tight')
        plt.show()


        def filter_brief_states(event_log, min_duration_seconds=5.0):
            """
            Remove state segments that are too brief by merging them with adjacent states
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
                            
                        else:  # Middle segment - merge with previous (you could choose which neighbor to merge with)
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

        # Apply the filter to your event log
        min_duration = 2.0  # Minimum duration in seconds (adjust as needed)
        filtered_event_log = filter_brief_states(interval_event_log_normalized, min_duration_seconds=min_duration)

        print(f"Original events: {len(interval_event_log_normalized)}")
        print(f"Filtered events: {len(interval_event_log_normalized)}")
        print("Removed", len(interval_event_log_normalized) - len(filtered_event_log), "brief state segments")

        filtered_log_path = "pasteurization_cleaned_event_log_MT.csv"
        filtered_event_log.to_csv(filtered_log_path, index=False)
        print(f"\nCleaned event log saved to: {filtered_log_path}")




