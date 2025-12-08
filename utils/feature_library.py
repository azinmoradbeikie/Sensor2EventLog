"""
Modular feature extraction library with diagnostic capabilities
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List
from .rule_analyzer import RuleDiagnosticAnalyzer


class ModularFeatureLibrary:
    """
    Modular feature extraction library supporting multiple feature families
    with integrated rule diagnostics.
    """
    
    def __init__(self, window_sizes=None, stability_eps=1, peak_threshold=0.1):
        self.window_sizes = window_sizes or [5]
        self.stability_eps = stability_eps
        self.peak_threshold = peak_threshold
        
        # Feature family implementations
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
        """Convert human-friendly logical ops to pandas-style bitwise ops."""
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
        """Evaluate a rule expression using available features."""
        normalized_expr = self._normalize_rule_expr(rule_expr)
        
        try:
            eval_env = {col: available_features[col] for col in available_features.columns}
            eval_env.update({
                'np': np, 'pd': pd, 'abs': np.abs, 
                'min': np.minimum, 'max': np.maximum
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
    
    def _safe_ratio(self, a, b):
        """Safe ratio calculation with log transformation."""
        a_safe = np.abs(a) + 1e-6
        b_safe = np.abs(b) + 1e-6
        ratio = np.log1p(a_safe) - np.log1p(b_safe)
        sign = np.sign(a * b)
        return ratio * sign
    
    def _compute_statistical_features(self, df, signals, **kwargs):
        """Statistical features: rolling means."""
        features = pd.DataFrame(index=df.index)
        
        for signal in signals:
            s = df[signal]
            for win in self.window_sizes:
                roll = s.rolling(win, min_periods=1)
                features[f"{signal}_roll_mean_{win}"] = roll.mean()
        
        return features
    
    def _compute_temporal_features(self, df, signals, **kwargs):
        """Temporal dynamics features: differences and rates."""
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
        """Stability features: stability flags and consecutive stable periods."""
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
        """Interaction features: products and ratios between signals."""
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
        """Event/regime features with rule-based definitions."""
        features = pd.DataFrame(index=df.index)
        
        # Create comprehensive set of available features
        available_features = df.copy()
        
        # Pre-compute derived features for all numeric columns
        for signal in df.columns:
            if pd.api.types.is_numeric_dtype(df[signal]):
                try:
                    diff = df[signal].diff().fillna(0)
                    available_features[f"{signal}_diff"] = diff
                    available_features[f"{signal}_diff_smooth"] = diff.ewm(span=5).mean()
                    available_features[f"{signal}_abs_diff"] = np.abs(diff)
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
        """Contextual features: batch position and boundaries."""
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
        """Compute features based on a feature plan."""
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
        Compute features and analyze rule performance.
        
        Parameters:
        -----------
        df : Input data with sensor signals and state labels
        feature_plan : Feature plan including event rules
        
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