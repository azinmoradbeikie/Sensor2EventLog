"""
Rule diagnostic analyzer for evaluating rule performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
import re


class RuleDiagnosticAnalyzer:
    """
    Analyzes rule performance using coverage, precision, and explainability metrics.
    """
    
    def __init__(self, coverage_threshold: float = 0.6, precision_threshold: float = 0.7, 
                 explainability_threshold: float = 0.3):
        self.c_low = coverage_threshold
        self.p_low = precision_threshold
        self.epsilon_unex = explainability_threshold
    
    def compute_rule_metrics(self, df: pd.DataFrame, event_features: pd.DataFrame, 
                           state_column: str = 'state') -> Dict:
        """
        Compute coverage, precision, and explainability metrics for all event features.
        
        Parameters:
        -----------
        df : DataFrame with state labels
        event_features : DataFrame containing event rule features (binary columns)
        state_column : Column name containing state labels
            
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
        """Generate actionable recommendations based on diagnostic metrics."""
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
        """Print comprehensive diagnostic report."""
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