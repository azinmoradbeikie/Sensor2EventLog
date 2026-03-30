"""
Main entry point for Sensor2EventLog framework
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import Sensor2EventLogPipeline
import config


def main():
    """Main entry point for the framework."""
    
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
    
    # Initialize pipeline
    pipeline = Sensor2EventLogPipeline(config)
    
    # Run analysis
    results = pipeline.run(
        data_path="synthetic_pasteurization_with_cip_signals.csv",
        feature_plan=feature_plan,
        mode="unsupervised",
        use_cip=False,
        n_unsup=None,
        random_seed=42,
        return_intermediate=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    
    print(f"\nEvent Log: {len(results['event_log'])} events")
    print(f"Cases: {results['event_log'].get_statistics()['total_cases']}")
    print(f"Activities: {results['event_log'].get_activities()}")
    
    # Show review summary
    review_summary = pipeline.mt_loop.get_review_summary()
    if review_summary and 'recommendations' in review_summary:
        print(f"\nRecommendations: {len(review_summary['recommendations'])}")
        for rec in review_summary['recommendations'][:3]:
            print(f"  - {rec['action'][:100]}...")
    
    # Export to XES if PM4Py is available
    try:
        results['event_log'].to_xes("output_event_log.xes")
    except ImportError:
        print("\nPM4Py not installed. Skipping XES export.")
    
    return results


if __name__ == "__main__":
    results = main()