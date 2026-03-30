"""
Core pipeline orchestrator for Sensor2EventLog framework
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, Union

from abstraction.mt_loop import MachineTeachingLoop
from contextualization.event_log import EventLog
from features.feature_library import ModularFeatureLibrary
from evaluation.rule_analyzer import RuleDiagnosticAnalyzer
import config


class Sensor2EventLogPipeline:
    """
    Main pipeline class for transforming sensor data to event logs.
    
    This class orchestrates the entire Machine Teaching process:
    1. Feature extraction
    2. Model training (HMM with supervised/unsupervised modes)
    3. Diagnostic analysis
    4. Event log generation
    
    Example:
        >>> pipeline = Sensor2EventLogPipeline(config)
        >>> result = pipeline.run(
        ...     data_path="sensor_data.csv",
        ...     feature_plan=feature_plan,
        ...     mode="unsupervised"
        ... )
        >>> event_log = result['event_log']
        >>> event_log.to_xes("output.xes")
    """
    
    def __init__(self, config_module=None):
        """
        Initialize the pipeline with configuration.
        
        Parameters:
        -----------
        config_module : module, optional
            Configuration module with all parameters. If None, uses default config
        """
        self.config = config_module or config
        
        # Initialize components
        self.feature_library = ModularFeatureLibrary(
            window_sizes=self.config.FEATURE_CONFIG["window_sizes"],
            stability_eps=self.config.FEATURE_CONFIG["stability_eps"],
            peak_threshold=self.config.FEATURE_CONFIG["peak_threshold"]
        )
        
        self.diagnostic_analyzer = RuleDiagnosticAnalyzer(
            coverage_threshold=self.config.DIAGNOSTIC_CONFIG["coverage_threshold"],
            precision_threshold=self.config.DIAGNOSTIC_CONFIG["precision_threshold"],
            explainability_threshold=self.config.DIAGNOSTIC_CONFIG["explainability_threshold"]
        )
        
        self.mt_loop = MachineTeachingLoop(
            model_type="hmm",
            feature_extractor=self.feature_library,
            diagnostic_analyzer=self.diagnostic_analyzer,
            config=self.config
        )
        
        self._scaler = StandardScaler()
        self._fitted = False
    
    def run(self, 
            data_path: str, 
            feature_plan: Dict[str, list],
            mode: str = "unsupervised",
            use_cip: bool = False,
            n_unsup: Optional[int] = None,
            random_seed: int = 42,
            min_duration_seconds: float = 2.0,
            return_intermediate: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to CSV data file
        feature_plan : dict
            Feature extraction plan with families and signals
        mode : str
            "supervised" or "unsupervised"
        use_cip : bool
            Whether to include CIP states
        n_unsup : int
            Number of states for unsupervised mode
        random_seed : int
            Random seed for reproducibility
        min_duration_seconds : float
            Minimum duration for filtering brief states
        return_intermediate : bool
            If True, returns intermediate results (features, diagnostics)
            
        Returns:
        --------
        dict with keys:
            - event_log: EventLog object
            - model: trained HMM model
            - predictions: predicted state sequences
            - features (if return_intermediate): extracted features
            - diagnostics (if return_intermediate): diagnostic results
        """
        # Load and prepare data
        df = self._load_and_prepare_data(data_path, use_cip)
        
        # Run Machine Teaching loop
        mt_result = self.mt_loop.run(
            df=df,
            feature_plan=feature_plan,
            mode=mode,
            n_unsup=n_unsup,
            random_seed=random_seed
        )
        
        features = mt_result['features']
        model = mt_result['model']
        predictions = mt_result['predictions']
        diagnostics = mt_result['diagnostics']
        state_mapping = mt_result['state_mapping']
        
        # Generate event log
        event_log = self._generate_event_log(
            df, predictions, state_mapping, min_duration_seconds
        )
        
        result = {
            'event_log': event_log,
            'model': model,
            'predictions': predictions
        }
        
        if return_intermediate:
            result['features'] = features
            result['diagnostics'] = diagnostics
        
        return result
    
    def _load_and_prepare_data(self, data_path: str, use_cip: bool) -> pd.DataFrame:
        """Load and prepare data for analysis."""
        df = pd.read_csv(data_path)
        
        # Determine state list
        state_list = self.config.PROCESS_STATES["production"].copy()
        if use_cip:
            state_list += self.config.PROCESS_STATES["cip"]
        
        # Filter to relevant states and sort
        df = df[df["state"].isin(state_list)].copy()
        df.sort_values(["batch_id", "timestamp"], inplace=True)
        
        return df
    
    def _generate_event_log(self, df: pd.DataFrame, predictions: np.ndarray, 
                           state_mapping: Dict, min_duration_seconds: float) -> 'EventLog':
        """Generate event log from predictions."""
        from contextualization.event_log import create_interval_event_log_normalized
        
        # Normalize timestamps
        from utils.hmm_utils import normalize_timestamps
        df_normalized = normalize_timestamps(df)
        
        # Create event log
        event_df = create_interval_event_log_normalized(
            df_normalized, predictions, state_mapping
        )
        
        # Filter brief states
        from utils.hmm_utils import filter_brief_states
        filtered_df = filter_brief_states(event_df, min_duration_seconds)
        
        # Wrap in EventLog object
        event_log = EventLog(filtered_df)
        
        # Save if paths are configured
        if hasattr(self.config, 'PATHS'):
            event_log.to_csv(self.config.PATHS.get("event_log", "event_log.csv"))
            event_log.to_csv(self.config.PATHS.get("filtered_log", "filtered_log.csv"), 
                            filtered=True)
        
        return event_log