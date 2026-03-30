"""
Event log object with PM4Py compatibility
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


class Event:
    """
    Single event in an event log.
    
    Attributes:
    -----------
    case_id : str
        Identifier for the process case
    activity : str
        Name of the activity/state
    start_time : datetime
        Start timestamp of the event
    end_time : datetime
        End timestamp of the event
    duration : float
        Duration in seconds
    """
    
    def __init__(self, case_id: str, activity: str, start_time: datetime, 
                 end_time: datetime, duration: float = None, **kwargs):
        self.case_id = str(case_id)
        self.activity = activity
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration or (end_time - start_time).total_seconds()
        self.attributes = kwargs
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary."""
        return {
            'case_id': self.case_id,
            'activity': self.activity,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            **self.attributes
        }


class EventLog:
    """
    Event log container with PM4Py compatibility.
    
    This class provides a standardized interface for event logs
    that can be exported to various formats (CSV, XES) and used
    with process mining tools like PM4Py.
    
    Example:
        >>> log = EventLog(df)
        >>> log.to_csv("event_log.csv")
        >>> log.to_xes("event_log.xes")
        >>> pm4py_log = log.to_pm4py()  # Use with PM4Py
    """
    
    def __init__(self, data: Union[pd.DataFrame, List[Event]]):
        """
        Initialize event log from DataFrame or list of Events.
        
        Parameters:
        -----------
        data : pd.DataFrame or List[Event]
            Input event log data
        """
        if isinstance(data, pd.DataFrame):
            self._df = self._validate_dataframe(data)
        elif isinstance(data, list):
            self._df = self._from_events(data)
        else:
            raise ValueError("Data must be DataFrame or list of Events")
        
        self._pm4py_log = None
    
    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and standardize DataFrame format."""
        required_cols = ['case_id', 'activity', 'start_timestamp', 'end_timestamp']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")
        
        # Ensure timestamp columns are datetime
        for col in ['start_timestamp', 'end_timestamp']:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        
        # Add duration if missing
        if 'duration_seconds' not in df.columns:
            df['duration_seconds'] = (
                pd.to_datetime(df['end_timestamp']) - 
                pd.to_datetime(df['start_timestamp'])
            ).dt.total_seconds()
        
        return df
    
    def _from_events(self, events: List[Event]) -> pd.DataFrame:
        """Convert list of Events to DataFrame."""
        return pd.DataFrame([e.to_dict() for e in events])
    
    def to_dataframe(self) -> pd.DataFrame:
        """Get event log as pandas DataFrame."""
        return self._df.copy()
    
    def to_csv(self, path: str, filtered: bool = False) -> None:
        """
        Export event log to CSV.
        
        Parameters:
        -----------
        path : str
            Output file path
        filtered : bool
            If True, saves the filtered version (if available)
        """
        df_to_save = self._df
        if filtered and 'filtered' in self._df.columns:
            df_to_save = self._df[self._df['filtered'] == True]
        
        df_to_save.to_csv(path, index=False)
        print(f"Event log saved to: {path}")
    
    def to_xes(self, path: str, case_id_key: str = 'case:concept:name',
               timestamp_key: str = 'time:timestamp') -> None:
        """
        Export event log to XES format using PM4Py.
        
        Parameters:
        -----------
        path : str
            Output file path
        case_id_key : str
            Column name to use as case identifier in XES
        timestamp_key : str
            Column name to use as timestamp in XES
        """
        try:
            import pm4py
        except ImportError:
            raise ImportError("PM4Py is required for XES export. Install with: pip install pm4py")
        
        # Convert to PM4Py format
        pm4py_log = self.to_pm4py(case_id_key, timestamp_key)
        
        # Export to XES
        pm4py.write_xes(pm4py_log, path)
        print(f"Event log exported to XES: {path}")
    
    def to_pm4py(self, case_id_key: str = 'case:concept:name',
                 timestamp_key: str = 'time:timestamp') -> 'pm4py.objects.log.obj.EventLog':
        """
        Convert to PM4Py EventLog object for further analysis.
        
        Parameters:
        -----------
        case_id_key : str
            Column name to use as case identifier
        timestamp_key : str
            Column name to use as timestamp
            
        Returns:
        --------
        pm4py.objects.log.obj.EventLog
            PM4Py event log object
        """
        try:
            import pm4py
        except ImportError:
            raise ImportError("PM4Py is required for this functionality. Install with: pip install pm4py")
        
        # Prepare data for PM4Py format
        df_for_pm4py = self._df.copy()
        
        # Rename columns for PM4Py
        df_for_pm4py = df_for_pm4py.rename(columns={
            'case_id': case_id_key,
            'activity': 'concept:name',
            'start_timestamp': timestamp_key
        })
        
        # Add end timestamp as separate attribute if available
        if 'end_timestamp' in df_for_pm4py.columns:
            df_for_pm4py['end_timestamp'] = df_for_pm4py['end_timestamp'].astype(str)
        
        # Convert to PM4Py event log
        event_log = pm4py.format_dataframe_to_event_log(
            df_for_pm4py,
            case_id=case_id_key,
            activity_key='concept:name',
            timestamp_key=timestamp_key
        )
        
        self._pm4py_log = event_log
        return event_log
    
    def filter_duration(self, min_seconds: float = 0, max_seconds: float = float('inf')) -> 'EventLog':
        """
        Filter events by duration.
        
        Parameters:
        -----------
        min_seconds : float
            Minimum duration in seconds
        max_seconds : float
            Maximum duration in seconds
            
        Returns:
        --------
        EventLog
            Filtered event log
        """
        filtered_df = self._df[
            (self._df['duration_seconds'] >= min_seconds) &
            (self._df['duration_seconds'] <= max_seconds)
        ].copy()
        filtered_df['filtered'] = True
        
        return EventLog(filtered_df)
    
    def get_cases(self) -> List[str]:
        """Get list of unique case IDs."""
        return self._df['case_id'].unique().tolist()
    
    def get_activities(self) -> List[str]:
        """Get list of unique activities."""
        return self._df['activity'].unique().tolist()
    
    def get_case(self, case_id: str) -> 'EventLog':
        """Get all events for a specific case."""
        case_df = self._df[self._df['case_id'] == str(case_id)].copy()
        return EventLog(case_df)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute basic statistics about the event log.
        
        Returns:
        --------
        dict with:
            - total_cases: number of cases
            - total_events: number of events
            - unique_activities: number of distinct activities
            - avg_case_duration: average case duration in seconds
            - activity_frequencies: frequency of each activity
        """
        stats = {
            'total_cases': self._df['case_id'].nunique(),
            'total_events': len(self._df),
            'unique_activities': self._df['activity'].nunique(),
            'avg_case_duration': self._df.groupby('case_id')['duration_seconds'].sum().mean(),
            'activity_frequencies': self._df['activity'].value_counts().to_dict()
        }
        return stats
    
    def __len__(self) -> int:
        """Return number of events."""
        return len(self._df)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"EventLog(cases={self.get_statistics()['total_cases']}, events={len(self._df)}, activities={self.get_statistics()['unique_activities']})"
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Return first n events."""
        return self._df.head(n)


def create_interval_event_log_normalized(df, y_pred, state_mapping, 
                                        case_id_col="batch_id", timestamp_col="timestamp"):
    """
    Create interval-based event log using normalized timestamps.
    
    This function is kept for backward compatibility.
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