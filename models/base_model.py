"""
Base model interface for pluggable models
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models in Sensor2EventLog.
    
    This interface ensures that all models can be used interchangeably
    in the Machine Teaching loop.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, lengths: List[int], y: Optional[np.ndarray] = None) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        lengths : List[int]
            Lengths of each sequence
        y : np.ndarray, optional
            Labels for supervised learning
            
        Returns:
        --------
        self : BaseModel
            Fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, lengths: List[int]) -> np.ndarray:
        """
        Predict states for new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        lengths : List[int]
            Lengths of each sequence
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted state indices (n_samples,)
        """
        pass
    
    @abstractmethod
    def get_state_mapping(self) -> Dict[int, str]:
        """
        Get mapping from state indices to state names.
        
        Returns:
        --------
        Dict[int, str]
            Mapping from index to state name
        """
        pass