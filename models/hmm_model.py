"""
Hidden Markov Model implementation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment

from models.base_model import BaseModel
from utils.hmm_utils import (
    empirical_start_trans, emissions_from_labels, viterbi_decode, print_evaluation
)


class HMMModel(BaseModel):
    """
    Gaussian Hidden Markov Model for process state discovery.
    
    Supports both supervised and unsupervised learning modes.
    """
    
    def __init__(self, config=None):
        """
        Initialize HMM model.
        
        Parameters:
        -----------
        config : module
            Configuration module with HMM parameters
        """
        self.config = config
        self.model = None
        self._state_mapping = None
        self._idx_to_state = None
        self._state_list = None
        
        # Default HMM parameters
        self.covariance_type = getattr(config, 'HMM_CONFIG', {}).get("covariance_type", "diag")
        self.n_iter = getattr(config, 'HMM_CONFIG', {}).get("n_iter", 100)
        self.random_seed = getattr(config, 'HMM_CONFIG', {}).get("random_seed", 42)
        self.tol = getattr(config, 'HMM_CONFIG', {}).get("tol", 1e-6)
    
    def fit(self, X: np.ndarray, lengths: List[int], y: Optional[np.ndarray] = None) -> 'HMMModel':
        """
        Fit HMM to data.
        
        If y is provided, uses supervised initialization.
        Otherwise, uses unsupervised learning.
        """
        if y is not None:
            # Supervised initialization
            n_states = len(np.unique(y))
            startprob, transmat = empirical_start_trans(y, lengths, n_states)
            means, covars = emissions_from_labels(X, y, n_states)
            
            self.model = GaussianHMM(
                n_components=n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                init_params="",
                random_state=self.random_seed,
                tol=self.tol
            )
            self.model.startprob_ = startprob
            self.model.transmat_ = transmat
            self.model.means_ = means
            self.model.covars_ = covars
            
            self.model.fit(X, lengths)
        else:
            # Unsupervised learning
            # n_components must be set externally
            pass
        
        return self
    
    def predict(self, X: np.ndarray, lengths: List[int]) -> np.ndarray:
        """Predict state sequence using Viterbi algorithm."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return viterbi_decode(self.model, X, lengths)
    
    def get_state_mapping(self) -> Dict[int, str]:
        """Get mapping from state indices to state names."""
        return self._state_mapping or {}
    
    def train_supervised(self, X_train, lengths_train, X_test, lengths_test,
                        y_train, y_test, state_list, idx_to_state) -> Tuple:
        """Train supervised HMM with labeled data."""
        print("\nTraining supervised HMM...")
        
        n_states = len(state_list)
        self._state_list = state_list
        self._idx_to_state = idx_to_state
        
        # Initialize from labels
        startprob, transmat = empirical_start_trans(y_train, lengths_train, n_states)
        means, covars = emissions_from_labels(X_train, y_train, n_states)
        
        # Build and fit HMM
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            init_params="",
            random_state=self.random_seed,
            tol=self.tol
        )
        self.model.startprob_ = startprob
        self.model.transmat_ = transmat
        self.model.means_ = means
        self.model.covars_ = covars
        
        self.model.fit(X_train, lengths_train)
        
        # Decode and evaluate
        y_pred_test = self.predict(X_test, lengths_test)
        print_evaluation(y_test, y_pred_test, idx_to_state, state_list, 
                        title="Supervised HMM (Test)")
        
        # Create state mapping (1:1 for supervised)
        state_mapping = {i: state_list[i] for i in range(n_states)}
        self._state_mapping = state_mapping
        
        return y_pred_test, self.model, state_mapping
    
    def train_unsupervised(self, X_train, lengths_train, X_test, lengths_test,
                          y_train, y_test, state_list, idx_to_state, n_unsup) -> Tuple:
        """Train unsupervised HMM with state mapping."""
        print("\nTraining unsupervised HMM...")
        
        n_states = n_unsup if n_unsup is not None else len(state_list)
        self._state_list = state_list
        self._idx_to_state = idx_to_state
        
        # Train unsupervised HMM
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_seed,
            tol=self.tol,
            init_params="stmc",
            params="stmc"
        )
        self.model.fit(X_train, lengths_train)
        
        # Predict and map states
        y_train_hat = self.predict(X_train, lengths_train)
        
        # Build contingency matrix for mapping
        K = len(state_list)
        cont = np.zeros((K, K), dtype=int)
        for t, p in zip(y_train, y_train_hat):
            if t < K and p < K:
                cont[t, p] += 1
        
        # Optimal mapping using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cont.max() - cont)
        mapping = {pred: true for true, pred in zip(row_ind, col_ind)}
        
        # Decode test set and map states
        y_test_hat = self.predict(X_test, lengths_test)
        y_test_mapped = np.array([mapping.get(s, 0) for s in y_test_hat], dtype=int)
        
        print_evaluation(y_test, y_test_mapped, idx_to_state, state_list,
                        title="Unsupervised HMM (mapped) — Test")
        
        # Create full mapping for event log
        state_mapping = {pred: idx_to_state[true] for pred, true in mapping.items() if true in idx_to_state}
        self._state_mapping = state_mapping
        
        return y_test_hat, self.model, state_mapping