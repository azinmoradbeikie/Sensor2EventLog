"""
Machine Teaching loop for Sensor2EventLog framework.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.hmm_model import HMMModel


class MachineTeachingLoop:
    """
    Orchestrates feature extraction, diagnostics, and model training.
    """

    def __init__(self, model_type: str, feature_extractor, diagnostic_analyzer, config):
        self.model_type = model_type
        self.feature_extractor = feature_extractor
        self.diagnostic_analyzer = diagnostic_analyzer
        self.config = config
        self._review_summary: Optional[Dict[str, Any]] = None

        if model_type != "hmm":
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.model = HMMModel(config=self.config)
        self._scaler = StandardScaler()

    def get_review_summary(self) -> Optional[Dict[str, Any]]:
        return self._review_summary

    def run(
        self,
        df: pd.DataFrame,
        feature_plan: Dict[str, list],
        mode: str = "unsupervised",
        n_unsup: Optional[int] = None,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        features, diagnostics = self._extract_features_and_diagnostics(df, feature_plan)

        X_train, X_test, df_train, df_test = self._split_train_test(df, features)
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        X_train_np, lengths_train = self._pack_sequences(df_train, X_train_scaled)
        X_test_np, lengths_test = self._pack_sequences(df_test, X_test_scaled)

        state_list, state_to_idx, idx_to_state = self._build_state_maps(df)
        y_train = df_train["state"].map(state_to_idx).values
        y_test = df_test["state"].map(state_to_idx).values

        if mode.lower() == "supervised":
            y_pred_test, model, state_mapping = self.model.train_supervised(
                X_train_np,
                lengths_train,
                X_test_np,
                lengths_test,
                y_train,
                y_test,
                state_list,
                idx_to_state,
            )
        else:
            y_pred_test, model, state_mapping = self.model.train_unsupervised(
                X_train_np,
                lengths_train,
                X_test_np,
                lengths_test,
                y_train,
                y_test,
                state_list,
                idx_to_state,
                n_unsup,
            )

        # Predict full sequence for event log generation
        X_full_scaled = self._scaler.transform(features)
        X_full_np, lengths_full = self._pack_sequences(df, X_full_scaled)
        y_pred_full = self.model.predict(X_full_np, lengths_full)

        self._review_summary = diagnostics

        return {
            "features": features,
            "diagnostics": diagnostics,
            "model": model,
            "predictions": y_pred_full,
            "state_mapping": state_mapping,
        }

    def _extract_features_and_diagnostics(
        self, df: pd.DataFrame, feature_plan: Dict[str, list]
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        if hasattr(self.feature_extractor, "analyze_rule_performance"):
            result = self.feature_extractor.analyze_rule_performance(df, feature_plan)
            return result["features"], result.get("diagnostics")

        features = self.feature_extractor.compute_features(df, feature_plan)
        event_cols = [c for c in features.columns if c.startswith("event_")]
        diagnostics = None
        if event_cols:
            diagnostics = self.diagnostic_analyzer.compute_rule_metrics(df, features[event_cols])
        return features, diagnostics

    def _split_train_test(
        self, df: pd.DataFrame, features: pd.DataFrame, train_ratio: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        batch_ids = df["batch_id"].unique()
        n_train = max(1, int(train_ratio * len(batch_ids)))
        train_batch_ids = set(batch_ids[:n_train])
        test_batch_ids = set(batch_ids[n_train:])

        df_train = df[df["batch_id"].isin(train_batch_ids)]
        df_test = df[df["batch_id"].isin(test_batch_ids)]

        X_train = features.loc[df_train.index].values
        X_test = features.loc[df_test.index].values

        return X_train, X_test, df_train, df_test

    @staticmethod
    def _pack_sequences(df_subset: pd.DataFrame, X_subset: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        lengths = df_subset.groupby("batch_id").size().tolist()
        return X_subset, lengths

    def _build_state_maps(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        states = df["state"].unique().tolist()
        ordered = []
        for key in ("production", "cip"):
            if hasattr(self.config, "PROCESS_STATES") and key in self.config.PROCESS_STATES:
                ordered.extend(self.config.PROCESS_STATES[key])
        state_list = [s for s in ordered if s in states] or sorted(states)
        state_to_idx = {s: i for i, s in enumerate(state_list)}
        idx_to_state = {i: s for s, i in state_to_idx.items()}
        return state_list, state_to_idx, idx_to_state
