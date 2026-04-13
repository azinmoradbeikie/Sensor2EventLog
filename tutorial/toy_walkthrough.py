"""
Self-contained tutorial for Sensor2EventLog.

This script generates a small synthetic sensor dataset, extracts features,
evaluates rule coverage/precision, and converts state sequences into a simple
interval event log.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import types

import config as base_config
from core.pipeline import Sensor2EventLogPipeline


OUTPUT_DIR = Path(__file__).resolve().parent / "output"

STATE_ORDER = ["Idle", "Fill", "HeatUp", "Hold", "Cool", "Discharge"]
STATE_DURATIONS = {
    "Idle": 3,
    "Fill": 5,
    "HeatUp": 5,
    "Hold": 4,
    "Cool": 4,
    "Discharge": 3,
}


def _state_profile(state: str, step: int, total_steps: int, batch_offset: float) -> dict[str, float]:
    """Create simple, interpretable sensor patterns for one process state."""
    progress = step / max(total_steps - 1, 1)

    if state == "Idle":
        return {"T": 24 + batch_offset, "Q_in": 0.0, "Q_out": 0.0}
    if state == "Fill":
        return {"T": 24 + batch_offset, "Q_in": 0.75, "Q_out": 0.02}
    if state == "HeatUp":
        return {
            "T": 30 + 42 * progress + batch_offset,
            "Q_in": 0.35,
            "Q_out": 0.0,
        }
    if state == "Hold":
        return {"T": 73 + batch_offset, "Q_in": 0.08, "Q_out": 0.03}
    if state == "Cool":
        return {
            "T": 73 - 33 * progress + batch_offset,
            "Q_in": 0.05,
            "Q_out": 0.18,
        }
    if state == "Discharge":
        return {"T": 37 + batch_offset, "Q_in": 0.0, "Q_out": 0.72}
    raise ValueError(f"Unsupported state: {state}")


def make_toy_dataset(n_batches: int = 3, random_seed: int = 7) -> pd.DataFrame:
    """Generate a tiny labeled dataset with batch-wise sensor traces."""
    rng = np.random.default_rng(random_seed)
    rows = []

    for batch_num in range(1, n_batches + 1):
        batch_id = f"batch_{batch_num:02d}"
        batch_offset = 0.4 * (batch_num - 1)
        timestamp = 0

        for state in STATE_ORDER:
            duration = STATE_DURATIONS[state]
            for step in range(duration):
                profile = _state_profile(state, step, duration, batch_offset)
                rows.append(
                    {
                        "batch_id": batch_id,
                        "timestamp": timestamp,
                        "state": state,
                        "T": round(profile["T"] + rng.normal(0, 0.45), 3),
                        "Q_in": max(0.0, round(profile["Q_in"] + rng.normal(0, 0.03), 3)),
                        "Q_out": max(0.0, round(profile["Q_out"] + rng.normal(0, 0.03), 3)),
                    }
                )
                timestamp += 60

    return pd.DataFrame(rows)


def build_feature_plan() -> dict[str, list]:
    """Define a small, readable set of tutorial features."""
    return {
        "statistical": ["T", "Q_in", "Q_out"],
        "temporal": ["T", "Q_in", "Q_out"],
        "stability": ["T"],
        "interaction": [["T", "Q_in"], ["T", "Q_out"]],
        "event": [
            {"fill_rule": "(Q_in > 0.5) & (Q_out < 0.1)"},
            {"hold_rule": "(T > 70) & (T_stable_flag == 1)"},
            {"discharge_rule": "(Q_out > 0.5) & (Q_in < 0.1)"},
        ],
        "contextual": ["batch_position"],
    }


def build_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """Create a toy event log directly from the labeled state sequence."""
    state_to_idx = {state: idx for idx, state in enumerate(STATE_ORDER)}
    state_mapping = {idx: state for state, idx in state_to_idx.items()}
    y_true = df["state"].map(state_to_idx).to_numpy()

    rows = []
    for batch_id, batch_df in df.assign(predicted_state=y_true).groupby("batch_id"):
        batch_df = batch_df.sort_values("timestamp").copy()
        batch_df["activity"] = batch_df["predicted_state"].map(state_mapping)

        current_activity = None
        segment_start = None
        segment_end = None
        event_count = 0

        for row in batch_df.itertuples(index=False):
            if current_activity is None:
                current_activity = row.activity
                segment_start = row.timestamp
                segment_end = row.timestamp
                event_count = 1
                continue

            if row.activity == current_activity:
                segment_end = row.timestamp
                event_count += 1
                continue

            rows.append(
                {
                    "case_id": batch_id,
                    "activity": current_activity,
                    "start_timestamp": segment_start,
                    "end_timestamp": segment_end,
                    "duration_seconds": segment_end - segment_start,
                    "event_count": event_count,
                }
            )
            current_activity = row.activity
            segment_start = row.timestamp
            segment_end = row.timestamp
            event_count = 1

        rows.append(
            {
                "case_id": batch_id,
                "activity": current_activity,
                "start_timestamp": segment_start,
                "end_timestamp": segment_end,
                "duration_seconds": segment_end - segment_start,
                "event_count": event_count,
            }
        )

    event_log = pd.DataFrame(rows)
    event_log["activity_sequence"] = event_log.groupby("case_id").cumcount() + 1
    return event_log[
        [
            "case_id",
            "activity_sequence",
            "activity",
            "start_timestamp",
            "end_timestamp",
            "duration_seconds",
            "event_count",
        ]
    ]


def plot_time_series(df: pd.DataFrame, batch_id: str, feature_cols: list[str]) -> None:
    """Plot selected time series for a single batch."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping time-series plots.")
        return

    batch_df = df[df["batch_id"] == batch_id].copy()
    batch_df = batch_df.sort_values("timestamp")

    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(10, 6), sharex=True)
    if len(feature_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, feature_cols):
        ax.plot(batch_df["timestamp"], batch_df[col], label=col, linewidth=1.5)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestamp (s)")
    fig.suptitle(f"Toy batch {batch_id} sensor traces")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def compute_dfg(event_log: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple Directly-Follows Graph (DFG) from the event log."""
    transitions = []
    for case_id, case_df in event_log.groupby("case_id"):
        ordered = case_df.sort_values("activity_sequence")
        activities = ordered["activity"].tolist()
        for a, b in zip(activities, activities[1:]):
            transitions.append((a, b))

    dfg = (
        pd.DataFrame(transitions, columns=["source", "target"])
        .value_counts()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return dfg


def plot_dfg(dfg: pd.DataFrame) -> None:
    """Plot a simple DFG with arrows sized by frequency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping DFG plot.")
        return

    if dfg.empty:
        print("DFG is empty; nothing to plot.")
        return

    nodes = pd.unique(dfg[["source", "target"]].values.ravel("K")).tolist()
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis("off")

    max_count = dfg["count"].max()
    for _, row in dfg.iterrows():
        x0, y0 = positions[row["source"]]
        x1, y1 = positions[row["target"]]
        width = 0.5 + 2.5 * (row["count"] / max_count)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", linewidth=width, alpha=0.7),
        )
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mid_x, mid_y, str(row["count"]), fontsize=10, ha="center", va="center")

    for node, (x, y) in positions.items():
        ax.scatter([x], [y], s=800, color="#dddddd", edgecolor="#333333")
        ax.text(x, y, node, ha="center", va="center", fontsize=10)

    ax.set_title("Directly-Follows Graph (DFG)")
    plt.show()


def plot_process_graph(dfg: pd.DataFrame, node_order: list[str]) -> None:
    """Plot the DFG as a linear left-to-right process graph."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("matplotlib is not installed; skipping process graph plot.")
        return

    if dfg.empty:
        print("DFG is empty; nothing to plot.")
        return

    nodes = node_order or pd.unique(dfg[["source", "target"]].values.ravel("K")).tolist()
    n = len(nodes)
    step = 2.8
    x_positions = {node: i * step for i, node in enumerate(nodes)}
    y_positions = {node: 0 for node in nodes}
    forward_y = 0.0
    backward_y = 0.35
    label_y = 0.18

    fig, ax = plt.subplots(figsize=(max(12, n * 2.8), 3.5))
    ax.axis("off")

    # Linear process band from start to end with light opacity.
    band_x = [-0.6, (n - 1) * step + 0.6]
    ax.fill_between(band_x, -0.22, 0.22, color="#cfd8dc", alpha=0.3, zorder=0)

    max_count = dfg["count"].max()
    for _, row in dfg.iterrows():
        src = row["source"]
        tgt = row["target"]
        x0, y0 = x_positions[src], y_positions[src]
        x1, y1 = x_positions[tgt], y_positions[tgt]
        strength = row["count"] / max_count
        width = 0.8 + 2.5 * strength
        alpha = 0.25 + 0.75 * strength
        box_half = 0.55
        is_forward = x0 <= x1
        y = forward_y if is_forward else backward_y
        ax.annotate(
            "",
            xy=(x1 - box_half, y),
            xytext=(x0 + box_half, y),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=width,
                alpha=alpha,
            ),
            zorder=2,
        )
        label_y_pos = label_y if is_forward else backward_y + 0.12
        ax.text((x0 + x1) / 2, label_y_pos, str(row["count"]), ha="center", va="bottom", fontsize=9, zorder=4)

    for node in nodes:
        x, y = x_positions[node], y_positions[node]
        box = FancyBboxPatch(
            (x - 0.55, y - 0.18),
            1.1,
            0.36,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor="#f2f2f2",
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, node, ha="center", va="center", fontsize=10)

    ax.set_xlim(-0.9, (n - 1) * step + 0.9)
    ax.set_ylim(-0.7, 0.9)
    ax.set_title("Process Graph (DFG)")
    plt.show()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = make_toy_dataset()
    feature_plan = build_feature_plan()
    dataset_path = OUTPUT_DIR / "toy_sensor_data.csv"
    df.to_csv(dataset_path, index=False)

    cfg = types.SimpleNamespace(
        **{k: v for k, v in base_config.__dict__.items() if k.isupper()}
    )
    cfg.PATHS = {
        "event_log": str(OUTPUT_DIR / "toy_event_log.csv"),
        "filtered_log": str(OUTPUT_DIR / "toy_event_log_filtered.csv"),
    }

    pipeline = Sensor2EventLogPipeline(cfg)
    results = pipeline.run(
        data_path=str(dataset_path),
        feature_plan=feature_plan,
        mode="unsupervised",
        use_cip=False,
        n_unsup=None,
        random_seed=42,
        return_intermediate=True,
        min_duration_seconds=0.0,
    )

    features = results["features"]
    diagnostics = results["diagnostics"]
    event_log = results["event_log"].to_dataframe()
    dfg = compute_dfg(event_log)

    features_path = OUTPUT_DIR / "toy_features.csv"
    event_log_path = OUTPUT_DIR / "toy_event_log.csv"
    features.to_csv(features_path, index=False)
    features.to_csv(features_path, index=False)
    event_log.to_csv(event_log_path, index=False)

    print("Toy dataset created:")
    print(df.head(10).to_string(index=False))
    print("\nEvent feature columns:")
    event_cols = [col for col in features.columns if col.startswith("event_")]
    print(event_cols)
    print("\nToy event log:")
    print(event_log.head(12).to_string(index=False))

    if diagnostics:
        print("\nTop recommendations:")
        for rec in diagnostics["recommendations"][:5]:
            print(f"- [{rec['priority']}] {rec['type']}: {rec['action']}")

    plot_time_series(df, batch_id="batch_01", feature_cols=["T", "Q_in", "Q_out"])
    plot_process_graph(dfg, node_order=STATE_ORDER)

    print(f"\nSaved tutorial artifacts to {OUTPUT_DIR}")
    print(f"- dataset: {dataset_path}")
    print(f"- features: {features_path}")
    print(f"- event log: {event_log_path}")


if __name__ == "__main__":
    main()
