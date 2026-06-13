"""Pareto-front and run-summary helpers for batch-correction experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Return True for non-dominated rows in a minimization problem.

    Parameters
    ----------
    values:
        Array of shape (n_points, n_objectives). Lower is better for every
        objective. NaNs/inf rows are treated as dominated.
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    mask = np.ones(n, dtype=bool)
    finite = np.all(np.isfinite(values), axis=1)
    mask[~finite] = False
    for i in range(n):
        if not mask[i]:
            continue
        # j dominates i if j is <= i on every objective and < i on at least one.
        dominates_i = np.all(values[finite] <= values[i], axis=1) & np.any(values[finite] < values[i], axis=1)
        if np.any(dominates_i):
            mask[i] = False
    return mask


def optuna_trials_to_frame(study) -> pd.DataFrame:
    """Convert an Optuna study into a flat DataFrame with params/user_attrs."""
    rows = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "state": str(t.state).split(".")[-1],
        }
        if t.values is not None:
            for i, v in enumerate(t.values):
                row[f"value_{i}"] = v
        elif t.value is not None:
            row["value_0"] = t.value
        for k, v in t.params.items():
            row[f"param_{k}"] = v
        for k, v in t.user_attrs.items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def add_pareto_columns(
    df: pd.DataFrame,
    objectives: Iterable[str] = ("batch_bacc", "bio_loss"),
) -> pd.DataFrame:
    """Add an is_pareto column using the requested objective columns.

    By default this computes the scientific batch/biology tradeoff front only.
    Distortion/RMSE should usually be shown as color or used as a selection
    constraint, not minimized as a Pareto objective, because no correction has
    near-zero distortion and can otherwise pollute the front.
    """
    out = df.copy()
    complete = out.get("state", pd.Series(["COMPLETE"] * len(out))).astype(str).str.contains("COMPLETE")
    cols = [c for c in objectives if c in out.columns]
    out["is_pareto"] = False
    if cols and complete.any():
        vals = out.loc[complete, cols].to_numpy(dtype=float)
        out.loc[complete, "is_pareto"] = pareto_mask(vals)
    return out


def select_recommended_trial(
    df: pd.DataFrame,
    min_bio_bacc: float = 0.98,
    max_rmse_change: float | None = None,
) -> pd.Series:
    """Choose a practical recommended trial from a Pareto/multi-objective run.

    Preference order:
    1. Complete trials satisfying bio_bacc >= min_bio_bacc and optional RMSE cap.
    2. Among eligible trials, choose lowest batch_bacc.
    3. If none are eligible, choose lexicographically by bio_loss then batch_bacc
       from complete trials.
    """
    if df.empty:
        raise ValueError("No trials available for recommendation")
    work = df.copy()
    if "state" in work.columns:
        work = work[work["state"].astype(str).str.contains("COMPLETE")]
    work = work[np.isfinite(work["batch_bacc"].astype(float))]
    if work.empty:
        raise ValueError("No complete finite trials available for recommendation")

    eligible = work[work["bio_bacc"].astype(float) >= float(min_bio_bacc)]
    if max_rmse_change is not None and np.isfinite(max_rmse_change):
        eligible = eligible[eligible["rmse_change"].astype(float) <= float(max_rmse_change)]
    if not eligible.empty:
        return eligible.sort_values(["batch_bacc", "rmse_change", "bio_loss"]).iloc[0]
    return work.sort_values(["bio_loss", "batch_bacc", "rmse_change"]).iloc[0]


def plot_pareto_2d(
    trials_df: pd.DataFrame,
    outpath: str | Path,
    baseline_df: pd.DataFrame | None = None,
    constrained_floor: float | None = None,
    title: str = "AE batch/biology tradeoff",
) -> None:
    """Create a 2D batch-vs-biology Pareto plot.

    The x-axis is batch balanced accuracy (lower is better). The y-axis is
    biology balanced accuracy (higher is better). Points are colored by RMSE
    change when available.
    """
    import matplotlib.pyplot as plt

    outpath = Path(outpath)
    df = trials_df.copy()
    df = df[df.get("state", "COMPLETE").astype(str).str.contains("COMPLETE")]
    if df.empty:
        raise ValueError("No complete trials to plot")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    color_values = df["rmse_change"] if "rmse_change" in df.columns else None
    sc = ax.scatter(
        df["batch_bacc"],
        df["bio_bacc"],
        c=color_values,
        s=28,
        alpha=0.65,
        label="AE trials",
    )
    if color_values is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("RMSE change")

    pareto = df[df.get("is_pareto", False).astype(bool)].sort_values("batch_bacc")
    if not pareto.empty:
        ax.plot(pareto["batch_bacc"], pareto["bio_bacc"], marker="o", linewidth=1.5, label="AE Pareto front")

    if baseline_df is not None and not baseline_df.empty:
        for _, row in baseline_df.iterrows():
            x = row.get("batch_bacc", np.nan)
            y = row.get("bio_bacc", np.nan)
            if np.isfinite(x) and np.isfinite(y):
                ax.scatter([x], [y], marker="X", s=90, label=str(row.get("method", "baseline")))
                ax.annotate(str(row.get("method", "baseline")), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)

    if constrained_floor is not None and np.isfinite(constrained_floor):
        ax.axvline(constrained_floor, linestyle="--", linewidth=1.0, label=f"biology-preserving floor ≈ {constrained_floor:.3f}")

    ax.set_xlabel("Batch balanced accuracy (lower is better)")
    ax.set_ylabel("Biology balanced accuracy (higher is better)")
    ax.set_title(title)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    # De-duplicate labels while preserving order.
    seen = set()
    unique = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique.append((h, l))
    if unique:
        ax.legend([h for h, _ in unique], [l for _, l in unique], fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def normalize_comparison_metrics(path: str | Path, run_name: str | None = None) -> pd.DataFrame:
    """Normalize a run_real_experiment comparison_metrics.csv file."""
    path = Path(path)
    df = pd.read_csv(path)
    run_name = run_name or path.parent.name
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "run": run_name,
            "source_file": str(path),
            "method": r.get("method"),
            "protocol": "transductive_correction_cv_probe",
            "batch_bacc": r.get("corr_batch_balanced_accuracy"),
            "batch_acc": r.get("corr_batch_accuracy"),
            "batch_chance": r.get("corr_batch_chance"),
            "bio_bacc": r.get("corr_biology_balanced_accuracy"),
            "bio_acc": r.get("corr_biology_accuracy"),
            "rmse_change": r.get("observed_rmse_change"),
        })
    return pd.DataFrame(rows)


def normalize_heldout_metrics(path: str | Path, run_name: str | None = None) -> pd.DataFrame:
    """Normalize a heldout_baseline_metrics.csv file."""
    path = Path(path)
    df = pd.read_csv(path)
    run_name = run_name or path.parent.name
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "run": run_name,
            "source_file": str(path),
            "method": r.get("method"),
            "protocol": r.get("evaluation_protocol", "heldout_correction_and_probe"),
            "batch_bacc": r.get("batch_bacc"),
            "batch_acc": r.get("batch_acc"),
            "batch_chance": r.get("batch_chance"),
            "bio_bacc": r.get("bio_bacc"),
            "bio_acc": r.get("bio_acc"),
            "rmse_change": r.get("observed_rmse_change"),
            "constrained_batch_floor": r.get("constrained_batch_floor"),
            "flag": r.get("flag"),
        })
    return pd.DataFrame(rows)
