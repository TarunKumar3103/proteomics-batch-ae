"""Evaluation metrics and objective scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ScoreConfig:
    min_cell_acc: float = 0.75
    clean_weight: float = 0.02
    mse_blowup_factor: float = 1.5
    bad_score: float = 10.0


def _safe_cv(labels: np.ndarray, requested_cv: int = 5) -> int:
    _, counts = np.unique(labels, return_counts=True)
    return int(max(2, min(requested_cv, counts.min())))


def adversarial_accuracy(X_np: np.ndarray, labels: np.ndarray, cv: int = 5, n_jobs: int = 1, seed: int = 42):
    """Cross-validated linear classifier accuracy on a representation/corrected data."""
    X_scaled = StandardScaler().fit_transform(X_np)
    labels = np.asarray(labels)
    cv_eff = _safe_cv(labels, cv)
    splitter = StratifiedKFold(n_splits=cv_eff, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=seed, n_jobs=n_jobs)
    scores = cross_val_score(clf, X_scaled, labels, cv=splitter, scoring="accuracy", n_jobs=n_jobs)
    return float(scores.mean()), float(scores.std())


def mse_to_clean(X: np.ndarray, X_clean: np.ndarray) -> float:
    return float(np.mean((X - X_clean) ** 2))


def corr_to_clean(X: np.ndarray, X_clean: np.ndarray) -> float:
    return float(np.corrcoef(X.flatten(), X_clean.flatten())[0, 1])


def evaluate_correction(
    X_raw: np.ndarray,
    X_corrected: np.ndarray,
    meta: pd.DataFrame,
    X_clean: np.ndarray | None = None,
    cv: int = 5,
    n_jobs: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    batch_labels = meta["batch"].values
    cell_labels = meta["cell_line"].values
    chance = 1.0 / meta["batch"].nunique()

    raw_b, raw_b_std = adversarial_accuracy(X_raw, batch_labels, cv=cv, n_jobs=n_jobs, seed=seed)
    cor_b, cor_b_std = adversarial_accuracy(X_corrected, batch_labels, cv=cv, n_jobs=n_jobs, seed=seed)
    raw_cl, raw_cl_std = adversarial_accuracy(X_raw, cell_labels, cv=cv, n_jobs=n_jobs, seed=seed)
    cor_cl, cor_cl_std = adversarial_accuracy(X_corrected, cell_labels, cv=cv, n_jobs=n_jobs, seed=seed)

    out: dict[str, Any] = {
        "raw_batch": raw_b,
        "raw_batch_std": raw_b_std,
        "corr_batch": cor_b,
        "corr_batch_std": cor_b_std,
        "chance": chance,
        "raw_cellline": raw_cl,
        "raw_cellline_std": raw_cl_std,
        "corr_cellline": cor_cl,
        "corr_cellline_std": cor_cl_std,
    }

    if X_clean is not None:
        out.update(
            {
                "raw_mse": mse_to_clean(X_raw, X_clean),
                "corr_mse": mse_to_clean(X_corrected, X_clean),
                "raw_corr": corr_to_clean(X_raw, X_clean),
                "corr_corr": corr_to_clean(X_corrected, X_clean),
            }
        )
    return out


def score_metrics(metrics: dict[str, Any], config: ScoreConfig | None = None) -> float:
    """Lower-is-better tuning score.

    Same spirit as the notebook objective, but reusable for autoencoder and
    baselines. Penalizes biology collapse and major MSE blow-up.
    """
    cfg = config or ScoreConfig()
    if metrics["corr_cellline"] < cfg.min_cell_acc:
        return cfg.bad_score
    if "corr_mse" in metrics and "raw_mse" in metrics:
        if metrics["corr_mse"] > metrics["raw_mse"] * cfg.mse_blowup_factor:
            return cfg.bad_score
        return float(metrics["corr_batch"] + cfg.clean_weight * metrics["corr_mse"])
    return float(metrics["corr_batch"])


def summarize_results(df: pd.DataFrame, group_col: str = "method") -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in {"seed", group_col, "config"} and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for method, g in df.groupby(group_col):
        row = {group_col: method, "n": len(g)}
        for c in metric_cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)
