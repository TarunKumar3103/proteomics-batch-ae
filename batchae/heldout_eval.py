"""Leakage-safe held-out evaluation for simple batch-correction baselines.

This module fixes the artifact caused by this transductive protocol:

    correct the full matrix once -> cross-validate a probe on the corrected matrix

For simple correctors such as batch mean-centering and OLS residualization, that
protocol lets held-out probe rows influence the correction parameters. The
functions here instead use a single outer split:

    for each fold:
        fit the correction on train rows only
        transform train and held-out test rows with the train-fitted corrector
        train the probe on corrected train rows
        predict corrected held-out rows

The headline metrics are then computed from the pooled held-out predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from batchae.metrics import constrained_batch_floor, reconstruction_distortion


@dataclass
class FittedCorrector:
    """Small callable object returned by train-fold corrector fits."""

    name: str
    transform: Callable[[np.ndarray, pd.DataFrame], np.ndarray]


def _nan_center(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.nanmean(X, axis=0)
    if mode == "median":
        return np.nanmedian(X, axis=0)
    raise ValueError(f"Unknown center mode: {mode!r}")


def fit_batch_center_corrector(
    X_train: np.ndarray,
    meta_train: pd.DataFrame,
    batch_col: str,
    mode: str = "mean",
) -> FittedCorrector:
    """Fit per-batch train-only centering and return a train/test transformer."""
    X_train = np.asarray(X_train, dtype=np.float64)
    batch_train = meta_train[batch_col].astype(str).to_numpy()
    global_center = _nan_center(X_train, mode)
    centers = {}
    for b in np.unique(batch_train):
        centers[b] = _nan_center(X_train[batch_train == b], mode)

    def transform(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        labels = meta[batch_col].astype(str).to_numpy()
        Xc = X.copy()
        for b in np.unique(labels):
            idx = labels == b
            center = centers.get(b, global_center)
            Xc[idx] = X[idx] - center + global_center
        return Xc.astype(np.float32)

    return FittedCorrector(name=f"batch_{mode}_center", transform=transform)


def _onehot_fixed(values, classes: list[str], drop_first: bool = True) -> np.ndarray:
    values = np.asarray(values).astype(str)
    use_classes = classes[1:] if drop_first and len(classes) > 1 else classes
    if not use_classes:
        return np.zeros((len(values), 0), dtype=float)
    return np.column_stack([(values == c).astype(float) for c in use_classes])


def fit_ols_corrector(
    X_train: np.ndarray,
    meta_train: pd.DataFrame,
    batch_col: str,
    biology_col: str | None = None,
    preserve_biology: bool = False,
) -> FittedCorrector:
    """Fit train-only OLS batch residualization and return a transformer.

    The fitted model is protein-wise linear regression. Correction subtracts only
    the batch-design contribution from any matrix passed to transform().
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    batch_classes = sorted(meta_train[batch_col].astype(str).unique())
    bio_classes: list[str] = []
    use_bio = bool(
        preserve_biology
        and biology_col is not None
        and biology_col in meta_train.columns
        and meta_train[biology_col].nunique() >= 2
    )
    if use_bio:
        bio_classes = sorted(meta_train[biology_col].astype(str).unique())

    def design(meta: pd.DataFrame) -> tuple[np.ndarray, int, int]:
        n = len(meta)
        B = _onehot_fixed(meta[batch_col].astype(str).to_numpy(), batch_classes, drop_first=True)
        parts = [np.ones((n, 1), dtype=float), B]
        batch_start = 1
        batch_end = 1 + B.shape[1]
        if use_bio:
            Bio = _onehot_fixed(meta[biology_col].astype(str).to_numpy(), bio_classes, drop_first=True)
            parts.append(Bio)
        return np.column_stack(parts), batch_start, batch_end

    D_train, batch_start, batch_end = design(meta_train)
    coef, *_ = np.linalg.lstsq(D_train, X_train, rcond=None)

    def transform(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        D, _, _ = design(meta)
        batch_effect = D[:, batch_start:batch_end] @ coef[batch_start:batch_end, :]
        return (X - batch_effect).astype(np.float32)

    name = "ols_preserve_biology" if use_bio else "ols_remove_batch"
    return FittedCorrector(name=name, transform=transform)


def _probe_train_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
    max_iter: int,
) -> np.ndarray:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(np.asarray(y_train).astype(str))
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", random_state=random_state)
    clf.fit(X_train_s, y_train_enc)
    y_pred_enc = clf.predict(X_test_s)
    return le.inverse_transform(y_pred_enc)


def _safe_outer_cv(labels, requested_cv: int) -> int:
    labels = np.asarray(labels).astype(str)
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2:
        return 0
    return max(0, min(int(requested_cv), int(counts.min())))


def evaluate_corrector_heldout(
    name: str,
    ds,
    fit_corrector: Callable[[np.ndarray, pd.DataFrame], FittedCorrector],
    batch_col: str,
    biology_col: str | None = None,
    cv: int = 5,
    random_state: int = 42,
    max_iter: int = 10000,
) -> tuple[dict, np.ndarray]:
    """Evaluate a correction method with train-only correction and held-out probes.

    Returns
    -------
    metrics, X_corrected_oof
        X_corrected_oof stores each row corrected by a model that did not fit on
        that row. The metrics are pooled held-out probe predictions.
    """
    X = np.asarray(ds.X, dtype=np.float32)
    meta = ds.meta.reset_index(drop=True)
    y_batch = meta[batch_col].astype(str).to_numpy()
    cv_eff = _safe_outer_cv(y_batch, cv)
    if cv_eff < 2:
        raise ValueError(f"Not enough samples per batch for cv={cv}")

    splitter = StratifiedKFold(n_splits=cv_eff, shuffle=True, random_state=random_state)

    X_oof = np.empty_like(X, dtype=np.float32)
    batch_pred = np.empty(len(meta), dtype=object)
    bio_pred = np.empty(len(meta), dtype=object) if biology_col and biology_col in meta.columns else None
    y_bio = meta[biology_col].astype(str).to_numpy() if bio_pred is not None else None

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_batch), start=1):
        meta_train = meta.iloc[train_idx].reset_index(drop=True)
        meta_test = meta.iloc[test_idx].reset_index(drop=True)
        corrector = fit_corrector(X[train_idx], meta_train)
        X_train_corr = corrector.transform(X[train_idx], meta_train)
        X_test_corr = corrector.transform(X[test_idx], meta_test)
        X_oof[test_idx] = X_test_corr

        batch_pred[test_idx] = _probe_train_predict(
            X_train_corr,
            y_batch[train_idx],
            X_test_corr,
            random_state=random_state + fold,
            max_iter=max_iter,
        )
        if bio_pred is not None and len(np.unique(y_bio[train_idx])) >= 2:
            bio_pred[test_idx] = _probe_train_predict(
                X_train_corr,
                y_bio[train_idx],
                X_test_corr,
                random_state=random_state + 1000 + fold,
                max_iter=max_iter,
            )

    batch_classes = np.unique(y_batch)
    result = {
        "method": name,
        "evaluation_protocol": "heldout_correction_and_probe",
        "batch_bacc": float(balanced_accuracy_score(y_batch, batch_pred)),
        "batch_acc": float(accuracy_score(y_batch, batch_pred)),
        "batch_chance": float(1.0 / len(batch_classes)),
        "batch_cv": cv_eff,
    }

    if y_bio is not None and len(np.unique(y_bio)) >= 2:
        # If any fold could not produce biology predictions, this will remain None-like.
        result.update({
            "bio_bacc": float(balanced_accuracy_score(y_bio, bio_pred)),
            "bio_acc": float(accuracy_score(y_bio, bio_pred)),
            "bio_chance": float(1.0 / len(np.unique(y_bio))),
            "constrained_batch_floor": constrained_batch_floor(y_batch, y_bio),
        })
    else:
        result.update({
            "bio_bacc": np.nan,
            "bio_acc": np.nan,
            "bio_chance": np.nan,
            "constrained_batch_floor": np.nan,
        })

    result.update(reconstruction_distortion(ds.X, X_oof, ds.M))
    return result, X_oof


def heldout_baseline_grid(
    ds,
    batch_col: str,
    biology_col: str | None = None,
    cv: int = 5,
    random_state: int = 42,
    max_iter: int = 10000,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Run leakage-safe held-out evaluation for raw, centering, and OLS baselines."""
    specs: list[tuple[str, Callable[[np.ndarray, pd.DataFrame], FittedCorrector]]] = [
        ("raw", lambda Xtr, meta_tr: FittedCorrector("raw", lambda X, meta: np.asarray(X, dtype=np.float32))),
        ("batch_mean_center", lambda Xtr, meta_tr: fit_batch_center_corrector(Xtr, meta_tr, batch_col, mode="mean")),
        ("batch_median_center", lambda Xtr, meta_tr: fit_batch_center_corrector(Xtr, meta_tr, batch_col, mode="median")),
        ("ols_remove_batch", lambda Xtr, meta_tr: fit_ols_corrector(Xtr, meta_tr, batch_col, biology_col=None, preserve_biology=False)),
    ]
    if biology_col and biology_col in ds.meta.columns and ds.meta[biology_col].nunique() >= 2:
        specs.append(
            (
                "ols_preserve_biology",
                lambda Xtr, meta_tr: fit_ols_corrector(
                    Xtr,
                    meta_tr,
                    batch_col,
                    biology_col=biology_col,
                    preserve_biology=True,
                ),
            )
        )

    results: list[dict] = []
    matrices: dict[str, np.ndarray] = {}
    for name, fit_fn in specs:
        print(f"[heldout] Evaluating {name}...", flush=True)
        metrics, X_oof = evaluate_corrector_heldout(
            name,
            ds,
            fit_fn,
            batch_col=batch_col,
            biology_col=biology_col,
            cv=cv,
            random_state=random_state,
            max_iter=max_iter,
        )
        results.append(metrics)
        matrices[name] = X_oof
    return results, matrices


def flag_result(row: dict, tolerance: float = 0.02) -> str:
    """Flag impossible/suspicious rows so they are not accidentally reported."""
    batch_bacc = row.get("batch_bacc", np.nan)
    chance = row.get("batch_chance", np.nan)
    bio_bacc = row.get("bio_bacc", np.nan)
    floor = row.get("constrained_batch_floor", np.nan)
    if np.isfinite(batch_bacc) and np.isfinite(chance) and batch_bacc < chance - 1e-6:
        return "SUB_CHANCE_CHECK_PROTOCOL"
    if (
        np.isfinite(batch_bacc)
        and np.isfinite(bio_bacc)
        and np.isfinite(floor)
        and bio_bacc >= 0.90
        and batch_bacc < floor - tolerance
    ):
        return "BELOW_BIOLOGY_PRESERVING_FLOOR"
    return "OK"
