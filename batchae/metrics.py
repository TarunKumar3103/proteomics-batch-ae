"""Evaluation metrics for real-data batch correction.

Real proteomics data usually has no clean ground truth, so this module avoids
synthetic-only metrics such as MSE-to-clean. It evaluates whether batch signal is
reduced while known biological signal is preserved.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _safe_cv(labels, requested_cv: int = 5) -> int:
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2:
        return 0
    return max(0, min(int(requested_cv), int(counts.min())))


def classifier_scores(X: np.ndarray, labels, cv: int = 5, n_jobs: int = 1, random_state: int = 42) -> dict:
    """Cross-validated linear classifier accuracy and balanced accuracy."""
    labels = np.asarray(labels).astype(str)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    cv_eff = _safe_cv(y, cv)

    if n_classes < 2 or cv_eff < 2:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "chance": np.nan,
            "majority_baseline": np.nan,
            "n_classes": n_classes,
            "cv": cv_eff,
        }

    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=5000, solver="lbfgs", n_jobs=None, random_state=random_state)
    splitter = StratifiedKFold(n_splits=cv_eff, shuffle=True, random_state=random_state)

    acc = cross_val_score(clf, Xs, y, cv=splitter, scoring="accuracy", n_jobs=n_jobs)
    bacc = cross_val_score(clf, Xs, y, cv=splitter, scoring="balanced_accuracy", n_jobs=n_jobs)

    counts = np.bincount(y)
    return {
        "accuracy": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "balanced_accuracy": float(np.mean(bacc)),
        "balanced_accuracy_std": float(np.std(bacc)),
        "chance": float(1.0 / n_classes),
        "majority_baseline": float(counts.max() / counts.sum()),
        "n_classes": n_classes,
        "cv": cv_eff,
    }


def reconstruction_distortion(X_raw: np.ndarray, X_corr: np.ndarray, M: np.ndarray) -> dict:
    """Magnitude of correction on originally observed values.

    This is not an accuracy metric. It is a sanity-check to flag methods that
    erase too much of the original observed signal.
    """
    obs = M.astype(bool)
    if obs.sum() == 0:
        return {"observed_rmse_change": np.nan, "observed_mae_change": np.nan}
    diff = X_corr[obs] - X_raw[obs]
    return {
        "observed_rmse_change": float(np.sqrt(np.mean(diff ** 2))),
        "observed_mae_change": float(np.mean(np.abs(diff))),
    }


def pca_silhouette(X: np.ndarray, labels, n_components: int = 10, random_state: int = 42) -> float:
    labels = np.asarray(labels).astype(str)
    if len(np.unique(labels)) < 2:
        return np.nan
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < 2:
        return np.nan
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_components < 2:
        return np.nan
    coords = PCA(n_components=n_components, random_state=random_state).fit_transform(StandardScaler().fit_transform(X))
    try:
        return float(silhouette_score(coords, labels))
    except Exception:
        return np.nan


def evaluate_correction(
    name: str,
    X_raw: np.ndarray,
    X_corr: np.ndarray,
    M: np.ndarray,
    meta: pd.DataFrame,
    batch_col: str = "batch",
    biology_col: str = "biology",
    cv: int = 5,
    n_jobs: int = 1,
    random_state: int = 42,
) -> dict:
    """Evaluate one corrected matrix against raw data."""
    out: dict[str, float | str] = {"method": name}

    raw_batch = classifier_scores(X_raw, meta[batch_col].values, cv=cv, n_jobs=n_jobs, random_state=random_state)
    cor_batch = classifier_scores(X_corr, meta[batch_col].values, cv=cv, n_jobs=n_jobs, random_state=random_state)
    out.update({f"raw_batch_{k}": v for k, v in raw_batch.items()})
    out.update({f"corr_batch_{k}": v for k, v in cor_batch.items()})

    if biology_col in meta.columns and meta[biology_col].nunique() >= 2:
        raw_bio = classifier_scores(X_raw, meta[biology_col].values, cv=cv, n_jobs=n_jobs, random_state=random_state)
        cor_bio = classifier_scores(X_corr, meta[biology_col].values, cv=cv, n_jobs=n_jobs, random_state=random_state)
        out.update({f"raw_biology_{k}": v for k, v in raw_bio.items()})
        out.update({f"corr_biology_{k}": v for k, v in cor_bio.items()})
        out["raw_biology_silhouette"] = pca_silhouette(X_raw, meta[biology_col].values, random_state=random_state)
        out["corr_biology_silhouette"] = pca_silhouette(X_corr, meta[biology_col].values, random_state=random_state)
    else:
        warnings.warn("Biology column missing or has <2 classes; biology preservation metrics skipped.")

    out["raw_batch_silhouette"] = pca_silhouette(X_raw, meta[batch_col].values, random_state=random_state)
    out["corr_batch_silhouette"] = pca_silhouette(X_corr, meta[batch_col].values, random_state=random_state)
    out.update(reconstruction_distortion(X_raw, X_corr, M))

    return out


def print_compact_results(results: list[dict]) -> None:
    rows = []
    for r in results:
        rows.append({
            "method": r.get("method"),
            "batch_bacc": r.get("corr_batch_balanced_accuracy"),
            "batch_acc": r.get("corr_batch_accuracy"),
            "batch_chance": r.get("corr_batch_chance"),
            "bio_bacc": r.get("corr_biology_balanced_accuracy"),
            "bio_acc": r.get("corr_biology_accuracy"),
            "batch_sil": r.get("corr_batch_silhouette"),
            "bio_sil": r.get("corr_biology_silhouette"),
            "rmse_change": r.get("observed_rmse_change"),
        })
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df.to_string(index=False))
