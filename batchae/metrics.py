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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _safe_cv(labels, requested_cv: int = 5) -> int:
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2:
        return 0
    return max(0, min(int(requested_cv), int(counts.min())))


def classifier_scores(
    X: np.ndarray,
    labels,
    cv: int = 5,
    n_jobs: int = 1,
    random_state: int = 42,
    max_iter: int = 10000,
) -> dict:
    """Cross-validated linear classifier accuracy and balanced accuracy.

    Uses one cross-validation prediction pass and computes both metrics from the
    same held-out predictions. This avoids fitting the same classifier twice.
    """
    labels = np.asarray(labels).astype(str)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    cv_eff = _safe_cv(y, cv)

    if n_classes < 2 or cv_eff < 2:
        return {
            "accuracy": np.nan,
            "accuracy_std": np.nan,
            "balanced_accuracy": np.nan,
            "balanced_accuracy_std": np.nan,
            "chance": np.nan,
            "majority_baseline": np.nan,
            "n_classes": n_classes,
            "cv": cv_eff,
        }

    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", n_jobs=None, random_state=random_state)
    splitter = StratifiedKFold(n_splits=cv_eff, shuffle=True, random_state=random_state)

    y_pred = cross_val_predict(clf, Xs, y, cv=splitter, n_jobs=n_jobs)

    # Fold-level values are useful for rough variability/debugging. The headline
    # metric is the global held-out prediction score above.
    acc_fold = []
    bacc_fold = []
    for _, test_idx in splitter.split(Xs, y):
        acc_fold.append(accuracy_score(y[test_idx], y_pred[test_idx]))
        bacc_fold.append(balanced_accuracy_score(y[test_idx], y_pred[test_idx]))

    counts = np.bincount(y)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "accuracy_std": float(np.std(acc_fold)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "balanced_accuracy_std": float(np.std(bacc_fold)),
        "chance": float(1.0 / n_classes),
        "majority_baseline": float(counts.max() / counts.sum()),
        "n_classes": n_classes,
        "cv": cv_eff,
    }


def classifier_scores_fast(X: np.ndarray, labels, cv: int = 5, n_jobs: int = 1, random_state: int = 42) -> dict:
    """Cheaper classifier probe for Optuna's inner loop.

    It uses fewer optimizer iterations than the final reporting metric. The
    objective only needs a stable ranking signal; final metrics still call
    classifier_scores with the higher default max_iter.
    """
    return classifier_scores(
        X,
        labels,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        max_iter=2000,
    )


def constrained_batch_floor(batch_labels, biology_labels) -> float:
    """Expected batch-bAcc floor when biology is preserved but batch is erased within biology.

    If biology perfectly partitions the batches, a biology-preserving correction
    can still reveal which biology family a sample belongs to. The best possible
    within-family batch mixing therefore has an expected recall of 1/k for each
    batch, where k is the number of batch classes inside that biology family.
    """
    batch = np.asarray(batch_labels).astype(str)
    biology = np.asarray(biology_labels).astype(str)
    if len(batch) != len(biology) or len(np.unique(batch)) == 0:
        return np.nan

    floors = []
    for b in np.unique(batch):
        fams = np.unique(biology[batch == b])
        if len(fams) != 1:
            return np.nan
        fam = fams[0]
        n_batches_in_family = len(np.unique(batch[biology == fam]))
        floors.append(1.0 / max(n_batches_in_family, 1))
    return float(np.mean(floors))


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
