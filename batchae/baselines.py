"""Baseline batch-correction methods for fair comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


def batch_center(X: np.ndarray, batch_labels, mode: str = "mean") -> np.ndarray:
    """Remove per-batch protein-wise mean/median shift and add global center."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(batch_labels).astype(str)
    Xc = X.copy()
    global_center = np.nanmean(X, axis=0) if mode == "mean" else np.nanmedian(X, axis=0)

    for b in np.unique(labels):
        idx = labels == b
        center = np.nanmean(X[idx], axis=0) if mode == "mean" else np.nanmedian(X[idx], axis=0)
        Xc[idx] = X[idx] - center + global_center
    return Xc.astype(np.float32)


def _onehot(values, drop_first: bool = True) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values).astype(str)
    classes = sorted(np.unique(values))
    start = 1 if drop_first and len(classes) > 1 else 0
    cols = []
    names = []
    for c in classes[start:]:
        cols.append((values == c).astype(float))
        names.append(c)
    if not cols:
        return np.zeros((len(values), 0)), []
    return np.column_stack(cols), names


def ols_remove_batch(X: np.ndarray, batch_labels, biology_labels=None, preserve_biology: bool = True) -> np.ndarray:
    """Linear residualization: remove batch coefficients, optionally preserve biology.

    For each protein, fit y ~ intercept + batch + biology. Corrected values are
    y - batch_effect. This is a simple transparent baseline, not a replacement
    for ComBat.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    B, _ = _onehot(batch_labels, drop_first=True)
    design_parts = [np.ones((n, 1)), B]
    batch_start = 1
    batch_end = 1 + B.shape[1]

    if preserve_biology and biology_labels is not None and len(np.unique(biology_labels)) >= 2:
        Bio, _ = _onehot(biology_labels, drop_first=True)
        design_parts.append(Bio)

    D = np.column_stack(design_parts)
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    batch_effect = D[:, batch_start:batch_end] @ coef[batch_start:batch_end, :]
    Xc = X - batch_effect
    return Xc.astype(np.float32)


def run_combat(
    X: np.ndarray,
    protein_ids: list[str],
    batch_labels,
    biology_labels=None,
    preserve_biology: bool = False,
    mean_only: bool = False,
) -> np.ndarray:
    """Run pycombat if the combat package is installed.

    Input/output convention here is samples x proteins, while pycombat expects
    features x samples.
    """
    try:
        from combat.pycombat import pycombat
    except Exception as exc:
        raise RuntimeError(
            "ComBat baseline requested, but the 'combat' package is not installed. "
            "Install with: pip install combat"
        ) from exc

    X_df = pd.DataFrame(np.asarray(X).T, index=protein_ids)
    batch = np.asarray(batch_labels).astype(str)

    kwargs = {}
    if preserve_biology and biology_labels is not None and len(np.unique(biology_labels)) >= 2:
        bio = pd.get_dummies(pd.Series(np.asarray(biology_labels).astype(str), name="biology"), drop_first=True)
        kwargs["mod"] = bio
    if mean_only:
        kwargs["mean_only"] = True

    try:
        corrected = pycombat(X_df, batch, **kwargs)
    except TypeError:
        # Some pycombat versions do not expose mean_only/mod with the same names.
        kwargs.pop("mean_only", None)
        corrected = pycombat(X_df, batch, **kwargs)

    return corrected.values.T.astype(np.float32)


def baseline_grid(X: np.ndarray, protein_ids: list[str], meta, batch_col: str, biology_col: str | None = None) -> dict[str, np.ndarray]:
    """Run a modest, defensible baseline grid."""
    batch = meta[batch_col].values
    bio = meta[biology_col].values if biology_col and biology_col in meta.columns and meta[biology_col].nunique() >= 2 else None

    outputs: dict[str, np.ndarray] = {
        "raw": np.asarray(X, dtype=np.float32),
        "batch_mean_center": batch_center(X, batch, mode="mean"),
        "batch_median_center": batch_center(X, batch, mode="median"),
        "ols_remove_batch": ols_remove_batch(X, batch, biology_labels=None, preserve_biology=False),
    }

    if bio is not None:
        outputs["ols_preserve_biology"] = ols_remove_batch(X, batch, biology_labels=bio, preserve_biology=True)

    # ComBat can fail on tiny/degenerate class layouts; keep the experiment going.
    for preserve in [False, True]:
        if preserve and bio is None:
            continue
        name = "combat_preserve_biology" if preserve else "combat"
        try:
            outputs[name] = run_combat(X, protein_ids, batch, biology_labels=bio, preserve_biology=preserve)
        except Exception as exc:
            print(f"WARNING: {name} failed: {exc}")

    try:
        outputs["combat_mean_only"] = run_combat(X, protein_ids, batch, biology_labels=bio, preserve_biology=bool(bio is not None), mean_only=True)
    except Exception as exc:
        print(f"WARNING: combat_mean_only failed: {exc}")

    return outputs
