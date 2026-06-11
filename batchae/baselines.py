"""Baseline batch-correction methods, including tuned ComBat variants."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    method: str
    use_bio_covariate: bool = False
    par_prior: bool = True
    mean_only: bool = False

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def get_baseline_configs(include_combat: bool = True) -> list[BaselineConfig]:
    """Return baseline configs to tune fairly on validation seeds."""
    configs: list[BaselineConfig] = [
        BaselineConfig(name="ols_batch_only", method="ols", use_bio_covariate=False),
        BaselineConfig(name="ols_preserve_cellline", method="ols", use_bio_covariate=True),
        BaselineConfig(name="mean_center_batch", method="mean_center", use_bio_covariate=False),
    ]
    if include_combat:
        configs.extend(
            [
                BaselineConfig(name="combat_default", method="combat", use_bio_covariate=False, par_prior=True, mean_only=False),
                BaselineConfig(name="combat_preserve_cellline", method="combat", use_bio_covariate=True, par_prior=True, mean_only=False),
                BaselineConfig(name="combat_mean_only", method="combat", use_bio_covariate=False, par_prior=True, mean_only=True),
                BaselineConfig(name="combat_mean_only_preserve_cellline", method="combat", use_bio_covariate=True, par_prior=True, mean_only=True),
                BaselineConfig(name="combat_nonparametric", method="combat", use_bio_covariate=False, par_prior=False, mean_only=False),
                BaselineConfig(name="combat_nonparametric_preserve_cellline", method="combat", use_bio_covariate=True, par_prior=False, mean_only=False),
            ]
        )
    return configs


def mean_center_batch_correct(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Simple batch mean-centering per protein."""
    Xc = X.astype(np.float64).copy()
    grand = Xc.mean(axis=0, keepdims=True)
    for b in np.unique(meta["batch"].values):
        idx = meta["batch"].values == b
        Xc[idx] = Xc[idx] - Xc[idx].mean(axis=0, keepdims=True) + grand
    return Xc.astype(np.float32)


def _design_matrix(meta: pd.DataFrame, use_bio_covariate: bool) -> tuple[np.ndarray, list[int]]:
    """Build OLS design and return columns corresponding to batch effects."""
    n = len(meta)
    cols = [np.ones((n, 1), dtype=np.float64)]
    batch_cols: list[int] = []

    batches = sorted(np.unique(meta["batch"].values.astype(int)))
    # Drop reference batch to avoid collinearity with intercept.
    for b in batches[1:]:
        batch_cols.append(sum(c.shape[1] for c in cols))
        cols.append((meta["batch"].values.astype(int) == b).astype(np.float64).reshape(-1, 1))

    if use_bio_covariate:
        cell_lines = sorted(np.unique(meta["cell_line"].values.astype(int)))
        for cl in cell_lines[1:]:
            cols.append((meta["cell_line"].values.astype(int) == cl).astype(np.float64).reshape(-1, 1))

    return np.concatenate(cols, axis=1), batch_cols


def ols_batch_correct(X: np.ndarray, meta: pd.DataFrame, use_bio_covariate: bool = True) -> np.ndarray:
    """Remove OLS-estimated batch coefficients, optionally preserving cell-line effects."""
    design, batch_cols = _design_matrix(meta, use_bio_covariate=use_bio_covariate)
    beta, *_ = np.linalg.lstsq(design, X.astype(np.float64), rcond=None)
    batch_design = np.zeros_like(design)
    batch_design[:, batch_cols] = design[:, batch_cols]
    batch_component = batch_design @ beta
    return (X.astype(np.float64) - batch_component).astype(np.float32)


def _combat_mod(meta: pd.DataFrame) -> pd.DataFrame:
    # Drop one level to avoid exact collinearity. pycombat accepts pandas/numpy in most versions.
    return pd.get_dummies(meta["cell_line"].astype(str), prefix="cell", drop_first=True).astype(float)


def combat_correct(
    X: np.ndarray,
    meta: pd.DataFrame,
    protein_ids: list[str] | None = None,
    use_bio_covariate: bool = False,
    par_prior: bool = True,
    mean_only: bool = False,
) -> np.ndarray:
    """Run pyComBat on proteins x samples, returning samples x proteins.

    The original notebook used pycombat(X_df, batch) only. Here, configs can also
    include a cell-line model matrix so biology is explicitly protected.
    """
    try:
        from combat.pycombat import pycombat
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install the optional package with `pip install combat` to run ComBat.") from exc

    if protein_ids is None:
        protein_ids = [f"PROT_{j:04d}" for j in range(X.shape[1])]

    data = pd.DataFrame(X.T, index=protein_ids)
    batch = meta["batch"].values
    mod = _combat_mod(meta) if use_bio_covariate else None

    # pycombat signatures vary a bit across package versions, so we try the most
    # informative call first and fall back carefully.
    kwargs = {"par_prior": par_prior, "mean_only": mean_only}
    if mod is not None and mod.shape[1] > 0:
        try:
            corrected = pycombat(data, batch, mod=mod, **kwargs)
        except TypeError:
            corrected = pycombat(data, batch, mod.values, **kwargs)
    else:
        corrected = pycombat(data, batch, **kwargs)

    if isinstance(corrected, pd.DataFrame):
        arr = corrected.values.T
    else:
        arr = np.asarray(corrected).T
    return arr.astype(np.float32)


def apply_baseline(config: BaselineConfig, X: np.ndarray, meta: pd.DataFrame, protein_ids: list[str] | None = None) -> np.ndarray:
    if config.method == "mean_center":
        return mean_center_batch_correct(X, meta)
    if config.method == "ols":
        return ols_batch_correct(X, meta, use_bio_covariate=config.use_bio_covariate)
    if config.method == "combat":
        return combat_correct(
            X,
            meta,
            protein_ids=protein_ids,
            use_bio_covariate=config.use_bio_covariate,
            par_prior=config.par_prior,
            mean_only=config.mean_only,
        )
    raise ValueError(f"Unknown baseline method: {config.method}")


def tune_baseline_configs(
    configs: list[BaselineConfig],
    datasets: list[tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], np.ndarray]],
    evaluate_fn: Callable[[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray], dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], float],
) -> tuple[BaselineConfig, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        scores = []
        failed = False
        err = ""
        for X, _M, meta, protein_ids, X_clean in datasets:
            try:
                X_corr = apply_baseline(cfg, X, meta, protein_ids=protein_ids)
                metrics = evaluate_fn(X, X_corr, meta, X_clean)
                score = score_fn(metrics)
                scores.append(score)
            except Exception as exc:
                failed = True
                err = repr(exc)
                break
        rows.append(
            {
                "name": cfg.name,
                "method": cfg.method,
                "use_bio_covariate": cfg.use_bio_covariate,
                "par_prior": cfg.par_prior,
                "mean_only": cfg.mean_only,
                "score_mean": float(np.mean(scores)) if scores else np.inf,
                "score_std": float(np.std(scores)) if scores else np.inf,
                "failed": failed,
                "error": err,
            }
        )
    df = pd.DataFrame(rows).sort_values("score_mean").reset_index(drop=True)
    if len(df) == 0 or np.isinf(df.iloc[0]["score_mean"]):
        raise RuntimeError("All baseline configs failed. Check ComBat install or disable ComBat.")
    best_name = str(df.iloc[0]["name"])
    best_cfg = next(c for c in configs if c.name == best_name)
    return best_cfg, df
