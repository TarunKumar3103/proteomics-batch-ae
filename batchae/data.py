"""Synthetic proteomics data generator used in the original Colab cells."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_hard_proteomics(
    n_samples: int = 180,
    n_proteins: int = 1200,
    n_batches: int = 6,
    n_cell_lines: int = 3,
    missing_rate: float = 0.20,
    batch_effect_scale: float = 2.5,
    biology_scale: float = 1.2,
    noise_scale: float = 0.8,
    confounding_strength: float = 0.65,
    nonlinear_strength: float = 0.8,
    drift_strength: float = 1.0,
    outlier_frac: float = 0.04,
    seed: int = 123,
):
    """Generate a hard synthetic proteomics batch-correction benchmark.

    Returns
    -------
    X_filled : np.ndarray, shape (n_samples, n_proteins)
        Observed data with missing values mean-imputed protein-wise.
    M : np.ndarray, shape (n_samples, n_proteins)
        Observed/missing mask, where 1 means observed.
    meta : pd.DataFrame
        sample_id, batch, cell_line, run_order, sample_quality, is_outlier.
    protein_ids : list[str]
        Protein names.
    X_clean : np.ndarray
        Ground-truth biology without batch effects/noise.
    """

    rng = np.random.default_rng(seed)

    batch_sizes = rng.multinomial(n_samples, rng.dirichlet(np.ones(n_batches) * 1.5))
    while np.any(batch_sizes < 8):
        batch_sizes = rng.multinomial(n_samples, rng.dirichlet(np.ones(n_batches) * 1.5))

    batch_labels = np.concatenate([np.full(sz, b, dtype=int) for b, sz in enumerate(batch_sizes)])
    rng.shuffle(batch_labels)

    cell_line_labels = np.zeros(n_samples, dtype=int)
    batch_dominant_cl = rng.integers(0, n_cell_lines, size=n_batches)

    for i in range(n_samples):
        b = batch_labels[i]
        if rng.random() < confounding_strength:
            cell_line_labels[i] = batch_dominant_cl[b]
        else:
            cell_line_labels[i] = rng.integers(0, n_cell_lines)

    protein_base = rng.normal(14.0, 1.7, size=n_proteins)
    low_abundance = rng.choice(n_proteins, size=n_proteins // 5, replace=False)
    protein_base[low_abundance] -= rng.uniform(2.0, 4.0, size=len(low_abundance))

    cell_line_effects = np.zeros((n_cell_lines, n_proteins))
    for cl in range(n_cell_lines):
        active = rng.choice(n_proteins, size=n_proteins // 6, replace=False)
        cell_line_effects[cl, active] = rng.normal(0, biology_scale, size=len(active))

    n_bio_modules = 5
    bio_modules = rng.normal(0, 1, size=(n_bio_modules, n_proteins))
    bio_loadings = rng.normal(0, biology_scale * 0.5, size=(n_cell_lines, n_bio_modules))
    cell_line_effects += bio_loadings @ bio_modules

    n_batch_factors = 5
    batch_factors = rng.normal(0, 1, size=(n_batch_factors, n_proteins))
    batch_loadings = rng.normal(0, batch_effect_scale, size=(n_batches, n_batch_factors))
    additive_batch_effects = batch_loadings @ batch_factors

    batch_scale = rng.lognormal(mean=0.0, sigma=0.18, size=(n_batches, n_proteins))
    nonlinear_batch = rng.normal(0, nonlinear_strength, size=(n_batches, n_proteins))

    run_order = np.zeros(n_samples)
    for b in range(n_batches):
        idx = np.where(batch_labels == b)[0]
        order = np.linspace(-1, 1, len(idx))
        rng.shuffle(order)
        run_order[idx] = order

    drift_direction = rng.normal(0, 1, size=n_proteins)

    true_X_clean = np.zeros((n_samples, n_proteins))
    X_observed = np.zeros((n_samples, n_proteins))
    sample_quality = rng.lognormal(mean=0.0, sigma=0.20, size=n_samples)

    for i in range(n_samples):
        cl = cell_line_labels[i]
        b = batch_labels[i]

        clean = protein_base + cell_line_effects[cl]
        true_X_clean[i] = clean

        centered_clean = clean - clean.mean()
        additive = additive_batch_effects[b]
        multiplicative = batch_scale[b] * centered_clean
        nonlinear = nonlinear_batch[b] * np.tanh(centered_clean / 2.0)
        drift = drift_strength * run_order[i] * drift_direction
        noise = rng.normal(0, noise_scale * sample_quality[i], size=n_proteins)

        X_observed[i] = (
            protein_base
            + multiplicative
            + cell_line_effects[cl]
            + additive
            + nonlinear
            + drift
            + noise
        )

    n_outliers = max(1, int(outlier_frac * n_samples))
    outlier_samples = rng.choice(n_samples, size=n_outliers, replace=False)

    for i in outlier_samples:
        affected = rng.choice(n_proteins, size=max(1, n_proteins // 8), replace=False)
        X_observed[i, affected] += rng.normal(0, 5.0, size=len(affected))

    abundance_term = 1 / (1 + np.exp((protein_base - np.median(protein_base)) / 1.2))
    protein_missing_bias = rng.beta(2, 8, size=n_proteins)
    batch_missing_bias = rng.normal(0, 0.12, size=(n_batches, n_proteins))

    sample_missing_bias = sample_quality - sample_quality.min()
    sample_missing_bias = sample_missing_bias / (sample_missing_bias.max() + 1e-8)

    miss_prob = np.zeros((n_samples, n_proteins))
    for i in range(n_samples):
        b = batch_labels[i]
        logits = (
            -2.2
            + 2.2 * abundance_term
            + 1.5 * protein_missing_bias
            + batch_missing_bias[b]
            + 0.8 * sample_missing_bias[i]
        )
        p = 1 / (1 + np.exp(-logits))
        p = missing_rate * p / p.mean()
        miss_prob[i] = np.clip(p, 0.01, 0.75)

    M = (rng.random(size=(n_samples, n_proteins)) > miss_prob).astype(np.float32)

    X_filled = X_observed.copy()
    for j in range(n_proteins):
        obs_mask = M[:, j].astype(bool)
        if obs_mask.sum() > 0:
            X_filled[~obs_mask, j] = X_observed[obs_mask, j].mean()
        else:
            X_filled[:, j] = protein_base[j]

    meta = pd.DataFrame(
        {
            "sample_id": [f"S{i:03d}" for i in range(n_samples)],
            "batch": batch_labels,
            "cell_line": cell_line_labels,
            "run_order": run_order,
            "sample_quality": sample_quality,
            "is_outlier": np.isin(np.arange(n_samples), outlier_samples).astype(int),
        }
    )

    protein_ids = [f"PROT_{j:04d}" for j in range(n_proteins)]

    return (
        X_filled.astype(np.float32),
        M.astype(np.float32),
        meta,
        protein_ids,
        true_X_clean.astype(np.float32),
    )
