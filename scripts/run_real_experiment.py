#!/usr/bin/env python3
"""Run real-data proteomics batch-correction experiments.

This script has no dependency on synthetic data. It can load either:
1. Recursive search_results/protein.tsv files, as in Ian's CyVerse dataset.
2. A generic abundance matrix plus metadata file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import optuna
import pandas as pd
import torch

from batchae.baselines import baseline_grid
from batchae.data import (
    assert_experiment_ready,
    load_matrix_with_metadata,
    load_protein_tsv_directory,
    preprocess_abundance_matrix,
)
from batchae.metrics import (
    classifier_scores,
    classifier_scores_fast,
    constrained_batch_floor,
    evaluate_correction,
    print_compact_results,
)
from batchae.model import ProteomicsBatchAE
from batchae.threading import configure_threads
from batchae.training import build_torch_dataset, correct_with_model, train_model


def parse_args():
    p = argparse.ArgumentParser(description="Real-data proteomics batch correction experiment")

    # Input options
    p.add_argument("--data-root", default="/iplant/home/shared/NCEMS/PPA/TestDatasets",
                   help="Root containing recursive search_results/protein.tsv files.")
    p.add_argument("--pattern", default="**/search_results/*/protein.tsv")
    p.add_argument("--matrix", default=None, help="Optional generic abundance matrix path.")
    p.add_argument("--metadata", default=None, help="Optional metadata CSV/TSV path.")
    p.add_argument("--orientation", choices=["samples_rows", "proteins_rows"], default="samples_rows")
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--batch-col", default="batch")
    p.add_argument("--biology-col", default="biology")
    p.add_argument("--protein-id-col", default=None)
    p.add_argument("--abundance-col", default=None, help="Default auto-detects Razor intensity.")
    p.add_argument("--drop-unknown-biology", action="store_true")

    # Preprocessing
    p.add_argument("--min-present-frac", type=float, default=0.2)
    p.add_argument("--max-missing-frac", type=float, default=None)
    p.add_argument("--no-log2", action="store_true")
    p.add_argument("--impute", choices=["median", "mean", "zero"], default="median")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--zero-as-missing", action=argparse.BooleanOptionalAction, default=True)

    # AE/model/search
    p.add_argument("--ae-trials", type=int, default=30)
    p.add_argument("--trial-epochs", type=int, default=150)
    p.add_argument("--final-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--biology-drop-tolerance", type=float, default=0.10,
                   help="Allowed drop in biology balanced accuracy before objective penalty.")
    p.add_argument("--distortion-weight", type=float, default=0.05,
                   help="Weak penalty on observed-value RMSE change.")

    # CPU controls
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--interop-threads", type=int, default=1)
    p.add_argument("--dataloader-workers", type=int, default=0)
    p.add_argument("--optuna-jobs", type=int, default=1)
    p.add_argument("--sklearn-jobs", type=int, default=1)

    # Outputs
    p.add_argument("--outdir", default="results/real_run")
    p.add_argument("--save-corrected", action="store_true")
    p.add_argument("--quiet-optuna", action="store_true")

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return requested


def load_dataset_from_args(args):
    if args.matrix:
        if not args.metadata:
            raise ValueError("--metadata is required with --matrix")
        abundance, meta = load_matrix_with_metadata(
            args.matrix,
            args.metadata,
            sample_col=args.sample_col,
            batch_col=args.batch_col,
            biology_col=args.biology_col,
            orientation=args.orientation,
            zero_as_missing=args.zero_as_missing,
        )
    else:
        abundance, meta = load_protein_tsv_directory(
            args.data_root,
            pattern=args.pattern,
            abundance_col=args.abundance_col,
            protein_id_col=args.protein_id_col,
            zero_as_missing=args.zero_as_missing,
            metadata_path=args.metadata,
        )

    ds = preprocess_abundance_matrix(
        abundance,
        meta,
        min_present_frac=args.min_present_frac,
        max_missing_frac=args.max_missing_frac,
        log2_transform=not args.no_log2,
        impute_strategy=args.impute,
        standardize=not args.no_standardize,
        drop_unknown_biology=args.drop_unknown_biology,
        biology_col=args.biology_col,
    )
    assert_experiment_ready(ds.meta, args.batch_col, args.biology_col)
    return ds


def make_model(trial, ds, encoded, args, svd_cache=None):
    batch_dim = trial.suggest_int("batch_dim", 4, 14)
    biology_dim = trial.suggest_int("biology_dim", 2, 10)
    extra_dim = trial.suggest_int("extra_dim", 8, 32)
    latent_dim = batch_dim + biology_dim + extra_dim
    hidden_classifier_dim = trial.suggest_categorical("hidden_classifier_dim", [16, 32, 64])

    model = ProteomicsBatchAE(
        n_features=ds.X.shape[1],
        n_covariates=encoded.covariate_dim,
        n_batches=len(encoded.batch_classes),
        n_biology_classes=len(encoded.biology_classes),
        latent_dim=latent_dim,
        batch_dim=batch_dim,
        biology_dim=biology_dim,
        hidden_classifier_dim=hidden_classifier_dim,
        variational=False,
    )
    if svd_cache is None:
        model.init_weights_svd(ds.X - ds.X.mean(axis=0, keepdims=True))
    else:
        model.init_weights_svd(svd_cache=svd_cache)
    return model


def objective_factory(ds, encoded, args, device, raw_batch_bacc, raw_bio_bacc, svd_cache):
    def objective(trial):
        set_seed(args.seed + trial.number)
        hparams = {
            "lambda_missing": trial.suggest_float("lambda_missing", 0.01, 1.0, log=True),
            "eta_batch": trial.suggest_float("eta_batch", 0.2, 25.0, log=True),
            "eta_biology": trial.suggest_float("eta_biology", 0.2, 25.0, log=True),
            "eta_adv": trial.suggest_float("eta_adv", 0.05, 25.0, log=True),
            "gamma_indep": trial.suggest_float("gamma_indep", 0.001, 1.0, log=True),
        }
        model = make_model(trial, ds, encoded, args, svd_cache=svd_cache).to(device)
        train_model(
            model,
            encoded,
            hparams,
            n_epochs=args.trial_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            dataloader_workers=args.dataloader_workers,
            verbose=False,
        )
        X_corr, _ = correct_with_model(model, ds.X, ds.M, encoded, device=device)

        batch_scores = classifier_scores_fast(
            X_corr,
            ds.meta[args.batch_col].values,
            cv=5,
            n_jobs=args.sklearn_jobs,
            random_state=args.seed,
        )
        batch_bacc = batch_scores["balanced_accuracy"]

        # Penalize biology collapse if biology labels are available.
        bio_penalty = 0.0
        if args.biology_col in ds.meta.columns and ds.meta[args.biology_col].nunique() >= 2:
            bio_scores = classifier_scores_fast(
                X_corr,
                ds.meta[args.biology_col].values,
                cv=5,
                n_jobs=args.sklearn_jobs,
                random_state=args.seed,
            )
            bio_bacc = bio_scores["balanced_accuracy"]
            min_allowed = raw_bio_bacc - args.biology_drop_tolerance
            if not np.isnan(bio_bacc) and bio_bacc < min_allowed:
                bio_penalty = 5.0 * (min_allowed - bio_bacc)

        obs = ds.M.astype(bool)
        distortion = float(np.sqrt(np.mean((X_corr[obs] - ds.X[obs]) ** 2))) if obs.sum() else 0.0

        if np.isnan(batch_bacc):
            return 999.0
        return float(batch_bacc + bio_penalty + args.distortion_weight * distortion)

    return objective


def main():
    args = parse_args()
    configure_threads(args.torch_threads, args.interop_threads)
    set_seed(args.seed)
    device = choose_device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quiet_optuna:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n=== Configuration ===")
    print(json.dumps(vars(args), indent=2))
    print(f"Device: {device}")

    print("\n=== Loading data ===")
    ds = load_dataset_from_args(args)
    encoded = build_torch_dataset(ds.X, ds.M, ds.meta, args.batch_col, args.biology_col)

    print("[svd] Computing SVD cache once for all AE trials...", flush=True)
    _, svd_s, svd_vt = np.linalg.svd(ds.X - ds.X.mean(axis=0, keepdims=True), full_matrices=False)
    svd_cache = {"S": svd_s, "Vt": svd_vt}

    print(f"X shape: {ds.X.shape}")
    print(f"Missing rate: {1.0 - ds.M.mean():.1%}")
    print(f"Batches ({len(encoded.batch_classes)}): {encoded.batch_classes}")
    print(f"Biology classes ({len(encoded.biology_classes)}): {encoded.biology_classes}")
    print("\nBiology x Batch:")
    if args.biology_col in ds.meta.columns:
        print(pd.crosstab(ds.meta[args.biology_col], ds.meta[args.batch_col]).to_string())

    ds.meta.to_csv(outdir / "metadata_used.csv", index=False)
    pd.Series(ds.protein_ids, name="protein_id").to_csv(outdir / "protein_ids.csv", index=False)

    print("\n=== Raw metrics ===")
    raw_batch = classifier_scores(ds.X, ds.meta[args.batch_col].values, cv=5, n_jobs=args.sklearn_jobs, random_state=args.seed)
    raw_bio = {"balanced_accuracy": np.nan}
    if args.biology_col in ds.meta.columns and ds.meta[args.biology_col].nunique() >= 2:
        raw_bio = classifier_scores(ds.X, ds.meta[args.biology_col].values, cv=5, n_jobs=args.sklearn_jobs, random_state=args.seed)
    print(f"Raw batch balanced accuracy:   {raw_batch['balanced_accuracy']:.3f} | chance={raw_batch['chance']:.3f}")
    if args.biology_col in ds.meta.columns and ds.meta[args.biology_col].nunique() >= 2:
        floor = constrained_batch_floor(ds.meta[args.batch_col].values, ds.meta[args.biology_col].values)
        if not np.isnan(floor):
            print(f"Constrained batch-bAcc floor if biology is preserved: {floor:.3f}")
    if not np.isnan(raw_bio["balanced_accuracy"]):
        print(f"Raw biology balanced accuracy: {raw_bio['balanced_accuracy']:.3f}")

    print("\n=== Running baselines ===")
    all_results = []
    baseline_outputs = baseline_grid(ds.X, ds.protein_ids, ds.meta, args.batch_col, args.biology_col)
    for name, X_corr in baseline_outputs.items():
        result = evaluate_correction(
            name,
            ds.X,
            X_corr,
            ds.M,
            ds.meta,
            batch_col=args.batch_col,
            biology_col=args.biology_col,
            cv=5,
            n_jobs=args.sklearn_jobs,
            random_state=args.seed,
        )
        all_results.append(result)

    print_compact_results(all_results)

    print("\n=== Tuning adversarial autoencoder ===")
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = objective_factory(
        ds,
        encoded,
        args,
        device,
        raw_batch_bacc=raw_batch["balanced_accuracy"],
        raw_bio_bacc=raw_bio["balanced_accuracy"],
        svd_cache=svd_cache,
    )
    start = time.time()
    study.optimize(objective, n_trials=args.ae_trials, n_jobs=args.optuna_jobs, show_progress_bar=True)
    elapsed = time.time() - start

    print("\nBest AE score:", study.best_value)
    print("Best AE params:")
    print(json.dumps(study.best_params, indent=2))

    with open(outdir / "optuna_best.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params, "elapsed_sec": elapsed}, f, indent=2)
    study.trials_dataframe().to_csv(outdir / "optuna_trials.csv", index=False)

    print("\n=== Training final AE ===")
    fixed = optuna.trial.FixedTrial(study.best_params)
    final_model = make_model(fixed, ds, encoded, args, svd_cache=svd_cache).to(device)
    hparams = {
        "lambda_missing": study.best_params["lambda_missing"],
        "eta_batch": study.best_params["eta_batch"],
        "eta_biology": study.best_params["eta_biology"],
        "eta_adv": study.best_params["eta_adv"],
        "gamma_indep": study.best_params["gamma_indep"],
    }
    history = train_model(
        final_model,
        encoded,
        hparams,
        n_epochs=args.final_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        dataloader_workers=args.dataloader_workers,
        verbose=True,
    )
    pd.DataFrame(history).to_csv(outdir / "ae_training_history.csv", index=False)

    X_ae, z = correct_with_model(final_model, ds.X, ds.M, encoded, device=device)
    ae_result = evaluate_correction(
        "adversarial_ae",
        ds.X,
        X_ae,
        ds.M,
        ds.meta,
        batch_col=args.batch_col,
        biology_col=args.biology_col,
        cv=5,
        n_jobs=args.sklearn_jobs,
        random_state=args.seed,
    )
    all_results.append(ae_result)

    print("\n=== Final comparison ===")
    print_compact_results(all_results)
    pd.DataFrame(all_results).to_csv(outdir / "comparison_metrics.csv", index=False)
    np.save(outdir / "ae_latent.npy", z)

    if args.save_corrected:
        corrected_df = pd.DataFrame(X_ae, index=ds.meta["sample_id"].values, columns=ds.protein_ids)
        corrected_df.to_csv(outdir / "ae_corrected_matrix.csv")
        for name, X_corr in baseline_outputs.items():
            if name == "raw":
                continue
            pd.DataFrame(X_corr, index=ds.meta["sample_id"].values, columns=ds.protein_ids).to_csv(outdir / f"{name}_corrected_matrix.csv")

    torch.save(final_model.state_dict(), outdir / "ae_model.pt")
    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
