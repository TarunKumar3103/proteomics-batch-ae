#!/usr/bin/env python3
"""Run a batch/biology Pareto AE search and optional held-out AE evaluation.

This runner is separate from ``run_real_experiment.py``. The regular runner
selects a model through one blended scalar objective. Here, the search exposes
the two scientific objectives separately:

    1. minimize residual batch predictability (batch balanced accuracy),
    2. minimize biology loss (1 - biology balanced accuracy).

Observed-value RMSE change is still saved for every trial and used as a point
color / optional selection constraint, but it is not a Pareto objective. If RMSE
is minimized as a third objective, no-correction or barely-correction models can
look Pareto-optimal simply because they do not move the matrix.

The selected in-trial point is only an operating point. The script retrains and
re-evaluates that selected configuration. If ``--evaluate-heldout-ae`` is used,
it also performs a leakage-safe outer evaluation:

    for each fold:
        train a fresh AE on train rows only,
        correct held-out test rows,
        train probes on corrected train rows,
        predict corrected held-out rows.

This puts the selected AE on the same held-out protocol used for OLS/centering
baselines, at the cost of training one AE per fold.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

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
from batchae.heldout_eval import evaluate_ae_heldout, flag_result, heldout_baseline_grid
from batchae.metrics import (
    classifier_scores,
    classifier_scores_fast,
    constrained_batch_floor,
    evaluate_correction,
    print_compact_results,
)
from batchae.model import ProteomicsBatchAE
from batchae.pareto import add_pareto_columns, optuna_trials_to_frame, plot_pareto_2d, select_recommended_trial
from batchae.threading import configure_threads
from batchae.training import build_torch_dataset, correct_with_model, train_model


def parse_args():
    p = argparse.ArgumentParser(description="Batch/biology Pareto AE experiment")

    # Input options: mirrored from run_real_experiment.py.
    p.add_argument("--data-root", default="/iplant/home/shared/NCEMS/PPA/TestDatasets")
    p.add_argument("--pattern", default="**/search_results/*/protein.tsv")
    p.add_argument("--matrix", default=None)
    p.add_argument("--metadata", default=None)
    p.add_argument("--orientation", choices=["samples_rows", "proteins_rows"], default="samples_rows")
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--batch-col", default="batch")
    p.add_argument("--biology-col", default="biology")
    p.add_argument("--protein-id-col", default=None)
    p.add_argument("--abundance-col", default=None)
    p.add_argument("--drop-unknown-biology", action="store_true")

    # Preprocessing.
    p.add_argument("--min-present-frac", type=float, default=0.2)
    p.add_argument("--max-missing-frac", type=float, default=None)
    p.add_argument("--no-log2", action="store_true")
    p.add_argument("--impute", choices=["median", "mean", "zero"], default="median")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--zero-as-missing", action=argparse.BooleanOptionalAction, default=True)

    # Search/model.
    p.add_argument("--ae-trials", type=int, default=75)
    p.add_argument("--trial-epochs", type=int, default=150)
    p.add_argument("--final-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--sampler", choices=["nsga2", "tpe"], default="nsga2",
                   help="Multi-objective sampler. nsga2 is a safe default; tpe uses Optuna's MOTPE support.")

    # Final model selection from the trial set.
    p.add_argument("--select-bio-min", type=float, default=0.98,
                   help="Select the lowest-batch final model among trials with bio_bacc >= this value.")
    p.add_argument("--select-rmse-max", type=float, default=float("inf"),
                   help="Optional selection cap on observed-value RMSE change.")
    p.add_argument("--skip-final-training", action="store_true",
                   help="Only run the Pareto search/plot; do not train the selected final AE.")

    # Baseline overlays.
    p.add_argument("--include-heldout-baselines", action=argparse.BooleanOptionalAction, default=True,
                   help="Also compute leakage-safe held-out centering/OLS points for the plot/table.")
    p.add_argument("--heldout-cv", type=int, default=5)

    # Leakage-safe held-out AE evaluation.
    p.add_argument("--evaluate-heldout-ae", action=argparse.BooleanOptionalAction, default=False,
                   help="Train AE on train folds only and evaluate corrected held-out rows. Expensive but protocol-clean.")
    p.add_argument("--heldout-ae-epochs", type=int, default=0,
                   help="Epochs for each held-out AE fold. 0 means use --final-epochs.")
    p.add_argument("--heldout-ae-verbose", action="store_true",
                   help="Print epoch logs for each held-out AE fold. Default prints fold starts only.")

    # CPU controls.
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--interop-threads", type=int, default=1)
    p.add_argument("--dataloader-workers", type=int, default=0)
    p.add_argument("--optuna-jobs", type=int, default=4)
    p.add_argument("--sklearn-jobs", type=int, default=1)

    # Outputs.
    p.add_argument("--outdir", default="results/pareto_run")
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


def _params_from_trial(trial) -> dict:
    return {
        "lambda_missing": trial.suggest_float("lambda_missing", 0.01, 1.0, log=True),
        "eta_batch": trial.suggest_float("eta_batch", 0.2, 25.0, log=True),
        "eta_biology": trial.suggest_float("eta_biology", 0.2, 25.0, log=True),
        "eta_adv": trial.suggest_float("eta_adv", 0.05, 25.0, log=True),
        "gamma_indep": trial.suggest_float("gamma_indep", 0.001, 1.0, log=True),
        "batch_dim": trial.suggest_int("batch_dim", 4, 14),
        "biology_dim": trial.suggest_int("biology_dim", 2, 10),
        "extra_dim": trial.suggest_int("extra_dim", 8, 32),
        "hidden_classifier_dim": trial.suggest_categorical("hidden_classifier_dim", [16, 32, 64]),
    }


def make_model_from_params(params: dict, n_features: int, encoded, X_for_svd: np.ndarray | None = None, svd_cache=None):
    batch_dim = int(params["batch_dim"])
    biology_dim = int(params["biology_dim"])
    extra_dim = int(params["extra_dim"])
    latent_dim = batch_dim + biology_dim + extra_dim
    model = ProteomicsBatchAE(
        n_features=n_features,
        n_covariates=encoded.covariate_dim,
        n_batches=len(encoded.batch_classes),
        n_biology_classes=len(encoded.biology_classes),
        latent_dim=latent_dim,
        batch_dim=batch_dim,
        biology_dim=biology_dim,
        hidden_classifier_dim=int(params["hidden_classifier_dim"]),
        variational=False,
    )
    if svd_cache is not None:
        model.init_weights_svd(svd_cache=svd_cache)
    elif X_for_svd is not None:
        model.init_weights_svd(X_for_svd - X_for_svd.mean(axis=0, keepdims=True))
    else:
        raise ValueError("Need either X_for_svd or svd_cache for model initialization")
    return model


def _hparams(params: dict) -> dict:
    return {
        "lambda_missing": float(params["lambda_missing"]),
        "eta_batch": float(params["eta_batch"]),
        "eta_biology": float(params["eta_biology"]),
        "eta_adv": float(params["eta_adv"]),
        "gamma_indep": float(params["gamma_indep"]),
    }


def objective_factory(ds, encoded, args, device, svd_cache):
    obs = ds.M.astype(bool)
    has_bio = args.biology_col in ds.meta.columns and ds.meta[args.biology_col].nunique() >= 2

    def objective(trial):
        set_seed(args.seed + trial.number)
        params = _params_from_trial(trial)
        model = make_model_from_params(params, ds.X.shape[1], encoded, svd_cache=svd_cache).to(device)
        train_model(
            model,
            encoded,
            _hparams(params),
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
        if not np.isfinite(batch_bacc):
            batch_bacc = 999.0

        if has_bio:
            bio_scores = classifier_scores_fast(
                X_corr,
                ds.meta[args.biology_col].values,
                cv=5,
                n_jobs=args.sklearn_jobs,
                random_state=args.seed,
            )
            bio_bacc = bio_scores["balanced_accuracy"]
            if not np.isfinite(bio_bacc):
                bio_bacc = 0.0
        else:
            bio_bacc = 0.0
        bio_loss = 1.0 - float(bio_bacc)
        rmse = float(np.sqrt(np.mean((X_corr[obs] - ds.X[obs]) ** 2))) if obs.sum() else 0.0

        trial.set_user_attr("batch_bacc", float(batch_bacc))
        trial.set_user_attr("bio_bacc", float(bio_bacc))
        trial.set_user_attr("bio_loss", float(bio_loss))
        trial.set_user_attr("rmse_change", float(rmse))
        return float(batch_bacc), float(bio_loss)

    return objective


def _make_sampler(args):
    if args.sampler == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=args.seed)
    return optuna.samplers.TPESampler(seed=args.seed)


def _baseline_overlay_from_results(all_results: list[dict], heldout_df: pd.DataFrame | None, heldout_ae_result: dict | None = None) -> pd.DataFrame:
    rows = []
    for r in all_results:
        method = r.get("method")
        if method not in {"raw", "combat"}:
            continue
        rows.append({
            "method": method,
            "protocol": "transductive",
            "batch_bacc": r.get("corr_batch_balanced_accuracy"),
            "bio_bacc": r.get("corr_biology_balanced_accuracy"),
            "rmse_change": r.get("observed_rmse_change"),
        })
    if heldout_df is not None and not heldout_df.empty:
        for _, r in heldout_df.iterrows():
            method = str(r.get("method"))
            if method in {"ols_preserve_biology", "batch_mean_center", "ols_remove_batch"}:
                rows.append({
                    "method": f"{method} (heldout)",
                    "protocol": "heldout",
                    "batch_bacc": r.get("batch_bacc"),
                    "bio_bacc": r.get("bio_bacc"),
                    "rmse_change": r.get("observed_rmse_change"),
                })
    if heldout_ae_result is not None:
        rows.append({
            "method": "AE selected (heldout)",
            "protocol": "heldout_ae",
            "batch_bacc": heldout_ae_result.get("batch_bacc"),
            "bio_bacc": heldout_ae_result.get("bio_bacc"),
            "rmse_change": heldout_ae_result.get("observed_rmse_change"),
        })
    return pd.DataFrame(rows)


def _params_from_selected_row(row: pd.Series) -> dict:
    params = {}
    prefix = "param_"
    for k, v in row.items():
        if str(k).startswith(prefix):
            name = str(k)[len(prefix):]
            if name in {"batch_dim", "biology_dim", "extra_dim", "hidden_classifier_dim"}:
                params[name] = int(v)
            else:
                params[name] = float(v)
    return params


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
    floor = np.nan
    if args.biology_col in ds.meta.columns and ds.meta[args.biology_col].nunique() >= 2:
        print("\nBiology x Batch:")
        print(pd.crosstab(ds.meta[args.biology_col], ds.meta[args.batch_col]).to_string())
        floor = constrained_batch_floor(ds.meta[args.batch_col].values, ds.meta[args.biology_col].values)
        if np.isfinite(floor):
            print(f"Constrained batch-bAcc floor if biology is preserved: {floor:.3f}")

    ds.meta.to_csv(outdir / "metadata_used.csv", index=False)
    pd.Series(ds.protein_ids, name="protein_id").to_csv(outdir / "protein_ids.csv", index=False)

    print("\n=== Raw metrics ===")
    raw_batch = classifier_scores(ds.X, ds.meta[args.batch_col].values, cv=5, n_jobs=args.sklearn_jobs, random_state=args.seed)
    print(f"Raw batch balanced accuracy: {raw_batch['balanced_accuracy']:.3f} | chance={raw_batch['chance']:.3f}")
    if args.biology_col in ds.meta.columns and ds.meta[args.biology_col].nunique() >= 2:
        raw_bio = classifier_scores(ds.X, ds.meta[args.biology_col].values, cv=5, n_jobs=args.sklearn_jobs, random_state=args.seed)
        print(f"Raw biology balanced accuracy: {raw_bio['balanced_accuracy']:.3f}")

    print("\n=== Baseline overlay points ===")
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
    pd.DataFrame(all_results).to_csv(outdir / "transductive_baseline_metrics.csv", index=False)

    heldout_df = None
    if args.include_heldout_baselines:
        print("\n=== Held-out baseline overlay points ===")
        heldout_results, _ = heldout_baseline_grid(
            ds,
            batch_col=args.batch_col,
            biology_col=args.biology_col,
            cv=args.heldout_cv,
            random_state=args.seed,
        )
        for row in heldout_results:
            row["flag"] = flag_result(row)
        heldout_df = pd.DataFrame(heldout_results)
        heldout_df.to_csv(outdir / "heldout_baseline_metrics.csv", index=False)
        print(heldout_df[["method", "batch_bacc", "bio_bacc", "observed_rmse_change", "flag"]].to_string(index=False))

    print("\n=== Multi-objective AE search ===")
    sampler = _make_sampler(args)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    objective = objective_factory(ds, encoded, args, device, svd_cache)
    start = time.time()
    study.optimize(objective, n_trials=args.ae_trials, n_jobs=args.optuna_jobs, show_progress_bar=True)
    elapsed = time.time() - start

    trials_df = optuna_trials_to_frame(study)
    trials_df = add_pareto_columns(trials_df, objectives=("batch_bacc", "bio_loss"))
    trials_df.to_csv(outdir / "ae_multiobjective_trials.csv", index=False)
    trials_df[trials_df["is_pareto"]].to_csv(outdir / "ae_pareto_trials.csv", index=False)

    selected = select_recommended_trial(
        trials_df,
        min_bio_bacc=args.select_bio_min,
        max_rmse_change=args.select_rmse_max,
    )
    selected_params = _params_from_selected_row(selected)
    selected_summary = {
        "selected_trial": int(selected["trial"]),
        "selection_rule": "operating point only: min batch_bacc among trials satisfying bio_bacc/rmse constraints; fallback lexicographic",
        "reporting_note": "Do not report these in-trial metrics as final performance; report retrained comparison_metrics.csv and heldout_ae_metrics.csv instead.",
        "select_bio_min": args.select_bio_min,
        "select_rmse_max": None if not np.isfinite(args.select_rmse_max) else args.select_rmse_max,
        "batch_bacc": float(selected["batch_bacc"]),
        "bio_bacc": float(selected["bio_bacc"]),
        "bio_loss": float(selected["bio_loss"]),
        "rmse_change": float(selected["rmse_change"]),
        "params": selected_params,
        "elapsed_sec": elapsed,
    }
    with open(outdir / "selected_trial.json", "w") as f:
        json.dump(selected_summary, f, indent=2)

    print("\nSelected practical AE trial:")
    print(json.dumps(selected_summary, indent=2))

    baseline_points = _baseline_overlay_from_results(all_results, heldout_df)
    baseline_points.to_csv(outdir / "pareto_baseline_points.csv", index=False)
    plot_pareto_2d(
        trials_df,
        outdir / "ae_pareto_front_search.png",
        baseline_df=baseline_points,
        constrained_floor=floor,
        title="AE batch/biology Pareto frontier vs baselines",
    )
    print(f"Saved search Pareto plot to: {outdir / 'ae_pareto_front_search.png'}")

    if args.skip_final_training:
        # Also save the canonical plot name for validation-only runs.
        plot_pareto_2d(
            trials_df,
            outdir / "ae_pareto_front.png",
            baseline_df=baseline_points,
            constrained_floor=floor,
            title="AE batch/biology Pareto frontier vs baselines",
        )
        print(f"\nSaved outputs to: {outdir.resolve()}")
        return

    print("\n=== Training final selected AE ===")
    final_model = make_model_from_params(selected_params, ds.X.shape[1], encoded, svd_cache=svd_cache).to(device)
    hparams = _hparams(selected_params)
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
    pd.DataFrame(history).to_csv(outdir / "selected_ae_training_history.csv", index=False)

    X_ae, z = correct_with_model(final_model, ds.X, ds.M, encoded, device=device)
    ae_result = evaluate_correction(
        "adversarial_ae_selected_from_pareto",
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
    final_results = all_results + [ae_result]
    pd.DataFrame(final_results).to_csv(outdir / "comparison_metrics.csv", index=False)
    print("\n=== Final selected-AE transductive comparison ===")
    print_compact_results(final_results)

    heldout_ae_result = None
    if args.evaluate_heldout_ae:
        print("\n=== Held-out selected-AE evaluation ===")
        heldout_epochs = args.heldout_ae_epochs if args.heldout_ae_epochs > 0 else args.final_epochs

        def make_fold_model(encoded_train, X_train, M_train, meta_train, fold):
            set_seed(args.seed + 10000 + fold)
            return make_model_from_params(selected_params, X_train.shape[1], encoded_train, X_for_svd=X_train, svd_cache=None)

        heldout_ae_result, X_ae_oof, z_ae_oof = evaluate_ae_heldout(
            ds,
            make_model_fn=make_fold_model,
            hparams=hparams,
            batch_col=args.batch_col,
            biology_col=args.biology_col,
            cv=args.heldout_cv,
            random_state=args.seed,
            n_epochs=heldout_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            dataloader_workers=args.dataloader_workers,
            verbose=args.heldout_ae_verbose,
        )
        heldout_ae_result["flag"] = flag_result(heldout_ae_result)
        pd.DataFrame([heldout_ae_result]).to_csv(outdir / "heldout_ae_metrics.csv", index=False)
        np.save(outdir / "heldout_ae_latent_oof.npy", z_ae_oof)
        if args.save_corrected:
            pd.DataFrame(X_ae_oof, index=ds.meta["sample_id"].values, columns=ds.protein_ids).to_csv(outdir / "heldout_ae_corrected_oof_matrix.csv")
        print(pd.DataFrame([heldout_ae_result]).to_string(index=False))

    baseline_points = _baseline_overlay_from_results(all_results, heldout_df, heldout_ae_result=heldout_ae_result)
    baseline_points.to_csv(outdir / "pareto_baseline_points.csv", index=False)
    plot_pareto_2d(
        trials_df,
        outdir / "ae_pareto_front.png",
        baseline_df=baseline_points,
        constrained_floor=floor,
        title="AE batch/biology Pareto frontier vs held-out baselines" if heldout_ae_result else "AE batch/biology Pareto frontier vs baselines",
    )
    print(f"Saved final Pareto plot to: {outdir / 'ae_pareto_front.png'}")

    np.save(outdir / "selected_ae_latent.npy", z)
    torch.save(final_model.state_dict(), outdir / "selected_ae_model.pt")
    if args.save_corrected:
        pd.DataFrame(X_ae, index=ds.meta["sample_id"].values, columns=ds.protein_ids).to_csv(outdir / "selected_ae_corrected_matrix.csv")

    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
