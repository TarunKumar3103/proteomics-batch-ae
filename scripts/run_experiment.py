#!/usr/bin/env python
"""Run fair synthetic proteomics batch-correction experiments.

Important: this script parses thread arguments and configures BLAS/PyTorch before
importing most scientific modules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running directly from repo root without pip install -e .
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Fair autoencoder vs tuned baselines on synthetic proteomics data")

    p.add_argument("--outdir", type=str, default="results/run")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # CPU controls
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--interop-threads", type=int, default=1)
    p.add_argument("--dataloader-workers", type=int, default=0)
    p.add_argument("--optuna-jobs", type=int, default=1)
    p.add_argument("--sklearn-jobs", type=int, default=1)

    # Data controls
    p.add_argument("--n-samples", type=int, default=180)
    p.add_argument("--n-proteins", type=int, default=1200)
    p.add_argument("--n-batches", type=int, default=6)
    p.add_argument("--n-cell-lines", type=int, default=3)
    p.add_argument("--batch-effect-scale", type=float, default=2.5)
    p.add_argument("--biology-scale", type=float, default=1.2)
    p.add_argument("--noise-scale", type=float, default=0.8)
    p.add_argument("--confounding-strength", type=float, default=0.65)
    p.add_argument("--missing-rate", type=float, default=0.20)

    # Fair tuning/eval controls
    p.add_argument("--tune-seeds", type=int, nargs="+", default=[123])
    p.add_argument("--test-seeds", type=int, nargs="+", default=[456])
    p.add_argument("--ae-trials", type=int, default=25)
    p.add_argument("--trial-epochs", type=int, default=300)
    p.add_argument("--final-epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--min-cell-acc", type=float, default=0.75)
    p.add_argument("--clean-weight", type=float, default=0.02)
    p.add_argument("--mse-blowup-factor", type=float, default=1.5)
    p.add_argument("--no-combat", action="store_true", help="Skip ComBat configs if package install is problematic")
    p.add_argument("--verbose-train", action="store_true")
    p.add_argument("--seed", type=int, default=42, help="Optuna/random seed")
    return p.parse_args()


def main():
    args = parse_args()

    from batchae.threading import configure_threads

    thread_cfg = configure_threads(
        torch_threads=args.torch_threads,
        interop_threads=args.interop_threads,
        dataloader_workers=args.dataloader_workers,
        optuna_jobs=args.optuna_jobs,
    )

    import numpy as np
    import optuna
    import pandas as pd
    import torch

    from batchae.baselines import apply_baseline, get_baseline_configs, tune_baseline_configs
    from batchae.data import make_hard_proteomics
    from batchae.metrics import ScoreConfig, evaluate_correction, score_metrics, summarize_results
    from batchae.training import (
        AEHyperParams,
        build_dataset,
        correct_model,
        make_model,
        seed_everything,
        suggest_hparams,
        train_model,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed_everything(args.seed)

    print("=== Run config ===")
    print(f"device={device}")
    print(f"thread_cfg={thread_cfg}")
    print(f"tune_seeds={args.tune_seeds}")
    print(f"test_seeds={args.test_seeds}")
    print(f"outdir={outdir}")

    with open(outdir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    def make_dataset(seed: int):
        return make_hard_proteomics(
            n_samples=args.n_samples,
            n_proteins=args.n_proteins,
            n_batches=args.n_batches,
            n_cell_lines=args.n_cell_lines,
            batch_effect_scale=args.batch_effect_scale,
            biology_scale=args.biology_scale,
            noise_scale=args.noise_scale,
            confounding_strength=args.confounding_strength,
            missing_rate=args.missing_rate,
            seed=seed,
        )

    tune_datasets = [make_dataset(s) for s in args.tune_seeds]
    score_cfg = ScoreConfig(
        min_cell_acc=args.min_cell_acc,
        clean_weight=args.clean_weight,
        mse_blowup_factor=args.mse_blowup_factor,
    )

    def eval_fn(X, X_corr, meta, X_clean):
        return evaluate_correction(X, X_corr, meta, X_clean, cv=args.cv, n_jobs=args.sklearn_jobs, seed=args.seed)

    def scoring_fn(metrics):
        return score_metrics(metrics, score_cfg)

    # ------------------------------------------------------------
    # Tune baselines fairly on validation synthetic seeds.
    # ------------------------------------------------------------
    print("\n=== Tuning baselines on validation seeds ===")
    baseline_configs = get_baseline_configs(include_combat=(not args.no_combat))
    best_baseline, baseline_val = tune_baseline_configs(baseline_configs, tune_datasets, eval_fn, scoring_fn)
    baseline_val.to_csv(outdir / "validation_baselines.csv", index=False)
    with open(outdir / "best_baseline_config.json", "w") as f:
        json.dump(best_baseline.asdict(), f, indent=2)
    print(baseline_val[["name", "score_mean", "failed"]].to_string(index=False))
    print(f"Best baseline: {best_baseline.name}")

    # ------------------------------------------------------------
    # Tune autoencoder hparams on the same validation synthetic seeds.
    # ------------------------------------------------------------
    print("\n=== Tuning autoencoder on validation seeds ===")

    def run_ae_once(seed: int, hparams: AEHyperParams, epochs: int, verbose: bool = False):
        seed_everything(seed + args.seed)
        X, M, meta, protein_ids, X_clean = make_dataset(seed)
        dataset = build_dataset(X, M, meta)
        model = make_model(X, meta, hparams, device=device)
        train_model(
            model,
            dataset,
            hparams.asdict(),
            n_epochs=epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            dataloader_workers=args.dataloader_workers,
            verbose=verbose,
        )
        X_corr, _z = correct_model(model, X, M, meta, device=device)
        metrics = eval_fn(X, X_corr, meta, X_clean)
        return metrics, X_corr

    def objective(trial):
        hp = suggest_hparams(trial)
        scores = []
        for seed in args.tune_seeds:
            metrics, _ = run_ae_once(seed, hp, epochs=args.trial_epochs, verbose=False)
            scores.append(scoring_fn(metrics))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.ae_trials, n_jobs=args.optuna_jobs, show_progress_bar=True)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(outdir / "optuna_trials.csv", index=False)

    best_params = study.best_params.copy()
    best_params["latent_dim"] = max(28, best_params["batch_dim"] + best_params["cellline_dim"] + 10)
    best_ae = AEHyperParams(
        lambda_=best_params["lambda_"],
        eta_b=best_params["eta_b"],
        eta_c=best_params["eta_c"],
        eta_adv=best_params["eta_adv"],
        gamma=best_params["gamma"],
        batch_dim=best_params["batch_dim"],
        cellline_dim=best_params["cellline_dim"],
        latent_dim=best_params["latent_dim"],
    )
    with open(outdir / "best_ae_hparams.json", "w") as f:
        json.dump({"best_value": study.best_value, **best_ae.asdict()}, f, indent=2)
    print(f"Best AE value: {study.best_value:.4f}")
    print(f"Best AE hparams: {best_ae.asdict()}")

    # ------------------------------------------------------------
    # Held-out test evaluation. No more tuning here.
    # ------------------------------------------------------------
    print("\n=== Held-out test evaluation ===")
    rows = []
    for seed in args.test_seeds:
        print(f"Test seed {seed}")
        X, M, meta, protein_ids, X_clean = make_dataset(seed)

        # Raw data row: corrected == raw, so corr_* fields describe raw.
        raw_metrics = eval_fn(X, X, meta, X_clean)
        rows.append({"seed": seed, "method": "raw", "config": "raw", **raw_metrics})

        # All baselines, not only selected, so the table is transparent.
        for cfg in baseline_configs:
            try:
                X_corr = apply_baseline(cfg, X, meta, protein_ids=protein_ids)
                metrics = eval_fn(X, X_corr, meta, X_clean)
                rows.append({"seed": seed, "method": cfg.name, "config": json.dumps(cfg.asdict()), **metrics})
            except Exception as exc:
                print(f"  baseline failed: {cfg.name}: {exc}")

        # AE with validation-selected hparams.
        ae_metrics, _ = run_ae_once(seed, best_ae, epochs=args.final_epochs, verbose=args.verbose_train)
        rows.append({"seed": seed, "method": "adversarial_ae", "config": json.dumps(best_ae.asdict()), **ae_metrics})

    results = pd.DataFrame(rows)
    summary = summarize_results(results, group_col="method")
    results.to_csv(outdir / "test_results_by_seed.csv", index=False)
    summary.to_csv(outdir / "test_summary.csv", index=False)

    print("\n=== Test summary ===")
    cols = [
        "method",
        "n",
        "corr_batch_mean",
        "corr_cellline_mean",
        "corr_mse_mean",
        "corr_corr_mean",
    ]
    existing = [c for c in cols if c in summary.columns]
    print(summary[existing].to_string(index=False))
    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
