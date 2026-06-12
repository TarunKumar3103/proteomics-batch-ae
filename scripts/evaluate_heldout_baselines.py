#!/usr/bin/env python3
"""Validate leakage-safe held-out evaluation for simple baselines.

This script is intentionally separate from run_real_experiment.py. It is meant
for debugging/validation of the 0.000 batch-bAcc artifact in centering and OLS
baselines. It does not retrain the adversarial AE and does not try to make ComBat
look held-out; ComBat is usually transductive and should be reported with that
caveat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from batchae.data import (
    assert_experiment_ready,
    load_matrix_with_metadata,
    load_protein_tsv_directory,
    preprocess_abundance_matrix,
)
from batchae.heldout_eval import flag_result, heldout_baseline_grid
from batchae.metrics import constrained_batch_floor
from batchae.threading import configure_threads


def parse_args():
    p = argparse.ArgumentParser(description="Leakage-safe held-out baseline evaluation")
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
    p.add_argument("--min-present-frac", type=float, default=0.2)
    p.add_argument("--max-missing-frac", type=float, default=None)
    p.add_argument("--no-log2", action="store_true")
    p.add_argument("--impute", choices=["median", "mean", "zero"], default="median")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--zero-as-missing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-iter", type=int, default=10000)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--interop-threads", type=int, default=1)
    p.add_argument("--outdir", default="results/heldout_baseline_eval")
    p.add_argument("--save-corrected", action="store_true", help="Save out-of-fold corrected matrices.")
    return p.parse_args()


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


def print_table(df: pd.DataFrame) -> None:
    cols = [
        "method",
        "batch_bacc",
        "batch_acc",
        "batch_chance",
        "constrained_batch_floor",
        "bio_bacc",
        "bio_acc",
        "observed_rmse_change",
        "flag",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))


def main():
    args = parse_args()
    configure_threads(args.torch_threads, args.interop_threads)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n=== Configuration ===")
    print(json.dumps(vars(args), indent=2))

    print("\n=== Loading data ===")
    ds = load_dataset_from_args(args)
    print(f"X shape: {ds.X.shape}")
    print(f"Missing rate: {1.0 - ds.M.mean():.1%}")
    print(f"Batches ({ds.meta[args.batch_col].nunique()}): {sorted(ds.meta[args.batch_col].astype(str).unique())}")
    if args.biology_col in ds.meta.columns:
        print(f"Biology classes ({ds.meta[args.biology_col].nunique()}): {sorted(ds.meta[args.biology_col].astype(str).unique())}")
        print("\nBiology x Batch:")
        print(pd.crosstab(ds.meta[args.biology_col], ds.meta[args.batch_col]).to_string())
        floor = constrained_batch_floor(ds.meta[args.batch_col].values, ds.meta[args.biology_col].values)
        if np.isfinite(floor):
            print(f"\nConstrained batch-bAcc floor if biology is preserved: {floor:.3f}")

    print("\n=== Held-out correction + held-out probe evaluation ===")
    print("Each outer fold fits correction on train rows only, transforms test rows, trains probe on corrected train rows, and predicts corrected test rows.")
    results, matrices = heldout_baseline_grid(
        ds,
        batch_col=args.batch_col,
        biology_col=args.biology_col,
        cv=args.cv,
        random_state=args.seed,
        max_iter=args.max_iter,
    )
    for row in results:
        row["flag"] = flag_result(row)

    df = pd.DataFrame(results)
    print("\n=== Held-out baseline comparison ===")
    print_table(df)
    df.to_csv(outdir / "heldout_baseline_metrics.csv", index=False)
    ds.meta.to_csv(outdir / "metadata_used.csv", index=False)
    pd.Series(ds.protein_ids, name="protein_id").to_csv(outdir / "protein_ids.csv", index=False)

    if args.save_corrected:
        for name, X_corr in matrices.items():
            pd.DataFrame(X_corr, index=ds.meta["sample_id"].values, columns=ds.protein_ids).to_csv(
                outdir / f"{name}_heldout_corrected_matrix.csv"
            )

    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
