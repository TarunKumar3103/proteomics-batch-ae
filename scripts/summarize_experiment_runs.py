#!/usr/bin/env python3
"""Aggregate batch-correction result directories across seeds/runs.

Examples
--------
Summarize patched 75-trial AE runs plus held-out OLS baselines:

    python scripts/summarize_experiment_runs.py \
      --results-root results \
      --run-glob "ian_real_75trial_*_patched" \
      --discover-heldout-baselines \
      --outdir results/summary_patched_75trial

This script normalizes two result formats:
- run_real_experiment.py / run_pareto_experiment.py comparison_metrics.csv
- evaluate_heldout_baselines.py heldout_baseline_metrics.csv
- run_pareto_experiment.py heldout_ae_metrics.csv

By default it reports only the methods that are currently meaningful for the
paper story: raw, combat, adversarial AE, and held-out OLS/centering baselines.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from batchae.pareto import normalize_comparison_metrics, normalize_heldout_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Summarize batch-correction runs across seeds")
    p.add_argument("--results-root", default="results")
    p.add_argument("--run-glob", action="append", default=[],
                   help="Glob under results-root for run dirs containing comparison_metrics.csv. Can repeat.")
    p.add_argument("--comparison-csv", action="append", default=[],
                   help="Explicit comparison_metrics.csv path. Can repeat.")
    p.add_argument("--heldout-csv", action="append", default=[],
                   help="Explicit heldout_baseline_metrics.csv path. Can repeat.")
    p.add_argument("--discover-heldout-baselines", action="store_true",
                   help="Also include results-root/**/heldout_baseline_metrics.csv files.")
    p.add_argument("--discover-heldout-ae", action="store_true",
                   help="Also include results-root/**/heldout_ae_metrics.csv files.")
    p.add_argument("--methods", nargs="*", default=[
        "raw",
        "combat",
        "adversarial_ae",
        "adversarial_ae_selected_from_pareto",
        "adversarial_ae_heldout",
        "ols_preserve_biology",
        "batch_mean_center",
        "ols_remove_batch",
    ])
    p.add_argument("--constrained-floor", type=float, default=0.25,
                   help="Used to compute excess residual batch signal above the biology-preserving floor.")
    p.add_argument("--outdir", default="results/run_summary")
    return p.parse_args()


def infer_seed(run_name: str) -> str:
    m = re.search(r"seed(\d+)", run_name)
    return m.group(1) if m else "unknown"


def collect_comparison_paths(args) -> list[Path]:
    root = Path(args.results_root)
    paths = [Path(p) for p in args.comparison_csv]
    for pattern in args.run_glob:
        for d in root.glob(pattern):
            p = d / "comparison_metrics.csv"
            if p.is_file():
                paths.append(p)
    # Preserve order but de-dupe.
    seen = set()
    out = []
    for p in paths:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def collect_heldout_paths(args) -> list[Path]:
    root = Path(args.results_root)
    paths = [Path(p) for p in args.heldout_csv]
    if args.discover_heldout_baselines:
        paths.extend(root.glob("**/heldout_baseline_metrics.csv"))
    if getattr(args, "discover_heldout_ae", False):
        paths.extend(root.glob("**/heldout_ae_metrics.csv"))
    seen = set()
    out = []
    for p in paths:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def clean_method_name(method: str) -> str:
    method = str(method)
    if method == "adversarial_ae_selected_from_pareto":
        return "adversarial_ae"
    if method == "adversarial_ae_heldout":
        return "adversarial_ae_heldout"
    return method


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    comparison_paths = collect_comparison_paths(args)
    heldout_paths = collect_heldout_paths(args)

    for p in comparison_paths:
        if not p.is_file():
            print(f"WARNING: missing comparison CSV: {p}")
            continue
        df = normalize_comparison_metrics(p, run_name=p.parent.name)
        frames.append(df)

    for p in heldout_paths:
        if not p.is_file():
            print(f"WARNING: missing heldout CSV: {p}")
            continue
        df = normalize_heldout_metrics(p, run_name=p.parent.name)
        frames.append(df)

    if not frames:
        raise SystemExit("No result files found. Pass --run-glob, --comparison-csv, or --heldout-csv.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["method"] = all_df["method"].map(clean_method_name)
    all_df["seed"] = all_df["run"].map(infer_seed)

    if args.methods:
        all_df = all_df[all_df["method"].isin(args.methods)].copy()

    floor = float(args.constrained_floor)
    if "constrained_batch_floor" not in all_df.columns:
        all_df["constrained_batch_floor"] = floor
    else:
        all_df["constrained_batch_floor"] = all_df["constrained_batch_floor"].fillna(floor)
    all_df["batch_excess_above_floor"] = all_df["batch_bacc"].astype(float) - all_df["constrained_batch_floor"].astype(float)
    raw_excess = all_df.loc[all_df["method"] == "raw", ["run", "batch_excess_above_floor"]].rename(
        columns={"batch_excess_above_floor": "raw_excess_above_floor"}
    )
    all_df = all_df.merge(raw_excess, on="run", how="left")
    all_df["fraction_excess_batch_removed"] = np.where(
        all_df["raw_excess_above_floor"] > 0,
        1.0 - (all_df["batch_excess_above_floor"] / all_df["raw_excess_above_floor"]),
        np.nan,
    )

    # Avoid aggregating artifact-prone old full-matrix OLS/centering rows with
    # held-out rows if both are present; prefer held-out protocol for these.
    simple = {"ols_preserve_biology", "batch_mean_center", "ols_remove_batch"}
    simple_heldout_keys = set(
        tuple(x) for x in all_df.loc[
            all_df["method"].isin(simple) & all_df["protocol"].astype(str).str.contains("heldout"),
            ["method"],
        ].drop_duplicates().to_numpy()
    )
    if simple_heldout_keys:
        mask_bad_simple = all_df["method"].isin(simple) & ~all_df["protocol"].astype(str).str.contains("heldout")
        all_df = all_df[~mask_bad_simple].copy()

    per_run_cols = [
        "run", "seed", "method", "protocol", "batch_bacc", "bio_bacc", "rmse_change",
        "constrained_batch_floor", "batch_excess_above_floor", "fraction_excess_batch_removed", "source_file",
    ]
    per_run_cols = [c for c in per_run_cols if c in all_df.columns]
    per_run = all_df[per_run_cols].sort_values(["method", "run"])

    agg = (
        all_df.groupby(["method", "protocol"], dropna=False)
        .agg(
            n_runs=("batch_bacc", "count"),
            batch_bacc_mean=("batch_bacc", "mean"),
            batch_bacc_std=("batch_bacc", "std"),
            bio_bacc_mean=("bio_bacc", "mean"),
            bio_bacc_std=("bio_bacc", "std"),
            rmse_change_mean=("rmse_change", "mean"),
            rmse_change_std=("rmse_change", "std"),
            excess_removed_mean=("fraction_excess_batch_removed", "mean"),
            excess_removed_std=("fraction_excess_batch_removed", "std"),
        )
        .reset_index()
        .sort_values(["method", "protocol"])
    )

    per_run.to_csv(outdir / "per_run_metrics.csv", index=False)
    agg.to_csv(outdir / "aggregate_metrics.csv", index=False)

    print("\n=== Per-run metrics ===")
    display_cols = ["run", "seed", "method", "protocol", "batch_bacc", "bio_bacc", "rmse_change", "fraction_excess_batch_removed"]
    display_cols = [c for c in display_cols if c in per_run.columns]
    with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 160):
        print(per_run[display_cols].to_string(index=False))

    print("\n=== Aggregate metrics ===")
    with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 160):
        print(agg.to_string(index=False))

    summary = {
        "comparison_paths": [str(p) for p in comparison_paths],
        "heldout_paths": [str(p) for p in heldout_paths],
        "constrained_floor": floor,
        "outdir": str(outdir),
    }
    with open(outdir / "summary_inputs.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
