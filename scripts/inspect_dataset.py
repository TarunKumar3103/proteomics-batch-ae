#!/usr/bin/env python3
"""Inspect a real proteomics dataset before running correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from batchae.data import (
    load_protein_tsv_directory,
    load_matrix_with_metadata,
    preprocess_abundance_matrix,
)


def parse_args():
    p = argparse.ArgumentParser(description="Inspect proteomics input files.")
    p.add_argument("--data-root", default=None, help="Root containing recursive search_results/protein.tsv files.")
    p.add_argument("--pattern", default="**/search_results/protein.tsv")
    p.add_argument("--matrix", default=None, help="Generic matrix path, samples x proteins unless --orientation proteins_rows.")
    p.add_argument("--metadata", default=None, help="Metadata CSV/TSV for generic matrix or to override protein.tsv metadata.")
    p.add_argument("--orientation", choices=["samples_rows", "proteins_rows"], default="samples_rows")
    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--batch-col", default="batch")
    p.add_argument("--biology-col", default="biology")
    p.add_argument("--protein-id-col", default=None)
    p.add_argument("--abundance-col", default=None)
    p.add_argument("--min-present-frac", type=float, default=0.2)
    p.add_argument("--drop-unknown-biology", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.matrix:
        if not args.metadata:
            raise SystemExit("--metadata is required when using --matrix")
        abundance, meta = load_matrix_with_metadata(
            args.matrix,
            args.metadata,
            sample_col=args.sample_col,
            batch_col=args.batch_col,
            biology_col=args.biology_col,
            orientation=args.orientation,
        )
    else:
        root = args.data_root or "/iplant/home/shared/NCEMS/PPA/TestDatasets"
        abundance, meta = load_protein_tsv_directory(
            root,
            pattern=args.pattern,
            abundance_col=args.abundance_col,
            protein_id_col=args.protein_id_col,
            metadata_path=args.metadata,
        )

    ds = preprocess_abundance_matrix(
        abundance,
        meta,
        min_present_frac=args.min_present_frac,
        drop_unknown_biology=args.drop_unknown_biology,
        biology_col=args.biology_col,
    )

    print("\n=== Files / raw matrix ===")
    print(f"Samples:  {abundance.shape[0]}")
    print(f"Proteins before filtering: {abundance.shape[1]}")
    print(f"Observed entries before filtering: {abundance.notna().mean().mean():.1%}")

    print("\n=== After preprocessing ===")
    print(f"X shape: {ds.X.shape}")
    print(f"Missing rate: {1.0 - ds.M.mean():.1%}")

    print("\n=== Metadata columns ===")
    print(ds.meta.columns.tolist())

    print("\n=== Batch counts ===")
    print(ds.meta[args.batch_col].value_counts(dropna=False).to_string())

    if args.biology_col in ds.meta.columns:
        print("\n=== Biology counts ===")
        print(ds.meta[args.biology_col].value_counts(dropna=False).to_string())
        print("\n=== Biology x Batch ===")
        print(pd.crosstab(ds.meta[args.biology_col], ds.meta[args.batch_col]).to_string())

    print("\nFirst 5 metadata rows:")
    print(ds.meta.head().to_string(index=False))


if __name__ == "__main__":
    main()
