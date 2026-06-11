"""Data loading and preprocessing for real proteomics batch-correction experiments.

The main loader supports Ian's CyVerse layout:

    <root>/<PXD...>/<sample or run>/search_results/protein.tsv

Each protein.tsv is treated as one sample. The loader extracts one abundance
column, e.g. "Razor intensity", pivots proteins into columns, and returns a
sample x protein matrix plus metadata.

The module also supports generic matrix + metadata input for other datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_HEK293_PXDS = {
    "PXD000115", "PXD000197", "PXD000520", "PXD001085", "PXD001281",
    "PXD001468", "PXD001572", "PXD001828", "PXD001942", "PXD002070",
}

DEFAULT_HELA_PXDS = {
    "PXD000381", "PXD000883", "PXD001061", "PXD001101", "PXD001374",
    "PXD001660", "PXD001798", "PXD001805", "PXD002001", "PXD002039",
}

PROTEIN_ID_CANDIDATES = [
    "Protein IDs",
    "Protein.IDs",
    "Majority protein IDs",
    "Majority.protein.IDs",
    "Leading razor protein",
    "Leading.razor.protein",
    "Protein ID",
    "protein_id",
    "protein",
    "Protein",
    "Accession",
    "Entry",
    "Protein.Group",
    "Protein.Group.Accessions",
]

ABUNDANCE_CANDIDATES = [
    "Razor intensity",
    "Razor.Intensity",
    "razor_intensity",
    "RazorIntensity",
    "Razor Intensity",
    "Intensity",
    "intensity",
    "LFQ intensity",
    "LFQ.intensity",
    "Abundance",
    "abundance",
]


@dataclass
class ProteomicsDataset:
    """Container returned by all loaders.

    Attributes
    ----------
    X:
        Preprocessed abundance matrix with shape n_samples x n_proteins.
    M:
        Missingness/observation mask with 1 for originally observed values and
        0 for missing values, same shape as X.
    meta:
        Sample metadata. Must contain sample_id and usually batch/biology cols.
    protein_ids:
        Protein identifiers matching X columns.
    raw_X:
        Log-transformed but unimputed/unstandardized matrix with NaNs retained.
    """

    X: np.ndarray
    M: np.ndarray
    meta: pd.DataFrame
    protein_ids: list[str]
    raw_X: pd.DataFrame


def _read_table(path: str | Path, sep: str | None = None) -> pd.DataFrame:
    """Robust table reader for CyVerse/Data Store mounted files."""
    import time

    path = Path(path)
    if sep is None:
        sep = "	" if path.suffix.lower() in {".tsv", ".txt"} else ","

    last_error = None
    for attempt in range(1, 4):
        try:
            return pd.read_csv(path, sep=sep, low_memory=False)
        except Exception as e:
            last_error = e
            try:
                return pd.read_csv(
                    path,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception as e2:
                last_error = e2
                time.sleep(0.5 * attempt)

    raise RuntimeError(f"Failed to read table after retries: {path}. Last error: {last_error}")


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def detect_column(columns: Iterable[str], candidates: Iterable[str], requested: str | None = None) -> str:
    """Find a column by exact or normalized name."""
    columns = list(columns)
    if requested:
        if requested in columns:
            return requested
        req_norm = _normalize_col(requested)
        for col in columns:
            if _normalize_col(col) == req_norm:
                return col
        raise ValueError(f"Requested column {requested!r} was not found. Available columns: {columns[:30]}")

    normalized = {_normalize_col(col): col for col in columns}
    for cand in candidates:
        cand_norm = _normalize_col(cand)
        if cand_norm in normalized:
            return normalized[cand_norm]

    raise ValueError(
        "Could not auto-detect a required column. "
        f"Tried candidates={list(candidates)}. Available columns start with: {columns[:30]}"
    )


def extract_pxd_id(path: str | Path) -> str:
    match = re.search(r"PXD\d{6}", str(path), flags=re.IGNORECASE)
    return match.group(0).upper() if match else "UNKNOWN_PXD"


def family_from_pxd(pxd: str) -> str:
    pxd = str(pxd).upper()
    if pxd in DEFAULT_HEK293_PXDS:
        return "HEK293"
    if pxd in DEFAULT_HELA_PXDS:
        return "HeLa"
    return "unknown"


def discover_protein_tsvs(root: str | Path, pattern: str = "**/search_results/*/protein.tsv") -> list[Path]:
    """Find protein.tsv files with visible progress on CyVerse mounts."""
    import subprocess

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    print(f"[discover] Searching under: {root}", flush=True)

    cmd = [
        "find",
        str(root),
        "-type",
        "f",
        "-name",
        "protein.tsv",
    ]

    files: list[Path] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        # Keep the intended layout first:
        # .../search_results/<sample_id>/protein.tsv
        if "/search_results/" in line:
            files.append(Path(line))
            if len(files) == 1 or len(files) % 25 == 0:
                print(f"[discover] Found {len(files)} protein.tsv files so far...", flush=True)

    rc = proc.wait()
    if rc != 0:
        print(f"[discover] Warning: find exited with code {rc}", flush=True)

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No protein.tsv files found under {root}")

    print(f"[discover] Finished. Found {len(files)} protein.tsv files.", flush=True)
    return files


def _unique_sample_ids(ids: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for sample_id in ids:
        sample_id = str(sample_id)
        if sample_id not in counts:
            counts[sample_id] = 0
            out.append(sample_id)
        else:
            counts[sample_id] += 1
            out.append(f"{sample_id}__dup{counts[sample_id]}")
    return out


def load_protein_tsv_directory(
    root: str | Path,
    abundance_col: str | None = None,
    protein_id_col: str | None = None,
    pattern: str = "**/search_results/*/protein.tsv",
    zero_as_missing: bool = True,
    duplicate_policy: str = "max",
    metadata_path: str | Path | None = None,
    metadata_sample_col: str = "sample_id",
    metadata_batch_col: str | None = None,
    metadata_biology_col: str | None = None,
    verbose: bool = True,
    progress_every: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load recursive protein.tsv files into a samples x proteins DataFrame.

    Returns
    -------
    abundance_df:
        Samples x proteins, with NaNs for missing proteins.
    meta:
        Metadata with sample_id, path, dataset_id, batch, and biology columns.
    """
    files = discover_protein_tsvs(root, pattern=pattern)

    if verbose:
        print(f"[load] Found {len(files)} protein.tsv files under {root}", flush=True)
        print(f"[load] Pattern used: {pattern}", flush=True)

    sample_ids: list[str] = []
    rows: list[pd.Series] = []
    meta_rows: list[dict[str, str]] = []

    for i, file_path in enumerate(files, start=1):
        if verbose and (i == 1 or i % progress_every == 0 or i == len(files)):
            print(f"[load] Reading {i}/{len(files)}: {file_path}", flush=True)

        try:
            df = _read_table(file_path, sep="\t")
        except Exception as e:
            raise RuntimeError(f"Failed reading {file_path}: {e}") from e
        try:
            pid_col = detect_column(df.columns, PROTEIN_ID_CANDIDATES, protein_id_col)
            val_col = detect_column(df.columns, ABUNDANCE_CANDIDATES, abundance_col)
        except Exception as e:
            raise RuntimeError(
                f"Column detection failed for {file_path}. "
                f"Available columns: {list(df.columns)[:40]}. Original error: {e}"
            ) from e

        small = df[[pid_col, val_col]].copy()
        small[pid_col] = small[pid_col].astype(str).str.strip()
        small = small[small[pid_col].notna() & (small[pid_col] != "") & (small[pid_col].str.lower() != "nan")]
        small[val_col] = pd.to_numeric(small[val_col], errors="coerce")

        if zero_as_missing:
            small.loc[small[val_col] <= 0, val_col] = np.nan

        if duplicate_policy == "max":
            series = small.groupby(pid_col, sort=False)[val_col].max()
        elif duplicate_policy == "sum":
            series = small.groupby(pid_col, sort=False)[val_col].sum(min_count=1)
        elif duplicate_policy == "mean":
            series = small.groupby(pid_col, sort=False)[val_col].mean()
        else:
            raise ValueError("duplicate_policy must be one of: max, sum, mean")

        # Usually sample folder is the directory above search_results.
        sample_folder = file_path.parent.parent.name if file_path.parent.name == "search_results" else file_path.parent.name
        pxd = extract_pxd_id(file_path)
        sample_id = f"{pxd}__{sample_folder}"

        sample_ids.append(sample_id)
        rows.append(series)
        meta_rows.append({
            "sample_id": sample_id,
            "path": str(file_path),
            "dataset_id": pxd,
            "batch": pxd,
            "biology": family_from_pxd(pxd),
        })

    sample_ids = _unique_sample_ids(sample_ids)
    abundance_df = pd.DataFrame(rows, index=sample_ids)
    abundance_df.index.name = "sample_id"

    meta = pd.DataFrame(meta_rows)
    meta["sample_id"] = sample_ids

    if metadata_path:
        user_meta = _read_table(metadata_path)
        if metadata_sample_col not in user_meta.columns:
            raise ValueError(f"metadata_sample_col={metadata_sample_col!r} not in metadata file")
        meta = meta.merge(user_meta, left_on="sample_id", right_on=metadata_sample_col, how="left", suffixes=("", "_user"))
        if metadata_batch_col:
            meta["batch"] = meta[metadata_batch_col].astype(str)
        if metadata_biology_col:
            meta["biology"] = meta[metadata_biology_col].astype(str)

    return abundance_df, meta


def load_matrix_with_metadata(
    matrix_path: str | Path,
    metadata_path: str | Path,
    sample_col: str = "sample_id",
    batch_col: str = "batch",
    biology_col: str = "biology",
    sep: str | None = None,
    orientation: str = "samples_rows",
    zero_as_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a generic abundance matrix and sample metadata.

    matrix orientation:
    - samples_rows: first column/index is sample id; columns are proteins.
    - proteins_rows: first column/index is protein id; columns are samples.
    """
    matrix = _read_table(matrix_path, sep=sep)
    first_col = matrix.columns[0]
    matrix = matrix.set_index(first_col)

    if orientation not in {"samples_rows", "proteins_rows"}:
        raise ValueError("orientation must be 'samples_rows' or 'proteins_rows'")
    if orientation == "proteins_rows":
        matrix = matrix.T

    matrix.index = matrix.index.astype(str)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    if zero_as_missing:
        matrix = matrix.mask(matrix <= 0)

    meta = _read_table(metadata_path)
    required = [sample_col, batch_col]
    for col in required:
        if col not in meta.columns:
            raise ValueError(f"Required metadata column {col!r} missing from {metadata_path}")
    if biology_col not in meta.columns:
        meta[biology_col] = "unknown"

    meta = meta.copy()
    meta["sample_id"] = meta[sample_col].astype(str)
    meta["batch"] = meta[batch_col].astype(str)
    meta["biology"] = meta[biology_col].astype(str)
    meta = meta.set_index("sample_id").loc[matrix.index].reset_index()

    return matrix, meta


def preprocess_abundance_matrix(
    abundance_df: pd.DataFrame,
    meta: pd.DataFrame,
    min_present_frac: float = 0.2,
    max_missing_frac: float | None = None,
    log2_transform: bool = True,
    impute_strategy: str = "median",
    standardize: bool = True,
    drop_unknown_biology: bool = False,
    biology_col: str = "biology",
) -> ProteomicsDataset:
    """Filter, transform, impute, and optionally standardize abundance matrix."""
    abundance_df = abundance_df.copy()
    meta = meta.copy()

    if drop_unknown_biology and biology_col in meta.columns:
        keep_samples = ~meta[biology_col].astype(str).str.lower().isin({"unknown", "nan", "", "none"})
        meta = meta.loc[keep_samples].reset_index(drop=True)
        abundance_df = abundance_df.loc[meta["sample_id"].values]

    present = abundance_df.notna().mean(axis=0)
    keep = present >= float(min_present_frac)
    if max_missing_frac is not None:
        keep &= (1.0 - present) <= float(max_missing_frac)
    abundance_df = abundance_df.loc[:, keep]

    if abundance_df.shape[1] == 0:
        raise ValueError("No proteins remain after filtering. Lower --min-present-frac.")

    raw = abundance_df.astype(float)
    if log2_transform:
        raw = np.log2(raw + 1.0)

    M = raw.notna().astype(np.float32).values

    if impute_strategy == "median":
        fill_values = raw.median(axis=0)
    elif impute_strategy == "mean":
        fill_values = raw.mean(axis=0)
    elif impute_strategy == "zero":
        fill_values = pd.Series(0.0, index=raw.columns)
    else:
        raise ValueError("impute_strategy must be one of: median, mean, zero")

    X_df = raw.fillna(fill_values)
    # Proteins that are all NaN after filtering are unlikely but possible.
    X_df = X_df.fillna(0.0)

    if standardize:
        means = X_df.mean(axis=0)
        stds = X_df.std(axis=0, ddof=0).replace(0, 1.0)
        X_df = (X_df - means) / stds

    return ProteomicsDataset(
        X=X_df.values.astype(np.float32),
        M=M.astype(np.float32),
        meta=meta.reset_index(drop=True),
        protein_ids=[str(c) for c in X_df.columns],
        raw_X=raw,
    )


def assert_experiment_ready(meta: pd.DataFrame, batch_col: str, biology_col: str | None = None) -> None:
    if batch_col not in meta.columns:
        raise ValueError(f"Batch column {batch_col!r} not found in metadata")
    if meta[batch_col].nunique() < 2:
        raise ValueError("Need at least two batch classes for batch-correction evaluation")
    if biology_col and biology_col in meta.columns and meta[biology_col].nunique() < 2:
        print(f"WARNING: biology column {biology_col!r} has <2 classes; biology preservation metrics will be skipped.")
