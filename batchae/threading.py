"""CPU/thread controls for JupyterLab/CyVerse runs.

Set these before heavy NumPy/Torch work. The main CLI calls this before importing
most scientific modules. In notebooks, call configure_threads(...) near the top.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ThreadConfig:
    torch_threads: int = 1
    interop_threads: int = 1
    dataloader_workers: int = 0
    optuna_jobs: int = 1


def set_thread_env(num_threads: int) -> None:
    """Cap common BLAS/OpenMP thread pools.

    This is the main fix for the "one run eats all 64 cores" issue. PyTorch and
    NumPy linear algebra can use MKL/OpenMP threads even when DataLoader workers
    are zero.
    """

    n = str(max(1, int(num_threads)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = n


def configure_torch_threads(torch_threads: int = 1, interop_threads: int = 1) -> None:
    """Apply PyTorch-specific thread caps if torch is available."""

    try:
        import torch

        torch.set_num_threads(max(1, int(torch_threads)))
        # set_num_interop_threads can only be called before interop work starts.
        try:
            torch.set_num_interop_threads(max(1, int(interop_threads)))
        except RuntimeError:
            pass
    except Exception:
        pass


def configure_threads(
    torch_threads: int = 1,
    interop_threads: int = 1,
    dataloader_workers: int = 0,
    optuna_jobs: int = 1,
) -> ThreadConfig:
    """Configure CPU usage and return the chosen settings."""

    cfg = ThreadConfig(
        torch_threads=max(1, int(torch_threads)),
        interop_threads=max(1, int(interop_threads)),
        dataloader_workers=max(0, int(dataloader_workers)),
        optuna_jobs=max(1, int(optuna_jobs)),
    )
    set_thread_env(cfg.torch_threads)
    configure_torch_threads(cfg.torch_threads, cfg.interop_threads)
    return cfg
