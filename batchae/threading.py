"""Utilities for making CPU usage predictable on shared systems."""

from __future__ import annotations

import os


def configure_threads(torch_threads: int = 1, interop_threads: int = 1) -> None:
    """Limit BLAS/OpenMP/PyTorch thread usage.

    Call this near the top of a script, before large NumPy/SciPy/PyTorch work.
    This is especially important when Optuna runs several trials in parallel.
    """
    torch_threads = max(int(torch_threads), 1)
    interop_threads = max(int(interop_threads), 1)

    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[name] = str(torch_threads)

    try:
        import torch

        torch.set_num_threads(torch_threads)
        # set_num_interop_threads can only be called once in a process before
        # interop work begins; ignore RuntimeError if the runtime was already set.
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass
    except Exception:
        pass
