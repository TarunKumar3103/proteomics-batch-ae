# Proteomics Batch-Correction Autoencoder

GitHub-ready version of the synthetic proteomics batch-correction experiment.

This repo fixes two problems from the original Colab cells:

1. **Fair baseline comparison**: ComBat is no longer just run once with defaults. The experiment tunes/evaluates several baseline variants, including ComBat with and without biological covariates, and selects the baseline on validation synthetic seeds before reporting held-out test seeds.
2. **CPU control for CyVerse/JupyterLab**: CPU threads are explicitly capped through environment variables and PyTorch thread settings. `DataLoader(num_workers=0)` is kept by default because the dataset is already in memory; the important issue is BLAS/PyTorch thread oversubscription, not data loading.

## Repo layout

```text
proteomics-batch-ae/
├── batchae/
│   ├── __init__.py
│   ├── baselines.py          # ComBat and OLS/mean-centering baselines
│   ├── data.py               # hard synthetic proteomics generator
│   ├── metrics.py            # batch/cell-line classifiers, MSE/correlation, scoring
│   ├── model.py              # adversarial CVAE with gradient reversal
│   ├── threading.py          # CPU/thread controls for CyVerse
│   └── training.py           # dataset building, training, Optuna objective helpers
├── scripts/
│   └── run_experiment.py     # main CLI entry point
├── configs/
│   └── cyverse_cpu_commands.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Install on CyVerse/JupyterLab

```bash
git clone <YOUR_GITHUB_REPO_URL>.git
cd proteomics-batch-ae
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If the `combat` package fails to install on your environment, the script still runs the non-ComBat baselines. For final comparisons, install ComBat because that is the main literature baseline you want.

## Quick smoke test

This checks that the code runs without burning CPU time:

```bash
python scripts/run_experiment.py \
  --outdir results/smoke \
  --n-proteins 200 \
  --ae-trials 2 \
  --trial-epochs 5 \
  --final-epochs 10 \
  --tune-seeds 123 \
  --test-seeds 456 \
  --torch-threads 2 \
  --interop-threads 1 \
  --optuna-jobs 1
```

## Fuller CPU run

On a 64-core CyVerse CPU node, do **not** let a single trial use all 64 threads. Two reasonable modes are:

### Safe sequential mode

```bash
python scripts/run_experiment.py \
  --outdir results/full_seq \
  --ae-trials 75 \
  --trial-epochs 300 \
  --final-epochs 800 \
  --tune-seeds 101 102 103 \
  --test-seeds 201 202 203 204 205 \
  --torch-threads 8 \
  --interop-threads 1 \
  --optuna-jobs 1
```

### Parallel Optuna mode

This is usually better for many CPU cores because each trial gets fewer threads:

```bash
python scripts/run_experiment.py \
  --outdir results/full_parallel \
  --ae-trials 75 \
  --trial-epochs 300 \
  --final-epochs 800 \
  --tune-seeds 101 102 103 \
  --test-seeds 201 202 203 204 205 \
  --torch-threads 1 \
  --interop-threads 1 \
  --optuna-jobs 8
```

## Outputs

The script writes:

```text
results/<run>/
├── best_ae_hparams.json
├── best_baseline_config.json
├── optuna_trials.csv
├── validation_baselines.csv
├── test_results_by_seed.csv
└── test_summary.csv
```

Use `test_summary.csv` for your final table and `test_results_by_seed.csv` for seed-level variance/error bars.

## Experimental-design note

The autoencoder and baselines are both selected on validation synthetic seeds and evaluated on different held-out synthetic seeds. This avoids the strongest criticism of the original notebook: tuning the autoencoder directly on the same dataset that is later used for the final comparison.
