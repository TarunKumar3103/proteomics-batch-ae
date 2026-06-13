# CyVerse commands

## 1. Clone

```bash
git clone https://github.com/TarunKumar3103/proteomics-batch-ae.git
cd proteomics-batch-ae
```

## 2. Install

Recommended in a fresh environment:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `combat` fails, continue without it temporarily:

```bash
pip install numpy pandas scikit-learn torch optuna matplotlib
```

The script will still run OLS and centering baselines.

## 3. Inspect Ian's dataset

```bash
python scripts/inspect_dataset.py \
  --data-root /iplant/home/shared/NCEMS/PPA/TestDatasets \
  --min-present-frac 0.2
```

## 4. Smoke test

```bash
python scripts/run_real_experiment.py \
  --data-root /iplant/home/shared/NCEMS/PPA/TestDatasets \
  --outdir results/smoke_real \
  --ae-trials 2 \
  --trial-epochs 5 \
  --final-epochs 10 \
  --torch-threads 1 \
  --interop-threads 1 \
  --optuna-jobs 1
```

## 5. Real CPU run

Start conservatively. If the node has many cores, increase `--optuna-jobs`.

```bash
python scripts/run_real_experiment.py \
  --data-root /iplant/home/shared/NCEMS/PPA/TestDatasets \
  --outdir results/ian_real_v1 \
  --ae-trials 50 \
  --trial-epochs 150 \
  --final-epochs 500 \
  --min-present-frac 0.2 \
  --torch-threads 1 \
  --interop-threads 1 \
  --optuna-jobs 8 \
  --sklearn-jobs 1 \
  --save-corrected
```

## Notes

- `--torch-threads 1` prevents each trial from grabbing many CPU cores.
- `--optuna-jobs 8` runs multiple trials at once.
- If memory usage is high, lower `--optuna-jobs`.
- If too few proteins remain, lower `--min-present-frac` to 0.1.
- If too many sparse proteins remain, raise `--min-present-frac` to 0.3 or 0.5.

## Multi-objective Pareto run

```bash
PROTEIN_TSV_MANIFEST=results/protein_tsv_manifest.txt \
python -u scripts/run_pareto_experiment.py \
  --data-root /data-store/iplant/home/shared/NCEMS/PPA/TestDatasets \
  --outdir results/pareto_75trial_seed123 \
  --ae-trials 75 \
  --trial-epochs 150 \
  --final-epochs 500 \
  --min-present-frac 0.2 \
  --select-bio-min 0.98 \
  --torch-threads 1 \
  --interop-threads 1 \
  --optuna-jobs 4 \
  --sklearn-jobs 1 \
  --seed 123 \
  --save-corrected
```

For a quick validation-only Pareto run without final selected-AE training, add `--skip-final-training` and reduce trials/epochs.

## Aggregate seed summary

```bash
python -u scripts/summarize_experiment_runs.py \
  --results-root results \
  --run-glob "ian_real_75trial_*_patched" \
  --discover-heldout-baselines \
  --constrained-floor 0.25 \
  --outdir results/summary_patched_75trial
```

The summary writes:

- `per_run_metrics.csv`
- `aggregate_metrics.csv`
- `summary_inputs.json`

## Protocol-clean Pareto run with held-out AE

This run is expensive because after the Pareto search it retrains the selected AE once per held-out fold.
Use it when you need apples-to-apples AE vs held-out OLS/centering evaluation.

```bash
PROTEIN_TSV_MANIFEST=results/protein_tsv_manifest.txt \
python -u scripts/run_pareto_experiment.py \
  --data-root /data-store/iplant/home/shared/NCEMS/PPA/TestDatasets \
  --outdir results/pareto_75trial_seed123_heldoutae \
  --ae-trials 75 \
  --trial-epochs 150 \
  --final-epochs 500 \
  --min-present-frac 0.2 \
  --select-bio-min 0.98 \
  --torch-threads 1 \
  --interop-threads 1 \
  --optuna-jobs 4 \
  --sklearn-jobs 1 \
  --seed 123 \
  --evaluate-heldout-ae \
  --save-corrected
```

For a faster validation of the held-out AE code path, reduce the search and held-out epochs:

```bash
PROTEIN_TSV_MANIFEST=results/protein_tsv_manifest.txt \
python -u scripts/run_pareto_experiment.py \
  --data-root /data-store/iplant/home/shared/NCEMS/PPA/TestDatasets \
  --outdir results/pareto_smoke_heldoutae \
  --ae-trials 3 \
  --trial-epochs 3 \
  --final-epochs 3 \
  --heldout-ae-epochs 3 \
  --min-present-frac 0.2 \
  --torch-threads 1 \
  --interop-threads 1 \
  --optuna-jobs 1 \
  --sklearn-jobs 1 \
  --seed 123 \
  --evaluate-heldout-ae
```

To summarize transductive AE runs, held-out OLS/centering, and held-out AE runs together:

```bash
python -u scripts/summarize_experiment_runs.py \
  --results-root results \
  --run-glob "ian_real_75trial_*_patched" \
  --discover-heldout-baselines \
  --discover-heldout-ae \
  --constrained-floor 0.25 \
  --outdir results/summary_with_heldout_ae
```
