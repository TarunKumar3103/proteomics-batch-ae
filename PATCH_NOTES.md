# Patch notes: real-data performance and metric rigor

Applied targeted patches based on the CyVerse real-data runs and follow-up review.

## Data loading

- `PROTEIN_TSV_MANIFEST` is now honored in `batchae/data.py`.
  - Expected format: one absolute `protein.tsv` path per line.
  - If readable paths are found, discovery short-circuits and avoids a slow recursive Data Store walk.
- Default protein pattern is now `**/search_results/*/protein.tsv`, matching the CyVerse layout:
  `PXD.../pipeline_run/<run_id>/search_results/<sample_id>/protein.tsv`.
- Discovery uses Unix `find` first with visible progress messages, then falls back to `Path.glob`.
- Loading prints progress every 25 files by default.

## Optuna / reproducibility

- Optuna now uses `optuna.samplers.TPESampler(seed=args.seed)` instead of an unseeded default sampler.

## Speed

- The SVD used for AE initialization is computed once after loading the dataset and reused across Optuna trials and final training.
- `metrics.classifier_scores` now computes accuracy and balanced accuracy from one cross-validation prediction pass instead of fitting the same classifier twice.
- Optuna's inner loop uses `classifier_scores_fast` with lower logistic-regression `max_iter`; final reporting still uses the full classifier score function.

## Metric interpretation

- Added `constrained_batch_floor(batch_labels, biology_labels)`.
- The runner now prints the constrained batch-bAcc floor when biology labels are available. In the current HEK293/HeLa confounded design, this should be around 0.25, which is more scientifically relevant than naive 1/8 chance if biology is preserved.

## Files touched

- `batchae/data.py`
- `batchae/metrics.py`
- `batchae/model.py`
- `scripts/run_real_experiment.py`
- `scripts/inspect_dataset.py`
- `configs/cyverse_commands.md`

## Held-out baseline evaluation patch

Added `batchae/heldout_eval.py` and `scripts/evaluate_heldout_baselines.py`.

This script fixes the centering/OLS evaluation artifact by using one outer held-out protocol:
for each fold, fit the correction on train rows only, transform both train and test rows with that train-fitted corrector, train the probe on corrected train rows, and predict corrected held-out rows. This prevents held-out samples from influencing batch means or OLS coefficients.

The script is meant to validate/debug simple baselines (`raw`, `batch_mean_center`, `batch_median_center`, `ols_remove_batch`, `ols_preserve_biology`). It intentionally does not present ComBat as held-out because typical ComBat usage is transductive. It also does not retrain the AE per fold; use it first to validate that the `0.000` centering/OLS rows disappear under a proper train/test boundary.

## Pareto / seed-summary patch

Added a separate multi-objective experiment path and a run-summary tool.

New files:

- `batchae/pareto.py`
  - Pareto-front helpers.
  - Optuna trial DataFrame flattening.
  - Practical trial selection helper: choose the lowest batch-bAcc trial subject to biology/RMSE constraints.
  - 2D batch-vs-biology Pareto plotting with baseline overlays.
  - Normalizers for `comparison_metrics.csv` and `heldout_baseline_metrics.csv`.

- `scripts/run_pareto_experiment.py`
  - Runs multi-objective Optuna with two tradeoff objectives:
    1. minimize batch balanced accuracy,
    2. minimize biology loss (`1 - bio_bacc`).
  - Observed-value RMSE change is saved for every trial and can be used as a selection constraint / plot color, but is not minimized as a Pareto objective.
  - Saves all AE trials to `ae_multiobjective_trials.csv`.
  - Saves non-dominated trials to `ae_pareto_trials.csv`.
  - Overlays Raw/ComBat transductive points and optional leakage-safe held-out centering/OLS points.
  - Saves `ae_pareto_front.png`.
  - Selects a practical final AE trial using explicit constraints, e.g. `bio_bacc >= 0.98`, then trains/evaluates that selected final AE unless `--skip-final-training` is used.

- `scripts/summarize_experiment_runs.py`
  - Aggregates multiple result folders across seeds.
  - Normalizes both standard `comparison_metrics.csv` outputs and held-out baseline `heldout_baseline_metrics.csv` outputs.
  - Writes `per_run_metrics.csv` and `aggregate_metrics.csv` with mean/std across runs.
  - Computes residual batch signal above the constrained biology-preserving floor when a floor is supplied.

Rationale:

The original scalar Optuna objective forces batch removal, biology preservation, and distortion into one blended score. The Pareto script exposes the full tradeoff instead of making the result depend on one weight choice. The summary script prevents cherry-picking by reporting mean ± spread across seeds.

## Pareto protocol cleanup + held-out AE patch

Added protocol-clean held-out AE evaluation and corrected Pareto front semantics.

Changes:

- `scripts/run_pareto_experiment.py`
  - The multi-objective search now optimizes two true tradeoff objectives:
    1. minimize batch balanced accuracy,
    2. minimize biology loss (`1 - bio_bacc`).
  - `rmse_change` is still saved for each trial and used for plot coloring / optional selection constraints, but it is no longer a Pareto objective. This prevents low-distortion no-correction trials from contaminating the batch/biology frontier.
  - Added `--evaluate-heldout-ae`, which retrains the selected AE on train folds only, corrects held-out test rows, and evaluates pooled held-out probe predictions. This is expensive but puts AE on the same held-out protocol as OLS/centering baselines.
  - Added `--heldout-ae-epochs` and `--heldout-ae-verbose`.
  - `selected_trial.json` now explicitly warns that in-trial metrics are an operating-point selection signal, not the final reported performance. Report `comparison_metrics.csv` and, when used, `heldout_ae_metrics.csv`.

- `batchae/heldout_eval.py`
  - Added `evaluate_ae_heldout(...)`, a reusable outer-CV protocol for selected AE configurations.
  - For each outer fold, the AE is initialized and trained only on train rows, then used to correct train and held-out test rows. Probe classifiers are fit on corrected train rows and evaluated on corrected test rows.

- `batchae/pareto.py`
  - Default Pareto helper now uses the 2D batch/biology objective pair. RMSE remains diagnostic.

- `scripts/summarize_experiment_runs.py`
  - Can now discover and aggregate `heldout_ae_metrics.csv` via `--discover-heldout-ae`.
