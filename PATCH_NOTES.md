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
