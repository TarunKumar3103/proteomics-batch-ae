# CyVerse CPU commands

## Smoke test

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

## Full sequential CPU run

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

## Full parallel CPU run

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
