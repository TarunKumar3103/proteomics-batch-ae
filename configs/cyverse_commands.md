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
