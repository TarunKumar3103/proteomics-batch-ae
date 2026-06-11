# proteomics-batch-ae

Real-data proteomics batch-correction experiments using an adversarial guided autoencoder, ComBat-style baselines, and transparent linear baselines.

This version intentionally **does not depend on the old synthetic Colab dataset**. There is no `make_hard_proteomics()` path in the experiment runner. The default workflow reads real `protein.tsv` files from a directory tree such as Ian's CyVerse dataset:

```text
/iplant/home/shared/NCEMS/PPA/TestDatasets/
└── PXD.../
    └── <sample or run>/
        └── search_results/
            └── protein.tsv
```

Each `protein.tsv` is treated as one sample. The loader auto-detects a protein identifier column and uses `Razor intensity` as the default abundance proxy.

---

## What the code does

1. Recursively finds `search_results/protein.tsv` files.
2. Builds a sample-by-protein abundance matrix.
3. Uses the PXD accession as the default batch label.
4. Uses built-in PXD family mappings for HEK293 vs HeLa as the default biology label.
5. Filters proteins by missingness.
6. Applies log2 transform, imputation, and standardization.
7. Runs baselines:
   - raw
   - batch mean centering
   - batch median centering
   - OLS batch removal
   - OLS batch removal preserving biology
   - ComBat, when the `combat` package works in the environment
8. Tunes and trains an adversarial autoencoder.
9. Evaluates real-data metrics:
   - batch classifier accuracy / balanced accuracy should decrease
   - biology classifier accuracy / balanced accuracy should stay high
   - batch PCA silhouette should decrease
   - biology PCA silhouette should ideally remain stable
   - observed-value RMSE change is reported as a distortion sanity check

Because real data has no clean ground truth, this repo does **not** report synthetic-only metrics like MSE-to-clean or correlation-to-clean.

---

## Install

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the `combat` package fails to install, you can still run the experiment without ComBat by installing the core packages:

```bash
pip install numpy pandas scikit-learn torch optuna matplotlib
```

The script will warn that ComBat failed and continue with OLS and centering baselines.

---

## Inspect Ian's CyVerse dataset

```bash
python scripts/inspect_dataset.py \
  --data-root /iplant/home/shared/NCEMS/PPA/TestDatasets \
  --min-present-frac 0.2
```

Check the output carefully. In particular, confirm:

- number of samples
- number of proteins after filtering
- missing rate
- batch counts
- HEK293 / HeLa counts
- biology × batch table

---

## Smoke test

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

---

## Full-ish CPU run

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

For a larger search, increase `--ae-trials` and/or `--trial-epochs`. If the node has many cores, increase `--optuna-jobs`; if memory use gets high, decrease it.

---

## Generic matrix input

For non-Ian datasets, provide a matrix and metadata file.

Samples as rows:

```bash
python scripts/run_real_experiment.py \
  --matrix abundance_matrix.tsv \
  --metadata sample_metadata.tsv \
  --orientation samples_rows \
  --sample-col sample_id \
  --batch-col batch \
  --biology-col condition \
  --outdir results/my_dataset
```

Proteins as rows:

```bash
python scripts/run_real_experiment.py \
  --matrix abundance_matrix.tsv \
  --metadata sample_metadata.tsv \
  --orientation proteins_rows \
  --sample-col sample_id \
  --batch-col batch \
  --biology-col condition \
  --outdir results/my_dataset
```

---

## Outputs

The output directory includes:

```text
metadata_used.csv
protein_ids.csv
comparison_metrics.csv
optuna_best.json
optuna_trials.csv
ae_training_history.csv
ae_latent.npy
ae_model.pt
```

If `--save-corrected` is used, it also saves corrected abundance matrices.

---

## Important interpretation warning

Batch correction is not automatically good just because batch predictability decreases. A useful correction should reduce batch signal while preserving biological signal. If the biology classifier collapses, the model is probably removing real biology along with batch effects.
