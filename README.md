# Mercor Cheating Detection (Kaggle) — Starter

This repo contains a tiny baseline that produces a valid Kaggle submission from the provided CSVs in `data/`.

## Quickstart (no dependencies)

From the repo root:

```bash
python3 scripts/baseline_stdlib.py --train sample-data/train.csv --test sample-data/test.csv --out submission.csv
```

Validate the submission matches `test.csv`:

```bash
python3 scripts/baseline_stdlib.py --out submission.csv --validate
```

This writes `submission.csv` with columns:

`user_hash,prediction`

Upload that file to Kaggle → **Submit Predictions**.

## Better baseline (still no dependencies)

This version learns logistic regression weights, uses `high_conf_clean=1` rows as weak negatives, and adds simple graph features from `social_graph.csv`.

```bash
python3 scripts/baseline_v2_stdlib.py --out submission.csv --validate --eval
```

## Baseline v4 (better training + better graph features, still no dependencies)

```bash
python3 scripts/baseline_v4_stdlib.py --out submission_v4.csv --validate --eval
```

If you want it to try a small hyperparameter search:

```bash
python3 scripts/baseline_v4_stdlib.py --out submission_v4.csv --validate --eval --tune
```

## Data

- `sample-data/train.csv`: includes `high_conf_clean` and `is_cheating` (labels are missing for unlabeled rows)
- `sample-data/test.csv`: predict for these rows
- `sample-data/social_graph.csv`: optional graph edges (not used in the baseline yet)
- `sample-data/feature_metadata.json`: feature types/ranges/missingness

## Next steps to improve

- Replace the baseline with LightGBM/CatBoost/XGBoost.
- Use `high_conf_clean=1` rows as weak negatives (carefully!) or for semi-supervised learning.
- Build graph features from `social_graph.csv` (degree, neighbor label rates, embeddings).


