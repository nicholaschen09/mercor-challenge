# Mercor Cheating Detection (Kaggle) — Starter

This repo contains a tiny baseline that produces a valid Kaggle submission from the provided CSVs in `data/`.

## Quickstart (no dependencies)

From the repo root:

```bash
python3 scripts/baseline_stdlib.py --train sample-data/train.csv --test sample-data/test.csv --out submission.csv
```

This writes `submission.csv` with columns:

`user_hash,prediction`

Upload that file to Kaggle → **Submit Predictions**.

## Data

- `sample-data/train.csv`: includes `high_conf_clean` and `is_cheating` (labels are missing for unlabeled rows)
- `sample-data/test.csv`: predict for these rows
- `sample-data/social_graph.csv`: optional graph edges (not used in the baseline yet)
- `sample-data/feature_metadata.json`: feature types/ranges/missingness

## Next steps to improve

- Replace the baseline with LightGBM/CatBoost/XGBoost.
- Use `high_conf_clean=1` rows as weak negatives (carefully!) or for semi-supervised learning.
- Build graph features from `social_graph.csv` (degree, neighbor label rates, embeddings).


