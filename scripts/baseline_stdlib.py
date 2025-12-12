#!/usr/bin/env python3
"""
Mercor Cheating Detection - starter baseline (no third-party deps).

What it does:
- Trains a simple linear score from labeled rows in train.csv using per-feature
  class means/variances (LDA-ish) + missingness signal.
- Writes Kaggle-ready submission.csv for test.csv.

This is intentionally dependency-free so it runs on a fresh Python install.
You can later replace this with LightGBM/CatBoost/graph features.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


EPS = 1e-9


def sigmoid(x: float) -> float:
    # Numerically-stable-ish sigmoid with clipping.
    if x >= 0:
        z = math.exp(-min(x, 50.0))
        return 1.0 / (1.0 + z)
    z = math.exp(min(x, 50.0))
    return z / (1.0 + z)


def safe_float(s: str) -> Optional[float]:
    s = s.strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class ClassStats:
    # Per-feature numeric aggregates over non-missing values.
    n: List[int]
    sum: List[float]
    sumsq: List[float]
    # Missingness counts per feature.
    n_missing: List[int]
    # Total rows for this class (labeled rows only).
    n_rows: int = 0


def read_header(path: str) -> List[str]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def feature_columns(header: List[str]) -> List[str]:
    return [c for c in header if c.startswith("feature_")]


def index_map(header: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(header)}


def init_stats(n_features: int) -> ClassStats:
    return ClassStats(
        n=[0] * n_features,
        sum=[0.0] * n_features,
        sumsq=[0.0] * n_features,
        n_missing=[0] * n_features,
        n_rows=0,
    )


def fit_from_train(train_path: str) -> Tuple[List[str], Dict[str, float], Dict[str, float], float, float]:
    """
    Returns:
      - features: ordered list of feature column names
      - w_value: per-feature weight for value contribution
      - w_missing: per-feature weight added if feature is missing
      - bias: bias term (logit prior)
      - prior: P(y=1) on labeled rows
    """
    header = read_header(train_path)
    idx = index_map(header)
    if "is_cheating" not in idx:
        raise RuntimeError("train.csv missing required column: is_cheating")

    feats = feature_columns(header)
    feat_idxs = [idx[c] for c in feats]

    stats0 = init_stats(len(feats))
    stats1 = init_stats(len(feats))

    with open(train_path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader)  # header
        for row in reader:
            y_raw = row[idx["is_cheating"]].strip() if idx["is_cheating"] < len(row) else ""
            if y_raw == "" or y_raw.lower() == "nan":
                # unlabeled row
                continue
            try:
                y = int(float(y_raw))
            except ValueError:
                continue
            if y not in (0, 1):
                continue

            st = stats1 if y == 1 else stats0
            st.n_rows += 1
            for j, col_i in enumerate(feat_idxs):
                v = safe_float(row[col_i]) if col_i < len(row) else None
                if v is None:
                    st.n_missing[j] += 1
                    continue
                st.n[j] += 1
                st.sum[j] += v
                st.sumsq[j] += v * v

    if stats0.n_rows == 0 or stats1.n_rows == 0:
        raise RuntimeError(
            f"Not enough labeled data to train baseline. labeled_0={stats0.n_rows}, labeled_1={stats1.n_rows}"
        )

    prior = stats1.n_rows / (stats0.n_rows + stats1.n_rows)
    prior = min(max(prior, 1e-6), 1.0 - 1e-6)
    bias = math.log(prior / (1.0 - prior))

    w_value: Dict[str, float] = {}
    w_missing: Dict[str, float] = {}

    # LDA-ish weights: (mu1 - mu0) / pooled_var
    for j, feat in enumerate(feats):
        # Means per class (fallback to 0 if completely missing in that class for that feature)
        mu0 = stats0.sum[j] / max(stats0.n[j], 1)
        mu1 = stats1.sum[j] / max(stats1.n[j], 1)
        var0 = max(stats0.sumsq[j] / max(stats0.n[j], 1) - mu0 * mu0, 0.0)
        var1 = max(stats1.sumsq[j] / max(stats1.n[j], 1) - mu1 * mu1, 0.0)
        pooled = 0.5 * (var0 + var1) + EPS
        w = (mu1 - mu0) / pooled
        # Clamp extreme weights a bit to avoid numerical blowups on weird ranges.
        w = max(min(w, 10.0), -10.0)
        w_value[feat] = w

        # Missingness as signal: log-odds of missing given class.
        p_miss0 = (stats0.n_missing[j] + 1.0) / (stats0.n_rows + 2.0)
        p_miss1 = (stats1.n_missing[j] + 1.0) / (stats1.n_rows + 2.0)
        w_m = math.log(p_miss1 / p_miss0)
        w_m = max(min(w_m, 2.0), -2.0)
        w_missing[feat] = w_m

    return feats, w_value, w_missing, bias, prior


def predict_row(
    row: List[str],
    feat_idxs: List[int],
    feats: List[str],
    w_value: Dict[str, float],
    w_missing: Dict[str, float],
    bias: float,
) -> float:
    score = bias
    for j, col_i in enumerate(feat_idxs):
        feat = feats[j]
        v = safe_float(row[col_i]) if col_i < len(row) else None
        if v is None:
            score += w_missing.get(feat, 0.0)
        else:
            score += w_value.get(feat, 0.0) * v
    # Prevent extreme exps.
    score = max(min(score, 20.0), -20.0)
    return sigmoid(score)


def write_submission(
    test_path: str,
    out_path: str,
    feats: List[str],
    w_value: Dict[str, float],
    w_missing: Dict[str, float],
    bias: float,
) -> None:
    header = read_header(test_path)
    idx = index_map(header)
    if "user_hash" not in idx:
        raise RuntimeError("test.csv missing required column: user_hash")
    feat_idxs = [idx[c] for c in feats]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(test_path, "r", newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.reader(fin)
        _ = next(reader)  # header
        writer = csv.writer(fout)
        writer.writerow(["user_hash", "prediction"])
        for row in reader:
            user_hash = row[idx["user_hash"]]
            p = predict_row(row, feat_idxs, feats, w_value, w_missing, bias)
            writer.writerow([user_hash, f"{p:.10f}"])


def iter_csv_rows(path: str) -> Iterable[List[str]]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        yield next(reader)  # header
        for row in reader:
            yield row


def validate_submission(submission_path: str, test_path: str) -> None:
    """
    Raises ValueError with a helpful message if the submission doesn't match test.csv.
    Checks:
    - Header is exactly: user_hash,prediction
    - Same number of rows as test
    - Same set of user_hash values as test
    - No duplicate user_hash
    - prediction is numeric and in [0, 1]
    """
    # Load expected hashes from test.csv
    test_iter = iter_csv_rows(test_path)
    test_header = next(test_iter)
    try:
        test_user_idx = test_header.index("user_hash")
    except ValueError as e:
        raise ValueError("test.csv must contain a user_hash column") from e

    expected: Set[str] = set()
    for row in test_iter:
        if test_user_idx < len(row):
            expected.add(row[test_user_idx])
    if not expected:
        raise ValueError("test.csv appears empty or user_hash could not be read")

    # Validate submission rows
    sub_iter = iter_csv_rows(submission_path)
    sub_header = next(sub_iter)
    if sub_header != ["user_hash", "prediction"]:
        raise ValueError(f"Bad submission header {sub_header!r}. Expected ['user_hash','prediction']")

    seen: Set[str] = set()
    for row in sub_iter:
        if len(row) < 2:
            raise ValueError("Submission has a row with <2 columns")
        u = row[0]
        if u in seen:
            raise ValueError(f"Duplicate user_hash in submission: {u}")
        seen.add(u)
        p = safe_float(row[1])
        if p is None or p < 0.0 or p > 1.0:
            raise ValueError(f"Invalid prediction for {u}: {row[1]!r} (must be numeric in [0,1])")

    if len(seen) != len(expected):
        raise ValueError(f"Row count mismatch: submission has {len(seen)} rows, test has {len(expected)} rows")

    missing = expected - seen
    extra = seen - expected
    if missing:
        sample = sorted(list(missing))[:5]
        raise ValueError(f"Submission missing {len(missing)} user_hash values from test.csv. Example(s): {sample}")
    if extra:
        sample = sorted(list(extra))[:5]
        raise ValueError(f"Submission has {len(extra)} unknown user_hash values not in test.csv. Example(s): {sample}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="sample-data/train.csv", help="Path to train.csv")
    ap.add_argument("--test", default="sample-data/test.csv", help="Path to test.csv")
    ap.add_argument("--out", default="submission.csv", help="Output submission path")
    ap.add_argument("--validate", action="store_true", help="Validate the output submission against test.csv")
    args = ap.parse_args()

    # Friendly auto-detection if the user’s folder name differs (e.g. data/ vs sample-data/).
    if not os.path.exists(args.train):
        for candidate in ("data/train.csv", "sample-data/train.csv"):
            if os.path.exists(candidate):
                args.train = candidate
                break
    if not os.path.exists(args.test):
        for candidate in ("data/test.csv", "sample-data/test.csv"):
            if os.path.exists(candidate):
                args.test = candidate
                break
    if not os.path.exists(args.train) or not os.path.exists(args.test):
        raise FileNotFoundError(
            "Could not find train/test CSVs. Tried:\n"
            f"- {args.train}\n"
            f"- {args.test}\n"
            "Common locations:\n"
            "- data/train.csv and data/test.csv\n"
            "- sample-data/train.csv and sample-data/test.csv\n"
        )

    feats, w_value, w_missing, bias, prior = fit_from_train(args.train)
    write_submission(args.test, args.out, feats, w_value, w_missing, bias)

    if args.validate:
        validate_submission(args.out, args.test)
        print("Validation: OK")

    print(f"Wrote {args.out}")
    print(f"Trained on labeled prior P(cheating)= {prior:.4f} (labeled rows only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


