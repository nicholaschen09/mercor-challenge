#!/usr/bin/env python3
"""
Mercor Cheating Detection - stronger baseline (no third-party deps).

Upgrades over baseline_stdlib.py:
- Learns logistic regression weights with L2 regularization (batch GD).
- Uses missing indicators (NaNs are informative).
- Uses high_conf_clean=1 unlabeled rows as *weak negatives* (lower weight).
- Adds lightweight graph features from social_graph.csv:
  - degree in the social graph (restricted to nodes in train/test)
  - fraction of known-labeled neighbors that are cheating (uses labeled + weak negatives)

Also includes a local eval helper that approximates Kaggle's cost metric by
searching thresholds on labeled rows only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


EPS = 1e-12


def sigmoid(x: float) -> float:
    # clip for stability
    x = max(min(x, 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-x))


def safe_float(s: str) -> Optional[float]:
    s = s.strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def iter_csv(path: str) -> Iterable[List[str]]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            yield row


def read_header(path: str) -> List[str]:
    it = iter_csv(path)
    return next(it)


def index_map(header: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(header)}


def feature_columns(header: List[str]) -> List[str]:
    return [c for c in header if c.startswith("feature_")]


def autodetect_paths(train: str, test: str, graph: str) -> Tuple[str, str, str]:
    candidates = [
        ("sample-data/train.csv", "sample-data/test.csv", "sample-data/social_graph.csv"),
        ("data/train.csv", "data/test.csv", "data/social_graph.csv"),
    ]
    if os.path.exists(train) and os.path.exists(test) and os.path.exists(graph):
        return train, test, graph
    for t, te, g in candidates:
        if os.path.exists(t) and os.path.exists(te) and os.path.exists(g):
            return t, te, g
    # fall back to user inputs (will error later with clearer message)
    return train, test, graph


@dataclass
class GraphFeatures:
    degree: Dict[str, int]
    known_neighbor_cnt: Dict[str, int]
    known_neighbor_cheat_sum: Dict[str, float]


def build_interest_set(train_path: str, test_path: str) -> Set[str]:
    users: Set[str] = set()
    # train
    h = read_header(train_path)
    idx = index_map(h)
    if "user_hash" not in idx:
        raise RuntimeError("train.csv missing user_hash")
    uidx = idx["user_hash"]
    for row in iter_csv(train_path):
        if row and row[0] == "user_hash":
            continue
        if uidx < len(row):
            users.add(row[uidx])
    # test
    h = read_header(test_path)
    idx = index_map(h)
    if "user_hash" not in idx:
        raise RuntimeError("test.csv missing user_hash")
    uidx = idx["user_hash"]
    for row in iter_csv(test_path):
        if row and row[0] == "user_hash":
            continue
        if uidx < len(row):
            users.add(row[uidx])
    return users


def build_label_map_for_graph(train_path: str) -> Dict[str, float]:
    """
    Label map used only for graph neighbor aggregates.
    - Labeled rows: is_cheating in {0,1}
    - Unlabeled rows with high_conf_clean=1: treat as 0 (weak negative)
    - Other unlabeled: ignored (no entry)
    """
    header = read_header(train_path)
    idx = index_map(header)
    for col in ("user_hash", "is_cheating", "high_conf_clean"):
        if col not in idx:
            raise RuntimeError(f"train.csv missing required column: {col}")
    uidx = idx["user_hash"]
    yidx = idx["is_cheating"]
    cidx = idx["high_conf_clean"]

    labels: Dict[str, float] = {}
    it = iter_csv(train_path)
    _ = next(it)
    for row in it:
        u = row[uidx]
        y_raw = row[yidx].strip() if yidx < len(row) else ""
        if y_raw != "" and y_raw.lower() != "nan":
            y = safe_float(y_raw)
            if y is None:
                continue
            if y in (0.0, 1.0):
                labels[u] = y
            continue
        # unlabeled
        hc = row[cidx].strip() if cidx < len(row) else ""
        if hc != "" and hc.lower() != "nan":
            # treat as known clean for graph aggregates
            labels[u] = 0.0
    return labels


def compute_graph_features(
    graph_path: str, interest: Set[str], label_map: Dict[str, float]
) -> GraphFeatures:
    """
    Streams edges and computes degree + known-neighbor cheat fraction.
    We do not store adjacency (keeps memory small).
    """
    degree: Dict[str, int] = {}
    known_cnt: Dict[str, int] = {}
    known_cheat_sum: Dict[str, float] = {}

    header = read_header(graph_path)
    idx = index_map(header)
    if "user_a" not in idx or "user_b" not in idx:
        raise RuntimeError("social_graph.csv must contain user_a,user_b")
    aidx = idx["user_a"]
    bidx = idx["user_b"]

    it = iter_csv(graph_path)
    _ = next(it)
    for row in it:
        if not row:
            continue
        a = row[aidx]
        b = row[bidx]

        a_in = a in interest
        b_in = b in interest

        if a_in:
            degree[a] = degree.get(a, 0) + 1
            if b in label_map:
                known_cnt[a] = known_cnt.get(a, 0) + 1
                known_cheat_sum[a] = known_cheat_sum.get(a, 0.0) + label_map[b]
        if b_in:
            degree[b] = degree.get(b, 0) + 1
            if a in label_map:
                known_cnt[b] = known_cnt.get(b, 0) + 1
                known_cheat_sum[b] = known_cheat_sum.get(b, 0.0) + label_map[a]

    return GraphFeatures(degree=degree, known_neighbor_cnt=known_cnt, known_neighbor_cheat_sum=known_cheat_sum)


def build_feature_schema(train_path: str) -> Tuple[List[str], List[int], int, int]:
    header = read_header(train_path)
    idx = index_map(header)
    feats = feature_columns(header)
    feat_idxs = [idx[c] for c in feats]
    if "user_hash" not in idx or "is_cheating" not in idx or "high_conf_clean" not in idx:
        raise RuntimeError("train.csv missing required columns")
    return feats, feat_idxs, idx["user_hash"], idx["is_cheating"]


def vectorize_row(
    row: List[str],
    feat_idxs: List[int],
    graph: GraphFeatures,
    user_hash: str,
) -> List[float]:
    # 18 raw values (imputed to 0) + 18 missing flags
    x: List[float] = []
    for col_i in feat_idxs:
        v = safe_float(row[col_i]) if col_i < len(row) else None
        x.append(0.0 if v is None else v)
    for col_i in feat_idxs:
        v = safe_float(row[col_i]) if col_i < len(row) else None
        x.append(1.0 if v is None else 0.0)

    # Graph features (3)
    deg = float(graph.degree.get(user_hash, 0))
    known = float(graph.known_neighbor_cnt.get(user_hash, 0))
    cheat_sum = float(graph.known_neighbor_cheat_sum.get(user_hash, 0.0))
    frac = cheat_sum / max(known, 1.0)
    x.extend([deg, known, frac])
    return x


def standardize_fit(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    d = len(X[0])
    mean = [0.0] * d
    var = [0.0] * d
    n = len(X)
    for x in X:
        for j in range(d):
            mean[j] += x[j]
    for j in range(d):
        mean[j] /= max(n, 1)
    for x in X:
        for j in range(d):
            z = x[j] - mean[j]
            var[j] += z * z
    for j in range(d):
        var[j] = var[j] / max(n, 1)
    std = [math.sqrt(v) if v > 0 else 1.0 for v in var]
    return mean, std


def standardize_apply(x: List[float], mean: List[float], std: List[float]) -> List[float]:
    return [(xj - mean[j]) / std[j] for j, xj in enumerate(x)]


def dot(w: List[float], x: List[float]) -> float:
    s = 0.0
    for j in range(len(w)):
        s += w[j] * x[j]
    return s


def train_logreg(
    X: List[List[float]],
    y: List[int],
    sample_w: List[float],
    l2: float = 1e-2,
    lr: float = 0.1,
    epochs: int = 30,
    seed: int = 1337,
) -> List[float]:
    """
    Batch gradient descent on weighted logistic loss with L2.
    """
    random.seed(seed)
    n = len(X)
    d = len(X[0])
    w = [0.0] * (d + 1)  # bias + weights

    # simple learning-rate decay
    for ep in range(epochs):
        g0 = 0.0
        g = [0.0] * d
        loss = 0.0

        for i in range(n):
            xi = X[i]
            yi = y[i]
            wi = sample_w[i]
            z = w[0] + dot(w[1:], xi)
            pi = sigmoid(z)
            # gradient of logloss
            diff = (pi - yi) * wi
            g0 += diff
            for j in range(d):
                g[j] += diff * xi[j]
            # loss for debug
            loss += wi * (-(yi * math.log(pi + EPS) + (1 - yi) * math.log(1 - pi + EPS)))

        # L2 (not on bias)
        for j in range(d):
            g[j] += l2 * w[j + 1]
            loss += 0.5 * l2 * w[j + 1] * w[j + 1]

        step = lr / (1.0 + 0.05 * ep)
        w[0] -= step * (g0 / max(n, 1))
        for j in range(d):
            w[j + 1] -= step * (g[j] / max(n, 1))

    return w


def predict_proba(w: List[float], x: List[float]) -> float:
    return sigmoid(w[0] + dot(w[1:], x))


def eval_min_cost_score(probs: List[float], y: List[int], grid: int = 200) -> float:
    """
    Approximates Kaggle metric by searching thresholds (t1 < t2) on a grid of
    quantiles of probs. Returns score = -min_cost (higher is better).

    Costs:
    - FN (cheating in auto-pass): 600
    - FP in auto-block: 300
    - FP in manual review: 150
    - TP in manual review: 5
    - Correct: 0
    """
    n = len(probs)
    if n == 0:
        return 0.0
    paired = list(zip(probs, y))
    paired.sort(key=lambda t: t[0])  # ascending
    p_sorted = [p for p, _ in paired]
    y_sorted = [yy for _, yy in paired]

    # thresholds as quantiles
    thr: List[float] = []
    for k in range(grid):
        idx = int((k / (grid - 1)) * (n - 1))
        thr.append(p_sorted[idx])
    # unique & sorted
    thr = sorted(set(thr))

    # Precompute cumulative counts from low to high
    cum_pos = [0] * (n + 1)
    cum_neg = [0] * (n + 1)
    for i in range(n):
        cum_pos[i + 1] = cum_pos[i] + (1 if y_sorted[i] == 1 else 0)
        cum_neg[i + 1] = cum_neg[i] + (1 if y_sorted[i] == 0 else 0)

    def counts_ge(t: float) -> Tuple[int, int]:
        # first index where p >= t
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if p_sorted[mid] >= t:
                hi = mid
            else:
                lo = mid + 1
        i0 = lo
        pos = cum_pos[n] - cum_pos[i0]
        neg = cum_neg[n] - cum_neg[i0]
        return pos, neg

    def counts_lt(t: float) -> Tuple[int, int]:
        # first index where p >= t
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if p_sorted[mid] >= t:
                hi = mid
            else:
                lo = mid + 1
        i0 = lo
        pos = cum_pos[i0]
        neg = cum_neg[i0]
        return pos, neg

    best = float("inf")
    # Iterate t1,t2 on thr grid
    for i, t1 in enumerate(thr):
        # Auto-pass: p < t1
        ap_pos, ap_neg = counts_lt(t1)  # pos here are FNs
        fn = ap_pos
        base_cost = 600 * fn  # auto-pass correct negatives cost 0
        if base_cost >= best:
            continue
        for t2 in thr[i:]:
            # Auto-block: p >= t2
            ab_pos, ab_neg = counts_ge(t2)
            fp_block = ab_neg
            # Manual review: t1 <= p < t2
            # Compute as <t2 minus <t1
            lt2_pos, lt2_neg = counts_lt(t2)
            mr_pos = lt2_pos - ap_pos
            mr_neg = lt2_neg - ap_neg
            tp_review = mr_pos
            fp_review = mr_neg

            cost = base_cost + 300 * fp_block + 150 * fp_review + 5 * tp_review
            if cost < best:
                best = cost

    return -best


def load_training_data(
    train_path: str,
    feats: List[str],
    feat_idxs: List[int],
    graph: GraphFeatures,
    pos_weight: float = 3.0,
    weak_neg_weight: float = 0.25,
    seed: int = 1337,
) -> Tuple[List[List[float]], List[int], List[float], List[List[float]], List[int]]:
    """
    Returns:
      X_train, y_train, w_train, X_labeled, y_labeled
    We train on labeled rows + weak negatives.
    We separately return labeled rows for local eval.
    """
    random.seed(seed)
    header = read_header(train_path)
    idx = index_map(header)
    uidx = idx["user_hash"]
    yidx = idx["is_cheating"]
    cidx = idx["high_conf_clean"]

    X: List[List[float]] = []
    y: List[int] = []
    w: List[float] = []
    X_l: List[List[float]] = []
    y_l: List[int] = []

    it = iter_csv(train_path)
    _ = next(it)
    for row in it:
        u = row[uidx]
        x = vectorize_row(row, feat_idxs, graph, u)

        y_raw = row[yidx].strip() if yidx < len(row) else ""
        if y_raw != "" and y_raw.lower() != "nan":
            yy = safe_float(y_raw)
            if yy is None:
                continue
            if yy not in (0.0, 1.0):
                continue
            yi = int(yy)
            X.append(x)
            y.append(yi)
            w.append(pos_weight if yi == 1 else 1.0)
            X_l.append(x)
            y_l.append(yi)
            continue

        hc = row[cidx].strip() if cidx < len(row) else ""
        if hc != "" and hc.lower() != "nan":
            # weak negative
            X.append(x)
            y.append(0)
            w.append(weak_neg_weight)

    if not X_l:
        raise RuntimeError("No labeled rows found to train/evaluate.")
    return X, y, w, X_l, y_l


def write_submission(
    test_path: str,
    out_path: str,
    feats: List[str],
    feat_idxs: List[int],
    graph: GraphFeatures,
    mean: List[float],
    std: List[float],
    w: List[float],
) -> None:
    header = read_header(test_path)
    idx = index_map(header)
    if "user_hash" not in idx:
        raise RuntimeError("test.csv missing required column: user_hash")
    uidx = idx["user_hash"]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    it = iter_csv(test_path)
    _ = next(it)
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["user_hash", "prediction"])
        for row in it:
            u = row[uidx]
            x = vectorize_row(row, feat_idxs, graph, u)
            xs = standardize_apply(x, mean, std)
            p = predict_proba(w, xs)
            writer.writerow([u, f"{p:.10f}"])


def validate_submission(submission_path: str, test_path: str) -> None:
    test_it = iter_csv(test_path)
    test_header = next(test_it)
    t_idx = index_map(test_header)
    uidx = t_idx.get("user_hash", None)
    if uidx is None:
        raise ValueError("test.csv missing user_hash")
    expected: Set[str] = set()
    for row in test_it:
        if uidx < len(row):
            expected.add(row[uidx])

    sub_it = iter_csv(submission_path)
    header = next(sub_it)
    if header != ["user_hash", "prediction"]:
        raise ValueError(f"Bad submission header: {header!r}")

    seen: Set[str] = set()
    for row in sub_it:
        if len(row) < 2:
            raise ValueError("Row has <2 columns")
        u = row[0]
        if u in seen:
            raise ValueError(f"Duplicate user_hash: {u}")
        seen.add(u)
        p = safe_float(row[1])
        if p is None or p < 0 or p > 1:
            raise ValueError(f"Bad probability for {u}: {row[1]!r}")

    if seen != expected:
        missing = expected - seen
        extra = seen - expected
        if missing:
            raise ValueError(f"Missing {len(missing)} hashes (e.g. {sorted(list(missing))[:5]})")
        if extra:
            raise ValueError(f"Extra {len(extra)} hashes (e.g. {sorted(list(extra))[:5]})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="sample-data/train.csv")
    ap.add_argument("--test", default="sample-data/test.csv")
    ap.add_argument("--graph", default="sample-data/social_graph.csv")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--eval", action="store_true", help="Print local labeled-only cost score")
    ap.add_argument("--pos-weight", type=float, default=3.0, help="Upweight positives in training")
    ap.add_argument("--weak-neg-weight", type=float, default=0.25, help="Weight for high_conf_clean weak negatives")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=35)
    args = ap.parse_args()

    train_path, test_path, graph_path = autodetect_paths(args.train, args.test, args.graph)
    for p in (train_path, test_path, graph_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    # Graph features
    interest = build_interest_set(train_path, test_path)
    label_map = build_label_map_for_graph(train_path)
    graph = compute_graph_features(graph_path, interest, label_map)

    # Feature schema from train
    header = read_header(train_path)
    idx = index_map(header)
    feats = feature_columns(header)
    feat_idxs = [idx[c] for c in feats]

    X, y, sw, X_l, y_l = load_training_data(
        train_path,
        feats,
        feat_idxs,
        graph,
        pos_weight=args.pos_weight,
        weak_neg_weight=args.weak_neg_weight,
    )

    mean, std = standardize_fit(X)
    Xs = [standardize_apply(x, mean, std) for x in X]
    w = train_logreg(Xs, y, sw, l2=args.l2, lr=args.lr, epochs=args.epochs)

    if args.eval:
        Xl_s = [standardize_apply(x, mean, std) for x in X_l]
        probs = [predict_proba(w, x) for x in Xl_s]
        score = eval_min_cost_score(probs, y_l, grid=220)
        print(f"Local labeled-only score (approx): {score:.2f}")

    write_submission(test_path, args.out, feats, feat_idxs, graph, mean, std, w)
    if args.validate:
        validate_submission(args.out, test_path)
        print("Validation: OK")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


