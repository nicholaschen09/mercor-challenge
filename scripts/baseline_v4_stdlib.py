#!/usr/bin/env python3
"""
Mercor Cheating Detection — baseline v4 (stdlib only, no third-party deps).

What’s better vs v2:
- Stronger graph features via TWO streaming passes over social_graph.csv:
  - degree / log1p(degree)
  - labeled-neighbor cheat fraction (using labeled + high_conf_clean weak negatives)
  - avg neighbor cheat-fraction (a cheap “2-hop” signal)
  - avg neighbor degree (connectivity context)
- Better tabular preprocessing:
  - winsorization (clip) per feature using sampled percentiles
  - signed log1p transform for heavy-tailed values
  - missing indicators
- Better training:
  - weighted logistic regression trained with mini-batch Adam + L2
  - early stopping using a validation split and an approximate Kaggle cost search
  - optional lightweight hyperparameter tuning

This stays pure Python so it works in minimal environments.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


EPS = 1e-12


def sigmoid(x: float) -> float:
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


def signed_log1p(x: float) -> float:
    if x >= 0:
        return math.log1p(x)
    return -math.log1p(-x)


def iter_csv(path: str) -> Iterable[List[str]]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            yield row


def read_header(path: str) -> List[str]:
    it = iter_csv(path)
    return next(it)


def index_map(header: Sequence[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(header)}


def feature_columns(header: Sequence[str]) -> List[str]:
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
    return train, test, graph


@dataclass
class GraphAgg:
    degree: Dict[str, int]
    known_neighbor_cnt: Dict[str, int]
    known_neighbor_cheat_sum: Dict[str, float]
    # second pass
    neigh_cheatfrac_sum: Dict[str, float]
    neigh_cnt: Dict[str, int]
    neigh_deg_sum: Dict[str, int]


def build_interest_set(train_path: str, test_path: str) -> Set[str]:
    users: Set[str] = set()
    for path in (train_path, test_path):
        h = read_header(path)
        idx = index_map(h)
        if "user_hash" not in idx:
            raise RuntimeError(f"{path} missing user_hash")
        uidx = idx["user_hash"]
        it = iter_csv(path)
        _ = next(it)
        for row in it:
            if uidx < len(row):
                users.add(row[uidx])
    return users


def build_label_map_for_graph(train_path: str) -> Dict[str, float]:
    """
    For graph neighbor aggregates:
    - labeled rows: is_cheating in {0,1}
    - unlabeled rows with high_conf_clean=1: treat as 0
    """
    header = read_header(train_path)
    idx = index_map(header)
    for col in ("user_hash", "is_cheating", "high_conf_clean"):
        if col not in idx:
            raise RuntimeError(f"train.csv missing required column: {col}")
    uidx, yidx, cidx = idx["user_hash"], idx["is_cheating"], idx["high_conf_clean"]

    label: Dict[str, float] = {}
    it = iter_csv(train_path)
    _ = next(it)
    for row in it:
        u = row[uidx]
        y_raw = row[yidx].strip() if yidx < len(row) else ""
        if y_raw != "" and y_raw.lower() != "nan":
            yv = safe_float(y_raw)
            if yv in (0.0, 1.0):
                label[u] = yv
            continue
        hc = row[cidx].strip() if cidx < len(row) else ""
        if hc != "" and hc.lower() != "nan":
            label[u] = 0.0
    return label


def graph_pass1(graph_path: str, interest: Set[str], label_map: Dict[str, float]) -> GraphAgg:
    header = read_header(graph_path)
    idx = index_map(header)
    if "user_a" not in idx or "user_b" not in idx:
        raise RuntimeError("social_graph.csv must contain user_a,user_b")
    aidx, bidx = idx["user_a"], idx["user_b"]

    degree: Dict[str, int] = {}
    known_cnt: Dict[str, int] = {}
    known_sum: Dict[str, float] = {}

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
                known_sum[a] = known_sum.get(a, 0.0) + label_map[b]
        if b_in:
            degree[b] = degree.get(b, 0) + 1
            if a in label_map:
                known_cnt[b] = known_cnt.get(b, 0) + 1
                known_sum[b] = known_sum.get(b, 0.0) + label_map[a]

    return GraphAgg(
        degree=degree,
        known_neighbor_cnt=known_cnt,
        known_neighbor_cheat_sum=known_sum,
        neigh_cheatfrac_sum={},
        neigh_cnt={},
        neigh_deg_sum={},
    )


def graph_pass2(graph_path: str, interest: Set[str], agg: GraphAgg) -> None:
    # Precompute cheat fraction for every node we have known-neighbor stats for
    cheat_frac: Dict[str, float] = {}
    for u, cnt in agg.known_neighbor_cnt.items():
        s = agg.known_neighbor_cheat_sum.get(u, 0.0)
        cheat_frac[u] = s / max(float(cnt), 1.0)

    header = read_header(graph_path)
    idx = index_map(header)
    aidx, bidx = idx["user_a"], idx["user_b"]

    it = iter_csv(graph_path)
    _ = next(it)
    for row in it:
        if not row:
            continue
        a = row[aidx]
        b = row[bidx]
        a_in = a in interest
        b_in = b in interest

        # Update A using B as neighbor
        if a_in:
            agg.neigh_cnt[a] = agg.neigh_cnt.get(a, 0) + 1
            agg.neigh_deg_sum[a] = agg.neigh_deg_sum.get(a, 0) + agg.degree.get(b, 0)
            agg.neigh_cheatfrac_sum[a] = agg.neigh_cheatfrac_sum.get(a, 0.0) + cheat_frac.get(b, 0.0)
        # Update B using A as neighbor
        if b_in:
            agg.neigh_cnt[b] = agg.neigh_cnt.get(b, 0) + 1
            agg.neigh_deg_sum[b] = agg.neigh_deg_sum.get(b, 0) + agg.degree.get(a, 0)
            agg.neigh_cheatfrac_sum[b] = agg.neigh_cheatfrac_sum.get(b, 0.0) + cheat_frac.get(a, 0.0)


def sample_feature_clips(train_path: str, feat_idxs: List[int], max_samples: int, seed: int) -> Tuple[List[float], List[float]]:
    """
    Approximate per-feature clip bounds via random reservoir sampling.
    Returns (low, high) for each feature using approx 1st/99th percentiles.
    """
    random.seed(seed)
    nfeat = len(feat_idxs)
    samples: List[List[float]] = [[] for _ in range(nfeat)]

    it = iter_csv(train_path)
    header = next(it)
    idx = index_map(header)
    yidx = idx.get("is_cheating", None)
    # include labeled + weak negatives when estimating scaling (more robust than labeled-only)
    cidx = idx.get("high_conf_clean", None)

    seen = 0
    for row in it:
        if yidx is not None:
            y_raw = row[yidx].strip() if yidx < len(row) else ""
            labeled = y_raw != "" and y_raw.lower() != "nan"
        else:
            labeled = True
        weak = False
        if not labeled and cidx is not None:
            hc = row[cidx].strip() if cidx < len(row) else ""
            weak = hc != "" and hc.lower() != "nan"
        if not labeled and not weak:
            continue

        seen += 1
        for j, col_i in enumerate(feat_idxs):
            v = safe_float(row[col_i]) if col_i < len(row) else None
            if v is None:
                continue
            arr = samples[j]
            if len(arr) < max_samples:
                arr.append(v)
            else:
                # reservoir replace
                r = random.randint(0, seen - 1)
                if r < max_samples:
                    arr[r] = v

    low = [0.0] * nfeat
    high = [0.0] * nfeat
    for j in range(nfeat):
        arr = samples[j]
        if not arr:
            low[j], high[j] = -1.0, 1.0
            continue
        arr.sort()
        p1 = arr[int(0.01 * (len(arr) - 1))]
        p99 = arr[int(0.99 * (len(arr) - 1))]
        if p1 == p99:
            p1 -= 1.0
            p99 += 1.0
        low[j], high[j] = p1, p99
    return low, high


def vectorize_row(
    row: List[str],
    feat_idxs: List[int],
    clip_low: List[float],
    clip_high: List[float],
    agg: GraphAgg,
    user_hash: str,
) -> List[float]:
    # Raw clipped values + signed_log1p(clipped) + missing flags
    x: List[float] = []
    raw: List[float] = []
    miss: List[float] = []
    for j, col_i in enumerate(feat_idxs):
        v = safe_float(row[col_i]) if col_i < len(row) else None
        if v is None:
            raw.append(0.0)
            miss.append(1.0)
        else:
            v = max(min(v, clip_high[j]), clip_low[j])
            raw.append(v)
            miss.append(0.0)

    x.extend(raw)
    x.extend([signed_log1p(v) for v in raw])
    x.extend(miss)

    # Graph features (6)
    deg = float(agg.degree.get(user_hash, 0))
    known = float(agg.known_neighbor_cnt.get(user_hash, 0))
    cheat_sum = float(agg.known_neighbor_cheat_sum.get(user_hash, 0.0))
    cheat_frac = cheat_sum / max(known, 1.0)

    ncnt = float(agg.neigh_cnt.get(user_hash, 0))
    n_cheatfrac_sum = float(agg.neigh_cheatfrac_sum.get(user_hash, 0.0))
    n_deg_sum = float(agg.neigh_deg_sum.get(user_hash, 0))
    avg_n_cheatfrac = n_cheatfrac_sum / max(ncnt, 1.0)
    avg_n_deg = n_deg_sum / max(ncnt, 1.0)

    x.extend(
        [
            math.log1p(deg),
            deg,
            known,
            cheat_frac,
            avg_n_cheatfrac,
            math.log1p(avg_n_deg),
        ]
    )
    return x


def standardize_fit(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    d = len(X[0])
    n = len(X)
    mean = [0.0] * d
    var = [0.0] * d
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
    return [(x[j] - mean[j]) / std[j] for j in range(len(x))]


def dot(w: List[float], x: List[float]) -> float:
    s = 0.0
    for j in range(len(x)):
        s += w[j] * x[j]
    return s


def predict_proba(wb: List[float], x: List[float]) -> float:
    # wb = [bias] + weights
    return sigmoid(wb[0] + dot(wb[1:], x))


def eval_min_cost_score(probs: List[float], y: List[int], grid: int = 140) -> float:
    """
    Approximate Kaggle scoring: find best thresholds (t1 < t2) minimizing cost.
    Returns score = -min_cost.
    """
    n = len(probs)
    if n == 0:
        return 0.0
    paired = list(zip(probs, y))
    paired.sort(key=lambda t: t[0])
    p_sorted = [p for p, _ in paired]
    y_sorted = [yy for _, yy in paired]

    thr: List[float] = []
    for k in range(grid):
        idx = int((k / (grid - 1)) * (n - 1))
        thr.append(p_sorted[idx])
    thr = sorted(set(thr))

    cum_pos = [0] * (n + 1)
    cum_neg = [0] * (n + 1)
    for i in range(n):
        cum_pos[i + 1] = cum_pos[i] + (1 if y_sorted[i] == 1 else 0)
        cum_neg[i + 1] = cum_neg[i] + (1 if y_sorted[i] == 0 else 0)

    def first_ge(t: float) -> int:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if p_sorted[mid] >= t:
                hi = mid
            else:
                lo = mid + 1
        return lo

    best = float("inf")
    for i, t1 in enumerate(thr):
        i1 = first_ge(t1)
        ap_pos = cum_pos[i1]  # FN
        base = 600 * ap_pos
        if base >= best:
            continue
        for t2 in thr[i:]:
            i2 = first_ge(t2)
            # auto-block region: [i2, n)
            ab_neg = cum_neg[n] - cum_neg[i2]  # FP block
            # manual review region: [i1, i2)
            mr_pos = cum_pos[i2] - cum_pos[i1]  # TP review
            mr_neg = cum_neg[i2] - cum_neg[i1]  # FP review
            cost = base + 300 * ab_neg + 150 * mr_neg + 5 * mr_pos
            if cost < best:
                best = cost
    return -best


def adam_train(
    X: List[List[float]],
    y: List[int],
    w_sample: List[float],
    X_val: List[List[float]],
    y_val: List[int],
    *,
    l2: float,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    patience: int,
    eval_every: int,
) -> Tuple[List[float], float]:
    """
    Mini-batch Adam on weighted logistic loss with L2.
    Returns (best_weights, best_val_score).
    """
    random.seed(seed)
    n = len(X)
    d = len(X[0])
    wb = [0.0] * (d + 1)  # bias + weights
    m = [0.0] * (d + 1)
    v = [0.0] * (d + 1)
    b1, b2 = 0.9, 0.999
    t = 0

    best_wb = list(wb)
    best_score = float("-inf")
    bad = 0

    idxs = list(range(n))

    for ep in range(1, epochs + 1):
        random.shuffle(idxs)
        for start in range(0, n, batch_size):
            batch = idxs[start : start + batch_size]
            # gradients
            g = [0.0] * (d + 1)
            for i in batch:
                xi = X[i]
                yi = y[i]
                wi = w_sample[i]
                pi = predict_proba(wb, xi)
                diff = (pi - yi) * wi
                g[0] += diff
                for j in range(d):
                    g[j + 1] += diff * xi[j]
            bs = max(len(batch), 1)
            g[0] /= bs
            for j in range(d):
                g[j + 1] = (g[j + 1] / bs) + l2 * wb[j + 1]

            # adam step
            t += 1
            for j in range(d + 1):
                m[j] = b1 * m[j] + (1 - b1) * g[j]
                v[j] = b2 * v[j] + (1 - b2) * (g[j] * g[j])
                mhat = m[j] / (1 - (b1**t))
                vhat = v[j] / (1 - (b2**t))
                wb[j] -= lr * mhat / (math.sqrt(vhat) + 1e-8)

        if (ep % eval_every) == 0:
            probs = [predict_proba(wb, xv) for xv in X_val]
            score = eval_min_cost_score(probs, y_val, grid=140)
            if score > best_score + 1e-6:
                best_score = score
                best_wb = list(wb)
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    return best_wb, best_score


def split_labeled(
    X_l: List[List[float]], y_l: List[int], *, val_frac: float, seed: int
) -> Tuple[List[List[float]], List[int], List[List[float]], List[int]]:
    random.seed(seed)
    idxs = list(range(len(X_l)))
    random.shuffle(idxs)
    cut = int((1.0 - val_frac) * len(idxs))
    tr_idx = idxs[:cut]
    va_idx = idxs[cut:]
    Xtr = [X_l[i] for i in tr_idx]
    ytr = [y_l[i] for i in tr_idx]
    Xva = [X_l[i] for i in va_idx]
    yva = [y_l[i] for i in va_idx]
    return Xtr, ytr, Xva, yva


def load_train_rows(
    train_path: str,
    feat_idxs: List[int],
    clip_low: List[float],
    clip_high: List[float],
    agg: GraphAgg,
    *,
    pos_weight: float,
    weak_neg_weight: float,
) -> Tuple[List[List[float]], List[int], List[float], List[List[float]], List[int]]:
    """
    Returns:
      X_all, y_all, w_all  (labeled + weak negatives)
      X_labeled, y_labeled (labeled only)
    """
    header = read_header(train_path)
    idx = index_map(header)
    for col in ("user_hash", "is_cheating", "high_conf_clean"):
        if col not in idx:
            raise RuntimeError(f"train.csv missing {col}")
    uidx, yidx, cidx = idx["user_hash"], idx["is_cheating"], idx["high_conf_clean"]

    X: List[List[float]] = []
    y: List[int] = []
    ws: List[float] = []
    Xl: List[List[float]] = []
    yl: List[int] = []

    it = iter_csv(train_path)
    _ = next(it)
    for row in it:
        u = row[uidx]
        x = vectorize_row(row, feat_idxs, clip_low, clip_high, agg, u)

        y_raw = row[yidx].strip() if yidx < len(row) else ""
        if y_raw != "" and y_raw.lower() != "nan":
            yv = safe_float(y_raw)
            if yv not in (0.0, 1.0):
                continue
            yi = int(yv)
            X.append(x)
            y.append(yi)
            ws.append(pos_weight if yi == 1 else 1.0)
            Xl.append(x)
            yl.append(yi)
            continue

        hc = row[cidx].strip() if cidx < len(row) else ""
        if hc != "" and hc.lower() != "nan":
            X.append(x)
            y.append(0)
            ws.append(weak_neg_weight)

    if not Xl:
        raise RuntimeError("No labeled rows found in train.csv")
    return X, y, ws, Xl, yl


def write_submission(
    test_path: str,
    out_path: str,
    feat_idxs: List[int],
    clip_low: List[float],
    clip_high: List[float],
    agg: GraphAgg,
    mean: List[float],
    std: List[float],
    wb: List[float],
) -> None:
    header = read_header(test_path)
    idx = index_map(header)
    if "user_hash" not in idx:
        raise RuntimeError("test.csv missing user_hash")
    uidx = idx["user_hash"]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    it = iter_csv(test_path)
    _ = next(it)
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["user_hash", "prediction"])
        for row in it:
            u = row[uidx]
            x = vectorize_row(row, feat_idxs, clip_low, clip_high, agg, u)
            xs = standardize_apply(x, mean, std)
            p = predict_proba(wb, xs)
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
    ap.add_argument("--out", default="submission_v4.csv")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--eval", action="store_true", help="Print local labeled-only approximate score")
    ap.add_argument("--tune", action="store_true", help="Light hyperparameter tuning on a labeled val split")
    ap.add_argument("--seed", type=int, default=1337)

    # training knobs
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--weak-neg-weight", type=float, default=0.25)
    ap.add_argument("--l2", type=float, default=3e-2)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--clip-samples", type=int, default=20000)
    args = ap.parse_args()

    train_path, test_path, graph_path = autodetect_paths(args.train, args.test, args.graph)
    for p in (train_path, test_path, graph_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    # Schema
    header = read_header(train_path)
    idx = index_map(header)
    feats = feature_columns(header)
    feat_idxs = [idx[c] for c in feats]

    # Tabular clipping stats
    clip_low, clip_high = sample_feature_clips(train_path, feat_idxs, args.clip_samples, args.seed)

    # Graph aggregates (2 streaming passes)
    interest = build_interest_set(train_path, test_path)
    label_map = build_label_map_for_graph(train_path)
    agg = graph_pass1(graph_path, interest, label_map)
    graph_pass2(graph_path, interest, agg)

    # Load training data
    X_all, y_all, w_all, X_l, y_l = load_train_rows(
        train_path,
        feat_idxs,
        clip_low,
        clip_high,
        agg,
        pos_weight=args.pos_weight,
        weak_neg_weight=args.weak_neg_weight,
    )

    # Fit standardization on all training rows (labeled + weak neg) for stability
    mean, std = standardize_fit(X_all)
    X_all_s = [standardize_apply(x, mean, std) for x in X_all]
    X_l_s = [standardize_apply(x, mean, std) for x in X_l]

    # Split labeled for early stopping / tuning
    Xtr_l, ytr_l, Xva_l, yva_l = split_labeled(X_l_s, y_l, val_frac=args.val_frac, seed=args.seed)
    # We train on: (all weak neg + labeled-train subset). Validation: labeled-val subset.
    # Build a mask for labeled-train subset to select from X_l_s; easiest: use object id matching.
    tr_set = {id(x) for x in Xtr_l}

    X_train: List[List[float]] = []
    y_train: List[int] = []
    w_train: List[float] = []
    for x, yy, ww in zip(X_all_s, y_all, w_all):
        # if it's labeled, keep only if in labeled-train subset; if weak neg, always keep
        # We detect labeled vs weak neg by weight != weak_neg_weight when yy==0? not reliable.
        # Instead: check if this exact vector is in labeled pool. Works because we built X_all from X_l plus weak neg,
        # and X_l_s elements are the same list objects reused.
        if id(x) in tr_set:
            X_train.append(x)
            y_train.append(yy)
            w_train.append(ww)
        else:
            # include weak negatives (and any non-labeled rows) by checking presence in labeled pool
            # If x is a labeled-val vector, it will be in X_l_s but not in tr_set; we should skip it.
            # If x is weak negative, it won't appear in X_l_s at all. Since we don't have a fast set,
            # we approximate by: labeled vectors are those in X_l_s (object ids).
            pass

    labeled_ids = {id(x) for x in X_l_s}
    for x, yy, ww in zip(X_all_s, y_all, w_all):
        if id(x) in labeled_ids:
            continue  # handled above
        X_train.append(x)
        y_train.append(yy)
        w_train.append(ww)

    # Optional tuning: small grid over a few key knobs (fast enough on stdlib)
    best_params = (args.pos_weight, args.weak_neg_weight, args.l2, args.lr)
    best_score = float("-inf")

    trials = [(args.pos_weight, args.weak_neg_weight, args.l2, args.lr)]
    if args.tune:
        grid_pos = [3.0, 4.0, 6.0]
        grid_weak = [0.10, 0.25, 0.50]
        grid_l2 = [1e-2, 3e-2, 1e-1]
        grid_lr = [0.01, 0.03, 0.06]
        # sample a handful of combos deterministically
        combos: List[Tuple[float, float, float, float]] = []
        for pw in grid_pos:
            for wn in grid_weak:
                for l2 in grid_l2:
                    for lr in grid_lr:
                        combos.append((pw, wn, l2, lr))
        random.seed(args.seed)
        random.shuffle(combos)
        trials = combos[:10]

    for (pw, wn, l2, lr) in trials:
        # rebuild training sample weights with tuned params (cheap: only weights change)
        # Note: X_train itself stays the same vectors; only w_train changes for labeled positives / weak negs.
        # We recompute w_train by re-loading labels quickly from original arrays.
        X_all2, y_all2, w_all2, X_l2, y_l2 = load_train_rows(
            train_path,
            feat_idxs,
            clip_low,
            clip_high,
            agg,
            pos_weight=pw,
            weak_neg_weight=wn,
        )
        X_all2_s = [standardize_apply(x, mean, std) for x in X_all2]
        X_l2_s = [standardize_apply(x, mean, std) for x in X_l2]
        Xtr2, ytr2, Xva2, yva2 = split_labeled(X_l2_s, y_l2, val_frac=args.val_frac, seed=args.seed)
        tr2_set = {id(x) for x in Xtr2}
        labeled2_ids = {id(x) for x in X_l2_s}

        Xtr_final: List[List[float]] = []
        ytr_final: List[int] = []
        wtr_final: List[float] = []
        for x, yy, ww in zip(X_all2_s, y_all2, w_all2):
            if id(x) in tr2_set:
                Xtr_final.append(x)
                ytr_final.append(yy)
                wtr_final.append(ww)
        for x, yy, ww in zip(X_all2_s, y_all2, w_all2):
            if id(x) in labeled2_ids:
                continue
            Xtr_final.append(x)
            ytr_final.append(yy)
            wtr_final.append(ww)

        wb, val_score = adam_train(
            Xtr_final,
            ytr_final,
            wtr_final,
            Xva2,
            yva2,
            l2=l2,
            lr=lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            patience=args.patience,
            eval_every=2,
        )
        if val_score > best_score:
            best_score = val_score
            best_params = (pw, wn, l2, lr)

    # Train final model on ALL rows (labeled + weak neg) with best params (no early stop split)
    pw, wn, l2, lr = best_params
    X_all3, y_all3, w_all3, X_l3, y_l3 = load_train_rows(
        train_path,
        feat_idxs,
        clip_low,
        clip_high,
        agg,
        pos_weight=pw,
        weak_neg_weight=wn,
    )
    X_all3_s = [standardize_apply(x, mean, std) for x in X_all3]
    # use labeled split purely for early stopping signal
    X_l3_s = [standardize_apply(x, mean, std) for x in X_l3]
    Xtr3, ytr3, Xva3, yva3 = split_labeled(X_l3_s, y_l3, val_frac=args.val_frac, seed=args.seed)
    wb_final, _ = adam_train(
        X_all3_s,
        y_all3,
        w_all3,
        Xva3,
        yva3,
        l2=l2,
        lr=lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=args.patience,
        eval_every=2,
    )

    if args.eval:
        # labeled-only score
        probs = [predict_proba(wb_final, xv) for xv in X_l3_s]
        score = eval_min_cost_score(probs, y_l3, grid=160)
        print(f"Local labeled-only score (approx): {score:.2f}")
        print(f"Best params: pos_weight={pw}, weak_neg_weight={wn}, l2={l2}, lr={lr}")

    write_submission(test_path, args.out, feat_idxs, clip_low, clip_high, agg, mean, std, wb_final)
    if args.validate:
        validate_submission(args.out, test_path)
        print("Validation: OK")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


