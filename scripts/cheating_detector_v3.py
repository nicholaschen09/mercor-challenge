#!/usr/bin/env python3
"""
Cheating Detection V3 - MAXIMUM GRAPH POWER
============================================
Double down on graph features since they're working!

Key graph signals:
1. Direct cheater neighbor count & rate
2. Clean neighbor count & rate (inverse signal)
3. Cheater-to-clean ratio among neighbors
4. Multiple cheater connection thresholds
5. Isolation score (only cheater friends?)
6. Neighbor quality scores
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

FEATURE_COLS = [f"feature_{i:03d}" for i in range(1, 19)]
RANDOM_STATE = 42
N_FOLDS = 5

# Competition cost structure
COST_FN = 600    # Miss a cheater (False Negative)
COST_FP = 300    # Block innocent (False Positive in auto-block)
COST_REVIEW = 150  # Manual review


def simulate_competition_cost(y_true, y_pred):
    """
    Simulate the competition's cost-based metric.
    Finds optimal thresholds (t_low, t_high) that minimize total cost.

    Decision regions:
    - p <= t_low: Auto-pass (predict clean)
    - t_low < p < t_high: Manual review
    - p >= t_high: Auto-block (predict cheater)

    KEY INSIGHT:
    - Missing cheater (FN) = $600
    - Manual review = $150
    - So review is 4x cheaper than missing a cheater!
    - Be VERY conservative with auto-pass
    """
    best_cost = float('inf')
    best_t_low = 0.05
    best_t_high = 0.7

    # Grid search - find TRUE optimal (not just conservative)
    for t_low in np.arange(0.01, 0.70, 0.01):
        for t_high in np.arange(0.30, 0.99, 0.01):
            if t_high <= t_low:
                continue

            cost = 0
            n_auto_pass = 0
            n_review = 0
            n_auto_block = 0
            n_fn = 0
            n_fp = 0

            for y, p in zip(y_true, y_pred):
                if p <= t_low:
                    # Auto-pass - ONLY if very confident they're clean
                    n_auto_pass += 1
                    if y == 1:  # Missed cheater!
                        cost += COST_FN
                        n_fn += 1
                elif p >= t_high:
                    # Auto-block
                    n_auto_block += 1
                    if y == 0:  # Blocked innocent!
                        cost += COST_FP
                        n_fp += 1
                else:
                    # Manual review - this is the SAFE option
                    n_review += 1
                    cost += COST_REVIEW

            if cost < best_cost:
                best_cost = cost
                best_t_low = t_low
                best_t_high = t_high
                best_breakdown = {
                    'auto_pass': n_auto_pass,
                    'review': n_review,
                    'auto_block': n_auto_block,
                    'false_negatives': n_fn,
                    'false_positives': n_fp
                }

    return best_cost, best_t_low, best_t_high, best_breakdown


def load_data(train_path, test_path, graph_path):
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    graph = pd.read_csv(graph_path) if Path(graph_path).exists() else None
    print(f"  Train: {len(train):,}, Test: {len(test):,}, Edges: {len(graph) if graph is not None else 0:,}")
    return train, test, graph


def build_maximum_graph_features(train, test, graph_df):
    """MAXIMUM graph features - all vectorized for speed."""
    if graph_df is None:
        return train, test

    print("\n" + "="*50)
    print("BUILDING MAXIMUM GRAPH FEATURES")
    print("="*50)

    # Setup
    all_users = pd.concat([train['user_hash'], test['user_hash']]).unique()
    user_df = pd.DataFrame({'user_hash': all_users})

    labeled = train[train['is_cheating'].notna()][['user_hash', 'is_cheating']].copy()
    cheater_set = set(labeled[labeled['is_cheating'] == 1]['user_hash'])
    clean_set = set(labeled[labeled['is_cheating'] == 0]['user_hash'])

    n_cheaters = len(cheater_set)
    n_clean = len(clean_set)
    base_rate = n_cheaters / (n_cheaters + n_clean)

    print(f"\n  Cheaters: {n_cheaters:,}, Clean: {n_clean:,}")
    print(f"  Base cheating rate: {base_rate:.3f}")

    # Build bidirectional edges
    print("\n  Building edge features...")
    edges = pd.concat([
        graph_df[['user_a', 'user_b']].rename(columns={'user_a': 'user', 'user_b': 'neighbor'}),
        graph_df[['user_a', 'user_b']].rename(columns={'user_b': 'user', 'user_a': 'neighbor'})
    ])

    # Tag each neighbor
    edges['neighbor_is_cheater'] = edges['neighbor'].isin(cheater_set).astype(int)
    edges['neighbor_is_clean'] = edges['neighbor'].isin(clean_set).astype(int)
    edges['neighbor_is_labeled'] = edges['neighbor_is_cheater'] + edges['neighbor_is_clean']
    edges['neighbor_is_unlabeled'] = 1 - edges['neighbor_is_labeled']

    # =========================================================================
    # CORE NEIGHBOR STATS
    # =========================================================================
    print("  Computing neighbor statistics...")

    neighbor_stats = edges.groupby('user').agg({
        'neighbor': 'count',  # degree
        'neighbor_is_cheater': 'sum',
        'neighbor_is_clean': 'sum',
        'neighbor_is_labeled': 'sum',
        'neighbor_is_unlabeled': 'sum'
    }).reset_index()

    neighbor_stats.columns = [
        'user_hash', 'graph_degree',
        'graph_n_cheater_neighbors', 'graph_n_clean_neighbors',
        'graph_n_labeled_neighbors', 'graph_n_unlabeled_neighbors'
    ]

    user_df = user_df.merge(neighbor_stats, on='user_hash', how='left')

    # Fill NaN for users with no connections
    for col in ['graph_degree', 'graph_n_cheater_neighbors', 'graph_n_clean_neighbors',
                'graph_n_labeled_neighbors', 'graph_n_unlabeled_neighbors']:
        user_df[col] = user_df[col].fillna(0).astype(int)

    # =========================================================================
    # CHEATER RATE FEATURES (Multiple formulations)
    # =========================================================================
    print("  Computing cheater rate features...")

    # Rate among labeled neighbors
    user_df['graph_cheat_rate_labeled'] = np.where(
        user_df['graph_n_labeled_neighbors'] > 0,
        user_df['graph_n_cheater_neighbors'] / user_df['graph_n_labeled_neighbors'],
        np.nan
    )

    # Rate among ALL neighbors (treating unlabeled as unknown)
    user_df['graph_cheat_rate_all'] = np.where(
        user_df['graph_degree'] > 0,
        user_df['graph_n_cheater_neighbors'] / user_df['graph_degree'],
        0
    )

    # Clean rate (inverse signal)
    user_df['graph_clean_rate_labeled'] = np.where(
        user_df['graph_n_labeled_neighbors'] > 0,
        user_df['graph_n_clean_neighbors'] / user_df['graph_n_labeled_neighbors'],
        np.nan
    )

    # Cheater to clean ratio (log scale for stability)
    user_df['graph_cheater_clean_ratio'] = np.log1p(user_df['graph_n_cheater_neighbors']) - np.log1p(user_df['graph_n_clean_neighbors'])

    # =========================================================================
    # THRESHOLD-BASED BINARY FEATURES
    # =========================================================================
    print("  Computing threshold features...")

    # Has ANY cheater neighbor
    user_df['graph_has_cheater_neighbor'] = (user_df['graph_n_cheater_neighbors'] > 0).astype(int)

    # Has MULTIPLE cheater neighbors (stronger signal)
    user_df['graph_has_2plus_cheaters'] = (user_df['graph_n_cheater_neighbors'] >= 2).astype(int)
    user_df['graph_has_3plus_cheaters'] = (user_df['graph_n_cheater_neighbors'] >= 3).astype(int)
    user_df['graph_has_5plus_cheaters'] = (user_df['graph_n_cheater_neighbors'] >= 5).astype(int)

    # High cheater rate thresholds
    user_df['graph_cheat_rate_gt_50'] = (user_df['graph_cheat_rate_labeled'] > 0.5).astype(int)
    user_df['graph_cheat_rate_gt_75'] = (user_df['graph_cheat_rate_labeled'] > 0.75).astype(int)

    # ONLY has cheater neighbors (no clean friends - suspicious!)
    user_df['graph_only_cheater_friends'] = (
        (user_df['graph_n_cheater_neighbors'] > 0) &
        (user_df['graph_n_clean_neighbors'] == 0)
    ).astype(int)

    # Has NO clean friends among labeled
    user_df['graph_no_clean_friends'] = (user_df['graph_n_clean_neighbors'] == 0).astype(int)

    # =========================================================================
    # ISOLATION / STRUCTURE FEATURES
    # =========================================================================
    print("  Computing isolation features...")

    # What % of neighbors are labeled?
    user_df['graph_labeled_neighbor_pct'] = np.where(
        user_df['graph_degree'] > 0,
        user_df['graph_n_labeled_neighbors'] / user_df['graph_degree'],
        0
    )

    # Is user isolated (low degree)?
    user_df['graph_is_isolated'] = (user_df['graph_degree'] <= 1).astype(int)

    # Log degree
    user_df['graph_log_degree'] = np.log1p(user_df['graph_degree'])

    # =========================================================================
    # CHEATER CLUSTER ANALYSIS
    # =========================================================================
    print("  Analyzing cheater clusters...")

    # Edges to cheaters
    cheater_edges_a = graph_df[graph_df['user_a'].isin(cheater_set)].groupby('user_b').size()
    cheater_edges_b = graph_df[graph_df['user_b'].isin(cheater_set)].groupby('user_a').size()

    cheater_edge_counts = cheater_edges_a.add(cheater_edges_b, fill_value=0).reset_index()
    cheater_edge_counts.columns = ['user_hash', 'graph_edges_to_cheaters']

    user_df = user_df.merge(cheater_edge_counts, on='user_hash', how='left')
    user_df['graph_edges_to_cheaters'] = user_df['graph_edges_to_cheaters'].fillna(0)

    # Edges to clean
    clean_edges_a = graph_df[graph_df['user_a'].isin(clean_set)].groupby('user_b').size()
    clean_edges_b = graph_df[graph_df['user_b'].isin(clean_set)].groupby('user_a').size()

    clean_edge_counts = clean_edges_a.add(clean_edges_b, fill_value=0).reset_index()
    clean_edge_counts.columns = ['user_hash', 'graph_edges_to_clean']

    user_df = user_df.merge(clean_edge_counts, on='user_hash', how='left')
    user_df['graph_edges_to_clean'] = user_df['graph_edges_to_clean'].fillna(0)

    # Dominance ratio
    user_df['graph_cheater_edge_dominance'] = (
        user_df['graph_edges_to_cheaters'] /
        (user_df['graph_edges_to_cheaters'] + user_df['graph_edges_to_clean'] + 1)
    )

    # =========================================================================
    # GUILT SCORES
    # =========================================================================
    print("  Computing guilt scores...")

    user_df['graph_guilt_v1'] = (
        0.5 * user_df['graph_cheat_rate_labeled'].fillna(base_rate) +
        0.3 * user_df['graph_cheat_rate_all'] +
        0.2 * user_df['graph_cheater_edge_dominance']
    )

    user_df['graph_guilt_v2'] = (
        0.4 * user_df['graph_cheat_rate_labeled'].fillna(base_rate) +
        0.3 * user_df['graph_has_cheater_neighbor'] +
        0.3 * user_df['graph_only_cheater_friends']
    )

    user_df['graph_guilt_v3'] = np.log1p(user_df['graph_n_cheater_neighbors']) / (np.log1p(user_df['graph_degree']) + 0.1)

    user_df['graph_risk_tier'] = (
        user_df['graph_has_cheater_neighbor'] +
        user_df['graph_has_2plus_cheaters'] +
        user_df['graph_cheat_rate_gt_50'] +
        user_df['graph_only_cheater_friends']
    )

    # =========================================================================
    # AGGRESSIVE CHEATER DETECTION FEATURES
    # =========================================================================
    print("  Computing aggressive detection features...")

    # Majority of labeled friends are cheaters
    user_df['graph_majority_cheaters'] = (user_df['graph_cheat_rate_labeled'] > 0.5).astype(int)

    # Strong cheater signal: 3+ cheater friends OR >75% rate
    user_df['graph_strong_cheater_signal'] = (
        (user_df['graph_n_cheater_neighbors'] >= 3) |
        (user_df['graph_cheat_rate_labeled'] > 0.75)
    ).astype(int)

    # Very suspicious: has cheaters but NO clean friends at all
    user_df['graph_very_suspicious'] = (
        (user_df['graph_n_cheater_neighbors'] > 0) &
        (user_df['graph_n_clean_neighbors'] == 0) &
        (user_df['graph_n_labeled_neighbors'] > 0)
    ).astype(int)

    # Cheater density: cheaters per connection
    user_df['graph_cheater_density'] = user_df['graph_n_cheater_neighbors'] / (user_df['graph_degree'] + 1)

    # Normalized cheater count (log scale)
    user_df['graph_log_cheater_neighbors'] = np.log1p(user_df['graph_n_cheater_neighbors'])

    # Difference between cheater and clean neighbors
    user_df['graph_cheater_minus_clean'] = user_df['graph_n_cheater_neighbors'] - user_df['graph_n_clean_neighbors']

    # Is this user well-connected to the "cheater network"?
    user_df['graph_cheater_network_strength'] = (
        user_df['graph_n_cheater_neighbors'] * user_df['graph_cheat_rate_labeled'].fillna(0)
    )

    # =========================================================================
    # MERGE
    # =========================================================================
    print("\n  Merging features...")

    graph_cols = [c for c in user_df.columns if c.startswith('graph_')]
    train = train.merge(user_df[['user_hash'] + graph_cols], on='user_hash', how='left')
    test = test.merge(user_df[['user_hash'] + graph_cols], on='user_hash', how='left')

    print(f"  Added {len(graph_cols)} graph features!")
    print(f"\n  Stats:")
    print(f"    Users with cheater neighbors: {(user_df['graph_has_cheater_neighbor'] == 1).sum():,}")
    print(f"    Users with 2+ cheater neighbors: {(user_df['graph_has_2plus_cheaters'] == 1).sum():,}")
    print(f"    Users with ONLY cheater friends: {(user_df['graph_only_cheater_friends'] == 1).sum():,}")

    return train, test


def engineer_basic_features(df):
    df = df.copy()
    df['missing_count'] = df[FEATURE_COLS].isna().sum(axis=1)
    for col in ['feature_010', 'feature_015']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].fillna(0).clip(lower=0))
    return df


def train_model(train_df, test_df, use_pseudo=True):
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)

    if use_pseudo:
        mask = (train_df['high_conf_clean'] == 1.0) & (train_df['is_cheating'].isna())
        if mask.sum() > 0:
            train_df = train_df.copy()
            train_df.loc[mask, 'is_cheating'] = 0.0
            print(f"\n  Added {mask.sum():,} pseudo-labels")

    labeled = train_df[train_df['is_cheating'].notna()].copy()

    exclude = {'user_hash', 'is_cheating', 'high_conf_clean', 'is_pseudo_label'}
    feature_cols = [c for c in labeled.columns
                   if c not in exclude and labeled[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    graph_feats = [c for c in feature_cols if c.startswith('graph_')]
    print(f"\n  Total features: {len(feature_cols)}, Graph features: {len(graph_feats)}")

    X = labeled[feature_cols].values
    y = labeled['is_cheating'].values
    X_test = test_df[feature_cols].values

    X = np.nan_to_num(X, nan=-999)
    X_test = np.nan_to_num(X_test, nan=-999)

    oof = np.zeros(len(labeled))
    test_preds = np.zeros(len(test_df))

    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS}", end=" ")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Balanced weight
        scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'max_depth': 8, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'min_child_samples': 30, 'verbose': -1, 'seed': RANDOM_STATE,
            'scale_pos_weight': scale_pos,
        }

        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(params, train_data, num_boost_round=500,
                         valid_sets=[val_data],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / N_FOLDS
        print(f"AUC: {roc_auc_score(y_val, oof[val_idx]):.4f}")

    # Feature importance
    imp = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_cols, imp), key=lambda x: -x[1])

    print("\n  Top 15 Features:")
    for i, (name, val) in enumerate(feat_imp[:15]):
        marker = " ***GRAPH***" if name.startswith('graph_') else ""
        print(f"    {i+1:2d}. {name}: {val:.0f}{marker}")

    auc = roc_auc_score(y, oof)
    print(f"\n  Overall AUC: {auc:.4f}")

    # Show prediction distribution by class
    print(f"\n  Prediction distribution:")
    print(f"    Cheaters (y=1): min={oof[y==1].min():.3f}, median={np.median(oof[y==1]):.3f}, max={oof[y==1].max():.3f}")
    print(f"    Clean (y=0):    min={oof[y==0].min():.3f}, median={np.median(oof[y==0]):.3f}, max={oof[y==0].max():.3f}")

    # Count cheaters at different probability thresholds
    print(f"\n  Cheaters by prediction threshold:")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        n_cheaters_below = (oof[y==1] < thresh).sum()
        print(f"    p < {thresh}: {n_cheaters_below:,} cheaters would be auto-passed")

    return test_preds, oof, y, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="sample-data/train.csv")
    parser.add_argument("--test", default="sample-data/test.csv")
    parser.add_argument("--graph", default="sample-data/social_graph.csv")
    parser.add_argument("--output", default="submission_v3.csv")
    args = parser.parse_args()

    for base in ["", "sample-data/", "data/"]:
        if Path(f"{base}train.csv").exists():
            args.train = f"{base}train.csv"
            args.test = f"{base}test.csv"
            args.graph = f"{base}social_graph.csv"
            break

    print("="*50)
    print("CHEATING DETECTOR V3 - MAXIMUM GRAPH POWER")
    print("="*50)

    train, test, graph = load_data(args.train, args.test, args.graph)

    if graph is not None:
        train, test = build_maximum_graph_features(train, test, graph)

    train = engineer_basic_features(train)
    test = engineer_basic_features(test)

    test_preds, oof_preds, y_true, auc = train_model(train, test)

    # =========================================================================
    # COST SIMULATION (Pre-submission assessment!)
    # =========================================================================
    print("\n" + "="*50)
    print("COST SIMULATION (CV Estimate)")
    print("="*50)

    cost, t_low, t_high, breakdown = simulate_competition_cost(y_true, oof_preds)

    # Scale to full dataset estimate (test is ~48K, train labeled is ~113K)
    # Competition score is negative cost
    estimated_score = -cost

    print(f"\n  Optimal thresholds: t_low={t_low:.2f}, t_high={t_high:.2f}")
    print(f"\n  Decision breakdown on CV:")
    print(f"    Auto-pass: {breakdown['auto_pass']:,} ({100*breakdown['auto_pass']/len(y_true):.1f}%)")
    print(f"    Manual review: {breakdown['review']:,} ({100*breakdown['review']/len(y_true):.1f}%)")
    print(f"    Auto-block: {breakdown['auto_block']:,} ({100*breakdown['auto_block']/len(y_true):.1f}%)")
    print(f"\n  Errors:")
    print(f"    False Negatives (missed cheaters): {breakdown['false_negatives']:,} x ${COST_FN} = ${breakdown['false_negatives']*COST_FN:,}")
    print(f"    False Positives (blocked innocent): {breakdown['false_positives']:,} x ${COST_FP} = ${breakdown['false_positives']*COST_FP:,}")
    print(f"    Review costs: {breakdown['review']:,} x ${COST_REVIEW} = ${breakdown['review']*COST_REVIEW:,}")
    print(f"\n  Total CV Cost: ${cost:,.0f}")
    print(f"  CV Score (negative cost): {estimated_score:,.0f}")

    # Estimate test score (test is ~42% size of train labeled)
    scale_factor = len(test) / len(y_true)
    estimated_test_cost = cost * scale_factor
    print(f"\n  Estimated Test Score: {-estimated_test_cost:,.0f} (scaled by {scale_factor:.2f})")

    # Boost predictions for suspicious users (optional)
    # If someone has cheater neighbors, boost their probability slightly
    # This makes us more conservative (send to review instead of auto-pass)

    # Save submission
    submission = pd.DataFrame({'user_hash': test['user_hash'], 'prediction': test_preds})
    submission.to_csv(args.output, index=False)

    print(f"\n" + "="*50)
    print(f"SAVED: {args.output}")
    print(f"AUC: {auc:.4f}")
    print(f"Estimated Score: {-estimated_test_cost:,.0f}")
    print("="*50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
