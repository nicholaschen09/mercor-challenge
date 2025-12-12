#!/usr/bin/env python3
"""
Cheating Detection V2 - FAST Guilt by Association
==================================================
Optimized for speed with vectorized operations.
"""

import argparse
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEATURE_COLS = [f"feature_{i:03d}" for i in range(1, 19)]
RANDOM_STATE = 42
N_FOLDS = 5


def load_data(train_path, test_path, graph_path):
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    graph = pd.read_csv(graph_path) if Path(graph_path).exists() else None
    print(f"  Train: {len(train):,}, Test: {len(test):,}, Edges: {len(graph) if graph is not None else 0:,}")
    return train, test, graph


def build_graph_features_fast(train, test, graph_df):
    """FAST graph features using pandas operations."""
    if graph_df is None:
        return train, test

    print("\nBuilding graph features (fast)...")

    # Get all users
    all_users = pd.concat([train['user_hash'], test['user_hash']]).unique()
    user_df = pd.DataFrame({'user_hash': all_users})

    # Get labeled status
    labeled = train[train['is_cheating'].notna()][['user_hash', 'is_cheating']].copy()
    cheater_set = set(labeled[labeled['is_cheating'] == 1]['user_hash'])
    clean_set = set(labeled[labeled['is_cheating'] == 0]['user_hash'])

    base_rate = len(cheater_set) / (len(cheater_set) + len(clean_set))
    print(f"  Cheaters: {len(cheater_set):,}, Clean: {len(clean_set):,}, Base rate: {base_rate:.3f}")

    # Build adjacency using pandas (fast!)
    print("  Building adjacency...")
    edges = pd.concat([
        graph_df[['user_a', 'user_b']].rename(columns={'user_a': 'user', 'user_b': 'neighbor'}),
        graph_df[['user_a', 'user_b']].rename(columns={'user_b': 'user', 'user_a': 'neighbor'})
    ])

    # Degree
    degree = edges.groupby('user').size().reset_index(name='graph_degree')
    degree = degree.rename(columns={'user': 'user_hash'})
    user_df = user_df.merge(degree, on='user_hash', how='left')
    user_df['graph_degree'] = user_df['graph_degree'].fillna(0).astype(int)

    # Add cheater/clean flags to neighbors
    print("  Computing neighbor cheating rates...")
    edges['is_cheater'] = edges['neighbor'].isin(cheater_set).astype(int)
    edges['is_clean'] = edges['neighbor'].isin(clean_set).astype(int)
    edges['is_labeled'] = edges['is_cheater'] + edges['is_clean']

    # Aggregate by user
    neighbor_stats = edges.groupby('user').agg({
        'is_cheater': 'sum',
        'is_clean': 'sum',
        'is_labeled': 'sum'
    }).reset_index()
    neighbor_stats.columns = ['user', 'graph_cheater_neighbors', 'graph_clean_neighbors', 'graph_labeled_neighbors']
    neighbor_stats = neighbor_stats.rename(columns={'user': 'user_hash'})

    user_df = user_df.merge(neighbor_stats, on='user_hash', how='left')
    user_df['graph_cheater_neighbors'] = user_df['graph_cheater_neighbors'].fillna(0)
    user_df['graph_clean_neighbors'] = user_df['graph_clean_neighbors'].fillna(0)
    user_df['graph_labeled_neighbors'] = user_df['graph_labeled_neighbors'].fillna(0)

    # Cheating rate among labeled neighbors
    user_df['graph_1hop_cheat_rate'] = np.where(
        user_df['graph_labeled_neighbors'] > 0,
        user_df['graph_cheater_neighbors'] / user_df['graph_labeled_neighbors'],
        np.nan
    )

    # Binary features
    user_df['graph_has_cheater_neighbor'] = (user_df['graph_cheater_neighbors'] > 0).astype(int)
    user_df['graph_cheater_ratio'] = user_df['graph_cheater_neighbors'] / user_df['graph_degree'].replace(0, 1)

    # =========================================================================
    # CHEATER CLUSTER CONNECTIONS (Fast - using pandas)
    # =========================================================================
    print("  Computing cheater connections...")

    # Edges where user_a is a cheater
    cheater_a = graph_df[graph_df['user_a'].isin(cheater_set)].groupby('user_b').size()
    # Edges where user_b is a cheater
    cheater_b = graph_df[graph_df['user_b'].isin(cheater_set)].groupby('user_a').size()

    # Combine
    cheater_connections = cheater_a.add(cheater_b, fill_value=0).reset_index()
    cheater_connections.columns = ['user_hash', 'graph_cheater_cluster_connections']

    user_df = user_df.merge(cheater_connections, on='user_hash', how='left')
    user_df['graph_cheater_cluster_connections'] = user_df['graph_cheater_cluster_connections'].fillna(0)

    # =========================================================================
    # DERIVED FEATURES
    # =========================================================================
    print("  Computing derived features...")

    # Guilt score (based on 1-hop rate and cheater connections)
    user_df['graph_guilt_score'] = (
        0.6 * user_df['graph_1hop_cheat_rate'].fillna(base_rate) +
        0.4 * (user_df['graph_cheater_cluster_connections'] / user_df['graph_degree'].replace(0, 1)).clip(0, 1)
    )

    # Risk tier
    user_df['graph_risk_tier'] = (
        (user_df['graph_has_cheater_neighbor']).astype(int) +
        (user_df['graph_guilt_score'] > 0.5).astype(int) +
        (user_df['graph_cheater_cluster_connections'] > 2).astype(int)
    )

    # High guilt flag
    user_df['graph_high_guilt'] = (user_df['graph_1hop_cheat_rate'] > 0.5).astype(int)

    # =========================================================================
    # MERGE TO TRAIN/TEST
    # =========================================================================
    print("  Merging features...")

    graph_cols = [c for c in user_df.columns if c.startswith('graph_')]
    train = train.merge(user_df[['user_hash'] + graph_cols], on='user_hash', how='left')
    test = test.merge(user_df[['user_hash'] + graph_cols], on='user_hash', how='left')

    print(f"  Added {len(graph_cols)} graph features")

    return train, test


def engineer_features(df):
    df = df.copy()

    # Missingness
    df['missing_count'] = df[FEATURE_COLS].isna().sum(axis=1)
    for col in FEATURE_COLS:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    # Aggregations
    numeric = df[FEATURE_COLS]
    df['feature_mean'] = numeric.mean(axis=1)
    df['feature_std'] = numeric.std(axis=1)

    # Log transforms
    for col in ['feature_010', 'feature_015']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].fillna(0).clip(lower=0))

    return df


def train_models(train_df, test_df, use_pseudo=True):
    print("\nTraining models...")

    if use_pseudo:
        mask = (train_df['high_conf_clean'] == 1.0) & (train_df['is_cheating'].isna())
        if mask.sum() > 0:
            train_df = train_df.copy()
            train_df.loc[mask, 'is_cheating'] = 0.0
            print(f"  Added {mask.sum():,} pseudo-labels")

    labeled = train_df[train_df['is_cheating'].notna()].copy()

    exclude = {'user_hash', 'is_cheating', 'high_conf_clean', 'is_pseudo_label'}
    feature_cols = [c for c in labeled.columns if c not in exclude and labeled[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"  Features: {len(feature_cols)}")

    X = labeled[feature_cols].values
    y = labeled['is_cheating'].values
    X_test = test_df[feature_cols].values

    X = np.nan_to_num(X, nan=-999)
    X_test = np.nan_to_num(X_test, nan=-999)

    oof = np.zeros(len(labeled))
    test_preds = np.zeros(len(test_df))

    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"  Fold {fold + 1}/{N_FOLDS}", end=" ")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1) * 1.5

        if HAS_LGB:
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
                             valid_sets=[val_data], callbacks=[
                                 lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)
                             ])

            oof[val_idx] = model.predict(X_val)
            test_preds += model.predict(X_test) / N_FOLDS

        print(f"AUC: {roc_auc_score(y_val, oof[val_idx]):.4f}")

    # Feature importance
    if HAS_LGB:
        imp = model.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:10]
        print("\nTop 10 Features:")
        for i, (name, val) in enumerate(feat_imp):
            print(f"  {i+1:2d}. {name}: {val:.0f}")

    auc = roc_auc_score(y, oof)
    print(f"\nOverall AUC: {auc:.4f}")

    return test_preds, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="sample-data/train.csv")
    parser.add_argument("--test", default="sample-data/test.csv")
    parser.add_argument("--graph", default="sample-data/social_graph.csv")
    parser.add_argument("--output", default="submission_v2.csv")
    args = parser.parse_args()

    for base in ["", "sample-data/", "data/"]:
        if Path(f"{base}train.csv").exists():
            args.train = f"{base}train.csv"
            args.test = f"{base}test.csv"
            args.graph = f"{base}social_graph.csv"
            break

    print("="*50)
    print("CHEATING DETECTOR V2 - GUILT BY ASSOCIATION")
    print("="*50)

    train, test, graph = load_data(args.train, args.test, args.graph)

    if graph is not None:
        train, test = build_graph_features_fast(train, test, graph)

    train = engineer_features(train)
    test = engineer_features(test)

    test_preds, auc = train_models(train, test)

    submission = pd.DataFrame({'user_hash': test['user_hash'], 'prediction': test_preds})
    submission.to_csv(args.output, index=False)

    print(f"\nSaved: {args.output}")
    print(f"AUC: {auc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
