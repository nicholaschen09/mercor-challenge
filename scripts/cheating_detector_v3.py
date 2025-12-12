#!/usr/bin/env python3
"""
Cheating Detection V2 - GUILT BY ASSOCIATION
=============================================

Heavy emphasis on graph-based features:
- Multi-hop neighbor cheating propagation
- Label propagation through the network
- Community/ring detection with cheating rates
- PageRank-style cheating influence
- Network centrality in cheating subgraph

Cost structure (from user):
- Miss cheater (FN): $600 (MOST EXPENSIVE!)
- Block innocent (FP): $300
- Flag innocent for review: $150
- Correct: $0

Strategy: Be aggressive about catching cheaters!

Usage:
    python cheating_detector_v2.py
"""

import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
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

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from community import community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

FEATURE_COLS = [f"feature_{i:03d}" for i in range(1, 19)]
RANDOM_STATE = 42
N_FOLDS = 5

# Updated cost parameters
COST_FALSE_NEGATIVE = 600.0  # Missing a cheater - MOST EXPENSIVE
COST_FALSE_POSITIVE_BLOCK = 300.0  # Wrongly auto-blocking
COST_MANUAL_REVIEW = 150.0  # Manual review cost


def load_data(train_path: str, test_path: str, graph_path: str):
    """Load all data files."""
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    graph = pd.read_csv(graph_path) if Path(graph_path).exists() else None

    print(f"  Train: {len(train):,} rows")
    print(f"  Test: {len(test):,} rows")
    if graph is not None:
        print(f"  Graph edges: {len(graph):,}")

    return train, test, graph


# =============================================================================
# GUILT BY ASSOCIATION - GRAPH FEATURES
# =============================================================================

def build_guilt_by_association_features(train: pd.DataFrame, test: pd.DataFrame,
                                         graph_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build extensive graph features based on GUILT BY ASSOCIATION principle.

    Key insight: Cheaters cluster together - if your friends are cheaters, you probably are too!
    """
    if not HAS_NX or graph_df is None:
        print("Skipping graph features")
        return train, test

    print("\n" + "="*60)
    print("BUILDING GUILT-BY-ASSOCIATION FEATURES")
    print("="*60)

    # Build the social graph
    G = nx.Graph()
    for _, row in graph_df.iterrows():
        G.add_edge(row['user_a'], row['user_b'])

    all_users = set(train['user_hash'].tolist() + test['user_hash'].tolist())
    for user in all_users:
        if user not in G:
            G.add_node(user)

    print(f"\nGraph statistics:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Avg degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")

    # Get labeled cheating status
    labeled_train = train[train['is_cheating'].notna()]
    cheater_set = set(labeled_train[labeled_train['is_cheating'] == 1]['user_hash'])
    clean_set = set(labeled_train[labeled_train['is_cheating'] == 0]['user_hash'])
    labeled_dict = {**{u: 1 for u in cheater_set}, **{u: 0 for u in clean_set}}

    print(f"\nLabeled data:")
    print(f"  Cheaters: {len(cheater_set):,}")
    print(f"  Clean: {len(clean_set):,}")

    # Count cheaters with connections to other cheaters
    cheater_connections = 0
    for cheater in cheater_set:
        if cheater in G:
            neighbors = set(G.neighbors(cheater))
            cheater_neighbors = neighbors & cheater_set
            if cheater_neighbors:
                cheater_connections += 1
    print(f"  Cheaters connected to other cheaters: {cheater_connections:,} ({100*cheater_connections/len(cheater_set):.1f}%)")

    # Initialize feature dictionaries
    features = {user: {} for user in all_users}

    # =========================================================================
    # FEATURE SET 1: DIRECT NEIGHBOR CHEATING RATES (1-hop)
    # =========================================================================
    print("\n1. Computing 1-hop neighbor features...")

    for user in all_users:
        neighbors = list(G.neighbors(user)) if user in G else []
        n_neighbors = len(neighbors)

        features[user]['graph_degree'] = n_neighbors

        if n_neighbors == 0:
            features[user]['graph_neighbor_cheat_rate_1hop'] = np.nan
            features[user]['graph_neighbor_cheat_count_1hop'] = 0
            features[user]['graph_neighbor_clean_count_1hop'] = 0
            features[user]['graph_neighbor_labeled_ratio_1hop'] = 0
            continue

        labeled_neighbors = [n for n in neighbors if n in labeled_dict]
        n_labeled = len(labeled_neighbors)

        if n_labeled == 0:
            features[user]['graph_neighbor_cheat_rate_1hop'] = np.nan
            features[user]['graph_neighbor_cheat_count_1hop'] = 0
            features[user]['graph_neighbor_clean_count_1hop'] = 0
            features[user]['graph_neighbor_labeled_ratio_1hop'] = 0
        else:
            n_cheaters = sum(1 for n in labeled_neighbors if labeled_dict[n] == 1)
            n_clean = n_labeled - n_cheaters

            features[user]['graph_neighbor_cheat_rate_1hop'] = n_cheaters / n_labeled
            features[user]['graph_neighbor_cheat_count_1hop'] = n_cheaters
            features[user]['graph_neighbor_clean_count_1hop'] = n_clean
            features[user]['graph_neighbor_labeled_ratio_1hop'] = n_labeled / n_neighbors

    # =========================================================================
    # FEATURE SET 2: 2-HOP NEIGHBOR CHEATING RATES
    # =========================================================================
    print("2. Computing 2-hop neighbor features...")

    for user in all_users:
        neighbors_1hop = set(G.neighbors(user)) if user in G else set()
        neighbors_2hop = set()

        for n in neighbors_1hop:
            neighbors_2hop.update(G.neighbors(n))

        # Remove 1-hop neighbors and self
        neighbors_2hop -= neighbors_1hop
        neighbors_2hop.discard(user)

        n_2hop = len(neighbors_2hop)
        features[user]['graph_degree_2hop'] = n_2hop

        if n_2hop == 0:
            features[user]['graph_neighbor_cheat_rate_2hop'] = np.nan
            features[user]['graph_neighbor_cheat_count_2hop'] = 0
            continue

        labeled_2hop = [n for n in neighbors_2hop if n in labeled_dict]

        if len(labeled_2hop) == 0:
            features[user]['graph_neighbor_cheat_rate_2hop'] = np.nan
            features[user]['graph_neighbor_cheat_count_2hop'] = 0
        else:
            n_cheaters = sum(1 for n in labeled_2hop if labeled_dict[n] == 1)
            features[user]['graph_neighbor_cheat_rate_2hop'] = n_cheaters / len(labeled_2hop)
            features[user]['graph_neighbor_cheat_count_2hop'] = n_cheaters

    # =========================================================================
    # FEATURE SET 3: 3-HOP NEIGHBOR CHEATING RATES
    # =========================================================================
    print("3. Computing 3-hop neighbor features...")

    for user in all_users:
        neighbors_1hop = set(G.neighbors(user)) if user in G else set()
        neighbors_2hop = set()
        for n in neighbors_1hop:
            neighbors_2hop.update(G.neighbors(n))
        neighbors_2hop -= neighbors_1hop
        neighbors_2hop.discard(user)

        neighbors_3hop = set()
        for n in neighbors_2hop:
            neighbors_3hop.update(G.neighbors(n))
        neighbors_3hop -= neighbors_2hop
        neighbors_3hop -= neighbors_1hop
        neighbors_3hop.discard(user)

        n_3hop = len(neighbors_3hop)
        features[user]['graph_degree_3hop'] = n_3hop

        if n_3hop == 0:
            features[user]['graph_neighbor_cheat_rate_3hop'] = np.nan
            continue

        labeled_3hop = [n for n in neighbors_3hop if n in labeled_dict]

        if len(labeled_3hop) == 0:
            features[user]['graph_neighbor_cheat_rate_3hop'] = np.nan
        else:
            n_cheaters = sum(1 for n in labeled_3hop if labeled_dict[n] == 1)
            features[user]['graph_neighbor_cheat_rate_3hop'] = n_cheaters / len(labeled_3hop)

    # =========================================================================
    # FEATURE SET 4: COMMUNITY DETECTION (FIND CHEATING RINGS)
    # =========================================================================
    print("4. Detecting communities (cheating rings)...")

    if HAS_LOUVAIN:
        try:
            partition = community_louvain.best_partition(G, random_state=RANDOM_STATE)

            # Calculate community statistics
            comm_members = defaultdict(list)
            for user, comm in partition.items():
                comm_members[comm].append(user)

            comm_stats = {}
            for comm, members in comm_members.items():
                size = len(members)
                labeled_members = [m for m in members if m in labeled_dict]

                if len(labeled_members) > 0:
                    n_cheaters = sum(1 for m in labeled_members if labeled_dict[m] == 1)
                    cheat_rate = n_cheaters / len(labeled_members)
                else:
                    n_cheaters = 0
                    cheat_rate = np.nan

                comm_stats[comm] = {
                    'size': size,
                    'n_cheaters': n_cheaters,
                    'n_labeled': len(labeled_members),
                    'cheat_rate': cheat_rate
                }

            # Assign community features
            for user in all_users:
                comm = partition.get(user, -1)
                if comm == -1 or comm not in comm_stats:
                    features[user]['graph_community_size'] = 1
                    features[user]['graph_community_cheat_rate'] = np.nan
                    features[user]['graph_community_cheat_count'] = 0
                    features[user]['graph_community_is_cheating_ring'] = 0
                else:
                    stats = comm_stats[comm]
                    features[user]['graph_community_size'] = stats['size']
                    features[user]['graph_community_cheat_rate'] = stats['cheat_rate']
                    features[user]['graph_community_cheat_count'] = stats['n_cheaters']
                    # Flag communities with >50% cheating rate as "cheating rings"
                    features[user]['graph_community_is_cheating_ring'] = int(
                        stats['cheat_rate'] > 0.5 if not np.isnan(stats['cheat_rate']) else 0
                    )

            # Count cheating rings
            n_cheating_rings = sum(1 for s in comm_stats.values()
                                   if not np.isnan(s['cheat_rate']) and s['cheat_rate'] > 0.5)
            print(f"   Found {len(comm_stats)} communities, {n_cheating_rings} potential cheating rings (>50% rate)")

        except Exception as e:
            print(f"   Community detection failed: {e}")
            for user in all_users:
                features[user]['graph_community_size'] = 1
                features[user]['graph_community_cheat_rate'] = np.nan
                features[user]['graph_community_cheat_count'] = 0
                features[user]['graph_community_is_cheating_ring'] = 0

    # =========================================================================
    # FEATURE SET 5: LABEL PROPAGATION (SPREAD CHEATING PROBABILITY)
    # =========================================================================
    print("5. Running label propagation...")

    # Initialize probabilities
    cheat_prob = {}
    base_rate = len(cheater_set) / (len(cheater_set) + len(clean_set))

    for user in all_users:
        if user in cheater_set:
            cheat_prob[user] = 1.0
        elif user in clean_set:
            cheat_prob[user] = 0.0
        else:
            cheat_prob[user] = base_rate  # Prior

    # Propagate for several iterations
    n_iterations = 5
    alpha = 0.5  # Weight for neighbor influence

    for iteration in range(n_iterations):
        new_probs = {}
        for user in all_users:
            if user in labeled_dict:
                # Keep labeled users fixed
                new_probs[user] = cheat_prob[user]
                continue

            neighbors = list(G.neighbors(user)) if user in G else []
            if len(neighbors) == 0:
                new_probs[user] = cheat_prob[user]
                continue

            # Average of neighbors
            neighbor_avg = np.mean([cheat_prob[n] for n in neighbors])

            # Blend with current probability
            new_probs[user] = (1 - alpha) * cheat_prob[user] + alpha * neighbor_avg

        cheat_prob = new_probs

    for user in all_users:
        features[user]['graph_propagated_cheat_prob'] = cheat_prob[user]

    # =========================================================================
    # FEATURE SET 6: CHEATER SUBGRAPH CENTRALITY
    # =========================================================================
    print("6. Computing cheater subgraph features...")

    # Build subgraph of known cheaters
    cheater_subgraph = G.subgraph(cheater_set).copy()

    # For each user, count direct connections to cheaters
    for user in all_users:
        if user not in G:
            features[user]['graph_direct_cheater_connections'] = 0
            features[user]['graph_cheater_connection_ratio'] = 0
            continue

        neighbors = set(G.neighbors(user))
        cheater_neighbors = neighbors & cheater_set

        features[user]['graph_direct_cheater_connections'] = len(cheater_neighbors)
        features[user]['graph_cheater_connection_ratio'] = (
            len(cheater_neighbors) / len(neighbors) if len(neighbors) > 0 else 0
        )

    # =========================================================================
    # FEATURE SET 7: NETWORK CENTRALITY MEASURES
    # =========================================================================
    print("7. Computing network centrality...")

    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    for user in all_users:
        features[user]['graph_degree_centrality'] = degree_cent.get(user, 0)

    # Clustering coefficient
    clustering = nx.clustering(G)
    for user in all_users:
        features[user]['graph_clustering'] = clustering.get(user, 0)

    # =========================================================================
    # FEATURE SET 8: WEIGHTED GUILT SCORE
    # =========================================================================
    print("8. Computing weighted guilt scores...")

    for user in all_users:
        # Weighted guilt: combine multiple signals
        guilt_score = 0.0
        n_signals = 0

        # 1-hop neighbor rate (highest weight)
        rate_1hop = features[user].get('graph_neighbor_cheat_rate_1hop', np.nan)
        if not np.isnan(rate_1hop):
            guilt_score += 3.0 * rate_1hop
            n_signals += 3

        # 2-hop neighbor rate
        rate_2hop = features[user].get('graph_neighbor_cheat_rate_2hop', np.nan)
        if not np.isnan(rate_2hop):
            guilt_score += 2.0 * rate_2hop
            n_signals += 2

        # 3-hop neighbor rate
        rate_3hop = features[user].get('graph_neighbor_cheat_rate_3hop', np.nan)
        if not np.isnan(rate_3hop):
            guilt_score += 1.0 * rate_3hop
            n_signals += 1

        # Community rate
        comm_rate = features[user].get('graph_community_cheat_rate', np.nan)
        if not np.isnan(comm_rate):
            guilt_score += 2.0 * comm_rate
            n_signals += 2

        # Propagated probability
        prop_prob = features[user].get('graph_propagated_cheat_prob', base_rate)
        guilt_score += 1.0 * prop_prob
        n_signals += 1

        features[user]['graph_weighted_guilt_score'] = guilt_score / n_signals if n_signals > 0 else base_rate

    # =========================================================================
    # FEATURE SET 9: RISK TIERS
    # =========================================================================
    print("9. Assigning risk tiers...")

    for user in all_users:
        # High risk if any of these:
        # - Direct connection to cheater
        # - In a cheating ring
        # - High weighted guilt score

        direct_cheater = features[user].get('graph_direct_cheater_connections', 0) > 0
        in_ring = features[user].get('graph_community_is_cheating_ring', 0) == 1
        high_guilt = features[user].get('graph_weighted_guilt_score', 0) > 0.5
        high_1hop = features[user].get('graph_neighbor_cheat_rate_1hop', 0) > 0.5

        risk_score = sum([direct_cheater, in_ring, high_guilt, high_1hop])

        features[user]['graph_risk_tier'] = risk_score
        features[user]['graph_is_high_risk'] = int(risk_score >= 2)

    # =========================================================================
    # ADD FEATURES TO DATAFRAMES
    # =========================================================================
    print("\nAdding graph features to dataframes...")

    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        feature_names = list(next(iter(features.values())).keys())

        for feat_name in feature_names:
            df[feat_name] = df['user_hash'].map(lambda u: features.get(u, {}).get(feat_name, np.nan))

        return df

    train = add_features(train)
    test = add_features(test)

    graph_features = [c for c in train.columns if c.startswith('graph_')]
    print(f"  Added {len(graph_features)} graph features")

    return train, test


# =============================================================================
# BASIC FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-graph features."""
    df = df.copy()

    # Missingness
    df['missing_count'] = df[FEATURE_COLS].isna().sum(axis=1)
    df['missing_ratio'] = df['missing_count'] / len(FEATURE_COLS)

    for col in FEATURE_COLS:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    # Aggregations
    numeric = df[FEATURE_COLS]
    df['feature_mean'] = numeric.mean(axis=1)
    df['feature_std'] = numeric.std(axis=1)
    df['feature_max'] = numeric.max(axis=1)
    df['feature_min'] = numeric.min(axis=1)
    df['feature_sum'] = numeric.sum(axis=1)

    # Log transforms for high-value features
    for col in ['feature_010', 'feature_015']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].fillna(0).clip(lower=0))

    # Interactions
    for f1, f2 in [('feature_008', 'feature_009'), ('feature_013', 'feature_014')]:
        if f1 in df.columns and f2 in df.columns:
            df[f'{f1}_x_{f2}'] = df[f1].fillna(0) * df[f2].fillna(0)

    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all feature columns."""
    exclude = {'user_hash', 'is_cheating', 'high_conf_clean', 'is_pseudo_label'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]


def train_lgb(X_tr, y_tr, X_val, y_val, feature_names):
    """Train LightGBM with settings optimized for catching cheaters."""

    # Heavier weight on cheaters (class 1) since missing them is expensive
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1) * 1.5

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'seed': RANDOM_STATE,
        'scale_pos_weight': scale_pos,
    }

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    model = lgb.train(
        params, train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    return model


def train_xgb(X_tr, y_tr, X_val, y_val):
    """Train XGBoost."""
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1) * 1.5

    model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=30,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE,
        eval_metric='auc',
        early_stopping_rounds=100,
        verbosity=0,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_ensemble(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   use_pseudo: bool = True):
    """Train ensemble with cross-validation."""
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE MODEL")
    print("="*60)

    # Add pseudo-labels
    if use_pseudo:
        mask = (train_df['high_conf_clean'] == 1.0) & (train_df['is_cheating'].isna())
        n_pseudo = mask.sum()
        if n_pseudo > 0:
            train_df = train_df.copy()
            train_df.loc[mask, 'is_cheating'] = 0.0
            print(f"\nAdded {n_pseudo:,} pseudo-labels from high-confidence clean")

    # Get labeled data
    labeled = train_df[train_df['is_cheating'].notna()].copy()

    feature_cols = get_feature_columns(labeled)
    print(f"\nUsing {len(feature_cols)} features")

    # Print graph features
    graph_feats = [c for c in feature_cols if c.startswith('graph_')]
    print(f"  Graph features: {len(graph_feats)}")

    X = labeled[feature_cols].values
    y = labeled['is_cheating'].values
    X_test = test_df[feature_cols].values

    # Handle NaN
    X = np.nan_to_num(X, nan=-999)
    X_test = np.nan_to_num(X_test, nan=-999)

    # Initialize
    oof_lgb = np.zeros(len(labeled))
    oof_xgb = np.zeros(len(labeled))
    test_lgb = np.zeros(len(test_df))
    test_xgb = np.zeros(len(test_df))

    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\nFold {fold + 1}/{N_FOLDS}")

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if HAS_LGB:
            lgb_model = train_lgb(X_tr, y_tr, X_val, y_val, feature_cols)
            oof_lgb[val_idx] = lgb_model.predict(X_val)
            test_lgb += lgb_model.predict(X_test) / N_FOLDS
            lgb_auc = roc_auc_score(y_val, oof_lgb[val_idx])
            print(f"  LightGBM AUC: {lgb_auc:.4f}")

        if HAS_XGB:
            xgb_model = train_xgb(X_tr, y_tr, X_val, y_val)
            oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
            test_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
            xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])
            print(f"  XGBoost AUC: {xgb_auc:.4f}")

        if HAS_LGB and HAS_XGB:
            combined = 0.5 * oof_lgb[val_idx] + 0.5 * oof_xgb[val_idx]
        elif HAS_LGB:
            combined = oof_lgb[val_idx]
        else:
            combined = oof_xgb[val_idx]

        fold_auc = roc_auc_score(y_val, combined)
        fold_aucs.append(fold_auc)
        print(f"  Ensemble AUC: {fold_auc:.4f}")

    # Final predictions
    if HAS_LGB and HAS_XGB:
        oof_preds = 0.5 * oof_lgb + 0.5 * oof_xgb
        test_preds = 0.5 * test_lgb + 0.5 * test_xgb
    elif HAS_LGB:
        oof_preds = oof_lgb
        test_preds = test_lgb
    else:
        oof_preds = oof_xgb
        test_preds = test_xgb

    # Print feature importance
    if HAS_LGB:
        imp = lgb_model.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
        print("\nTop 15 Feature Importances:")
        for i, (name, val) in enumerate(feat_imp[:15]):
            marker = "  **" if name.startswith('graph_') else ""
            print(f"  {i+1:2d}. {name}: {val:.0f}{marker}")

    # Metrics
    overall_auc = roc_auc_score(y, oof_preds)
    ap = average_precision_score(y, oof_preds)

    print(f"\nCV AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")

    # Create full oof array
    full_oof = np.full(len(train_df), np.nan)
    full_oof[train_df['is_cheating'].notna().values] = oof_preds

    metrics = {
        'cv_auc_mean': np.mean(fold_aucs),
        'cv_auc_std': np.std(fold_aucs),
        'overall_auc': overall_auc,
        'ap': ap
    }

    return full_oof, test_preds, metrics


# =============================================================================
# COST OPTIMIZATION
# =============================================================================

def optimize_thresholds(y_true, y_pred):
    """Find optimal thresholds to minimize cost."""
    print("\n" + "="*60)
    print("OPTIMIZING DECISION THRESHOLDS")
    print("="*60)
    print(f"\nCost structure:")
    print(f"  False Negative (miss cheater): ${COST_FALSE_NEGATIVE:.0f}")
    print(f"  False Positive (block innocent): ${COST_FALSE_POSITIVE_BLOCK:.0f}")
    print(f"  Manual Review: ${COST_MANUAL_REVIEW:.0f}")

    best_cost = float('inf')
    best_t_low = 0.3
    best_t_high = 0.7

    thresholds = np.linspace(0.01, 0.99, 50)

    for t_low in thresholds:
        for t_high in thresholds:
            if t_high <= t_low:
                continue

            cost = 0
            for y, p in zip(y_true, y_pred):
                if p <= t_low:
                    if y == 1:  # Missed cheater
                        cost += COST_FALSE_NEGATIVE
                elif p >= t_high:
                    if y == 0:  # Wrongly blocked
                        cost += COST_FALSE_POSITIVE_BLOCK
                else:
                    cost += COST_MANUAL_REVIEW

            if cost < best_cost:
                best_cost = cost
                best_t_low = t_low
                best_t_high = t_high

    print(f"\nOptimal thresholds: t_low={best_t_low:.3f}, t_high={best_t_high:.3f}")
    print(f"Estimated cost: ${best_cost:,.0f}")

    # Decision breakdown
    auto_pass = (y_pred <= best_t_low).sum()
    auto_block = (y_pred >= best_t_high).sum()
    manual = len(y_pred) - auto_pass - auto_block

    print(f"\nDecision breakdown:")
    print(f"  Auto-pass: {auto_pass:,} ({100*auto_pass/len(y_pred):.1f}%)")
    print(f"  Manual review: {manual:,} ({100*manual/len(y_pred):.1f}%)")
    print(f"  Auto-block: {auto_block:,} ({100*auto_block/len(y_pred):.1f}%)")

    return best_t_low, best_t_high, best_cost


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="sample-data/train.csv")
    parser.add_argument("--test", default="sample-data/test.csv")
    parser.add_argument("--graph", default="sample-data/social_graph.csv")
    parser.add_argument("--output", default="submission_v2.csv")
    args = parser.parse_args()

    # Auto-detect paths
    for base in ["", "sample-data/", "data/"]:
        if Path(f"{base}train.csv").exists():
            if args.train == "sample-data/train.csv":
                args.train = f"{base}train.csv"
            if args.test == "sample-data/test.csv":
                args.test = f"{base}test.csv"
            if args.graph == "sample-data/social_graph.csv":
                args.graph = f"{base}social_graph.csv"
            break

    print("="*60)
    print("CHEATING DETECTOR V2 - GUILT BY ASSOCIATION")
    print("="*60)

    if not HAS_LGB and not HAS_XGB:
        print("Error: Install lightgbm or xgboost")
        return 1

    # Load data
    train, test, graph = load_data(args.train, args.test, args.graph)

    # Class distribution
    labeled = train[train['is_cheating'].notna()]
    n_cheat = (labeled['is_cheating'] == 1).sum()
    n_clean = (labeled['is_cheating'] == 0).sum()
    print(f"\nLabeled: {n_cheat:,} cheaters, {n_clean:,} clean ({100*n_cheat/(n_cheat+n_clean):.1f}% cheating rate)")

    # Build graph features (THE KEY!)
    if graph is not None:
        train, test = build_guilt_by_association_features(train, test, graph)

    # Basic features
    print("\nEngineering basic features...")
    train = engineer_features(train)
    test = engineer_features(test)

    # Train
    oof_preds, test_preds, metrics = train_ensemble(train, test)

    # Optimize thresholds
    mask = train['is_cheating'].notna()
    y_true = train.loc[mask, 'is_cheating'].values
    y_oof = oof_preds[mask.values]
    t_low, t_high, _ = optimize_thresholds(y_true, y_oof)

    # Save submission
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': test_preds
    })
    submission.to_csv(args.output, index=False)

    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Submission saved: {args.output}")
    print(f"CV AUC: {metrics['cv_auc_mean']:.4f}")
    print(f"Predictions: min={test_preds.min():.4f}, max={test_preds.max():.4f}, mean={test_preds.mean():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
