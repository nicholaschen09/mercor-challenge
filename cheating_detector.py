#!/usr/bin/env python3
"""
Cheating Detection Model for Online Interview Marketplace
==========================================================

A comprehensive solution featuring:
- Graph-based features from social network
- Semi-supervised learning with high-confidence clean examples
- Advanced feature engineering
- LightGBM + XGBoost ensemble
- Cost-aware threshold optimization

Usage:
    python cheating_detector.py
    python cheating_detector.py --train data/train.csv --test data/test.csv
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from collections import defaultdict

warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("Warning: NetworkX not installed. Run: pip install networkx")

try:
    from community import community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURE_COLS = [f"feature_{i:03d}" for i in range(1, 19)]
RANDOM_STATE = 42
N_FOLDS = 5

# Cost parameters for threshold optimization
COST_FALSE_NEGATIVE = 100.0  # Missing a cheater
COST_FALSE_POSITIVE_AUTO_BLOCK = 50.0  # Wrongly auto-blocking
COST_MANUAL_REVIEW = 10.0  # Manual review cost


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(train_path: str, test_path: str, graph_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load train, test, and social graph data."""
    print("Loading data...")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if Path(graph_path).exists():
        graph = pd.read_csv(graph_path)
        print(f"  Train: {len(train):,} rows")
        print(f"  Test: {len(test):,} rows")
        print(f"  Graph edges: {len(graph):,}")
    else:
        graph = None
        print(f"  Train: {len(train):,} rows")
        print(f"  Test: {len(test):,} rows")
        print("  No graph data found")

    return train, test, graph


# =============================================================================
# GRAPH FEATURE ENGINEERING
# =============================================================================

def build_graph_features(train: pd.DataFrame, test: pd.DataFrame, graph_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build features from the social graph:
    - Neighbor cheating rates (propagate labels through network)
    - Network centrality measures
    - Community detection
    """
    if not HAS_NX or graph_df is None:
        print("Skipping graph features (NetworkX not available or no graph data)")
        return train, test

    print("Building graph features...")

    # Build graph
    G = nx.Graph()
    for _, row in graph_df.iterrows():
        G.add_edge(row['user_a'], row['user_b'])

    all_users = set(train['user_hash'].tolist() + test['user_hash'].tolist())
    for user in all_users:
        if user not in G:
            G.add_node(user)

    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Get labeled cheating status
    labeled = train[train['is_cheating'].notna()].set_index('user_hash')['is_cheating'].to_dict()

    # --- Feature 1: Neighbor cheating rate (1-hop) ---
    neighbor_cheat_rate = {}
    neighbor_cheat_count = {}
    neighbor_count = {}

    for user in all_users:
        neighbors = list(G.neighbors(user)) if user in G else []
        neighbor_count[user] = len(neighbors)

        if len(neighbors) == 0:
            neighbor_cheat_rate[user] = np.nan
            neighbor_cheat_count[user] = 0
            continue

        labeled_neighbors = [n for n in neighbors if n in labeled]
        if len(labeled_neighbors) == 0:
            neighbor_cheat_rate[user] = np.nan
            neighbor_cheat_count[user] = 0
        else:
            cheaters = sum(1 for n in labeled_neighbors if labeled[n] == 1)
            neighbor_cheat_rate[user] = cheaters / len(labeled_neighbors)
            neighbor_cheat_count[user] = cheaters

    # --- Feature 2: 2-hop neighbor cheating rate ---
    neighbor_cheat_rate_2hop = {}
    for user in all_users:
        neighbors_1hop = set(G.neighbors(user)) if user in G else set()
        neighbors_2hop = set()
        for n in neighbors_1hop:
            neighbors_2hop.update(G.neighbors(n))
        neighbors_2hop -= neighbors_1hop
        neighbors_2hop.discard(user)

        if len(neighbors_2hop) == 0:
            neighbor_cheat_rate_2hop[user] = np.nan
            continue

        labeled_2hop = [n for n in neighbors_2hop if n in labeled]
        if len(labeled_2hop) == 0:
            neighbor_cheat_rate_2hop[user] = np.nan
        else:
            cheaters = sum(1 for n in labeled_2hop if labeled[n] == 1)
            neighbor_cheat_rate_2hop[user] = cheaters / len(labeled_2hop)

    # --- Feature 3: Degree centrality ---
    degree_centrality = nx.degree_centrality(G)

    # --- Feature 4: Community detection (if available) ---
    community_id = {}
    community_size = {}
    community_cheat_rate = {}

    if HAS_LOUVAIN:
        try:
            partition = community_louvain.best_partition(G, random_state=RANDOM_STATE)
            community_id = partition

            # Calculate community sizes and cheating rates
            comm_members = defaultdict(list)
            for user, comm in partition.items():
                comm_members[comm].append(user)

            for comm, members in comm_members.items():
                size = len(members)
                labeled_members = [m for m in members if m in labeled]
                if len(labeled_members) > 0:
                    cheat_rate = sum(1 for m in labeled_members if labeled[m] == 1) / len(labeled_members)
                else:
                    cheat_rate = np.nan

                for m in members:
                    community_size[m] = size
                    community_cheat_rate[m] = cheat_rate
        except Exception as e:
            print(f"  Community detection failed: {e}")

    # --- Feature 5: Local clustering coefficient ---
    clustering = nx.clustering(G)

    # --- Add features to dataframes ---
    def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['graph_neighbor_count'] = df['user_hash'].map(neighbor_count).fillna(0)
        df['graph_neighbor_cheat_rate'] = df['user_hash'].map(neighbor_cheat_rate)
        df['graph_neighbor_cheat_count'] = df['user_hash'].map(neighbor_cheat_count).fillna(0)
        df['graph_neighbor_cheat_rate_2hop'] = df['user_hash'].map(neighbor_cheat_rate_2hop)
        df['graph_degree_centrality'] = df['user_hash'].map(degree_centrality).fillna(0)
        df['graph_clustering'] = df['user_hash'].map(clustering).fillna(0)

        if community_id:
            df['graph_community_id'] = df['user_hash'].map(community_id).fillna(-1)
            df['graph_community_size'] = df['user_hash'].map(community_size).fillna(1)
            df['graph_community_cheat_rate'] = df['user_hash'].map(community_cheat_rate)

        return df

    train = add_graph_features(train)
    test = add_graph_features(test)

    print("  Graph features added successfully")
    return train, test


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from base features:
    - Missingness patterns
    - Feature ratios and interactions
    - Statistical aggregations
    """
    df = df.copy()

    # --- Missingness features ---
    df['missing_count'] = df[FEATURE_COLS].isna().sum(axis=1)
    df['missing_ratio'] = df['missing_count'] / len(FEATURE_COLS)

    # Create missingness indicator for each feature
    for col in FEATURE_COLS:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    # --- Statistical aggregations across features ---
    numeric_features = df[FEATURE_COLS].copy()
    df['feature_mean'] = numeric_features.mean(axis=1)
    df['feature_std'] = numeric_features.std(axis=1)
    df['feature_max'] = numeric_features.max(axis=1)
    df['feature_min'] = numeric_features.min(axis=1)
    df['feature_range'] = df['feature_max'] - df['feature_min']
    df['feature_sum'] = numeric_features.sum(axis=1)

    # --- Key ratios (based on feature semantics) ---
    # These are common patterns in cheating detection
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ratio features - protect against division by zero
        if 'feature_015' in df.columns and 'feature_017' in df.columns:
            df['ratio_015_017'] = df['feature_015'] / (df['feature_017'] + 1e-8)

        if 'feature_016' in df.columns and 'feature_017' in df.columns:
            df['ratio_016_017'] = df['feature_016'] / (df['feature_017'] + 1e-8)

        if 'feature_008' in df.columns and 'feature_010' in df.columns:
            df['ratio_008_010'] = df['feature_008'] / (df['feature_010'] + 1e-8)

        # Feature 010 appears to be a high-value feature (potentially time-based)
        if 'feature_010' in df.columns:
            df['feature_010_log'] = np.log1p(df['feature_010'].fillna(0).clip(lower=0))
            df['feature_010_sqrt'] = np.sqrt(df['feature_010'].fillna(0).clip(lower=0))
            df['feature_010_is_high'] = (df['feature_010'] > 1000).astype(int)

        # Feature 015 appears time-related
        if 'feature_015' in df.columns:
            df['feature_015_log'] = np.log1p(df['feature_015'].fillna(0).clip(lower=0))
            df['feature_015_is_high'] = (df['feature_015'] > 10).astype(int)

    # --- Interaction features ---
    # Products of potentially correlated features
    interaction_pairs = [
        ('feature_001', 'feature_002'),
        ('feature_003', 'feature_004'),
        ('feature_008', 'feature_009'),
        ('feature_013', 'feature_014'),
    ]

    for f1, f2 in interaction_pairs:
        if f1 in df.columns and f2 in df.columns:
            df[f'{f1}_x_{f2}'] = df[f1].fillna(0) * df[f2].fillna(0)

    # --- Binary pattern features ---
    # Count of features with value 0
    df['zero_count'] = (numeric_features == 0).sum(axis=1)

    # Count of features with value 1
    df['one_count'] = (numeric_features == 1).sum(axis=1)

    return df


# =============================================================================
# SEMI-SUPERVISED LEARNING
# =============================================================================

def add_pseudo_labels(train: pd.DataFrame, confidence_threshold: float = 0.95) -> pd.DataFrame:
    """
    Use high-confidence clean examples as pseudo-labels.
    The high_conf_clean flag indicates candidates that are very likely NOT cheating.
    """
    train = train.copy()

    # Rows with high_conf_clean=1 and no label are pseudo-labeled as non-cheaters
    high_conf_mask = (train['high_conf_clean'] == 1.0) & (train['is_cheating'].isna())
    n_pseudo = high_conf_mask.sum()

    if n_pseudo > 0:
        train.loc[high_conf_mask, 'is_cheating'] = 0.0
        train.loc[high_conf_mask, 'is_pseudo_label'] = 1
        print(f"  Added {n_pseudo:,} pseudo-labels from high-confidence clean examples")

    return train


# =============================================================================
# MODEL TRAINING
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all feature columns for modeling."""
    exclude_cols = {'user_hash', 'is_cheating', 'high_conf_clean', 'is_pseudo_label'}
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    return feature_cols


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: List[str]) -> lgb.Booster:
    """Train a LightGBM model with hyperparameters tuned for fraud detection."""

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'seed': RANDOM_STATE,
        'is_unbalance': True,  # Handle class imbalance
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
    """Train an XGBoost model."""

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='auc',
        early_stopping_rounds=50,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def train_ensemble(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   use_pseudo_labels: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Train an ensemble model using cross-validation.
    Returns: (train_oof_preds, test_preds, metrics)
    """
    print("\nTraining ensemble model...")

    # Prepare data
    if use_pseudo_labels:
        train_df = add_pseudo_labels(train_df.copy())

    # Filter to labeled examples only
    labeled_mask = train_df['is_cheating'].notna()
    train_labeled = train_df[labeled_mask].copy()

    feature_cols = get_feature_columns(train_labeled)
    print(f"  Using {len(feature_cols)} features")

    X = train_labeled[feature_cols].values
    y = train_labeled['is_cheating'].values
    X_test = test_df[feature_cols].values

    # Handle missing values
    X = np.nan_to_num(X, nan=-999)
    X_test = np.nan_to_num(X_test, nan=-999)

    # Initialize arrays for predictions
    oof_preds_lgb = np.zeros(len(train_labeled))
    oof_preds_xgb = np.zeros(len(train_labeled))
    test_preds_lgb = np.zeros(len(test_df))
    test_preds_xgb = np.zeros(len(test_df))

    # Cross-validation
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS}")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Train LightGBM
        if HAS_LGB:
            lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val, feature_cols)
            oof_preds_lgb[val_idx] = lgb_model.predict(X_val)
            test_preds_lgb += lgb_model.predict(X_test) / N_FOLDS

            lgb_auc = roc_auc_score(y_val, oof_preds_lgb[val_idx])
            print(f"    LightGBM AUC: {lgb_auc:.4f}")

        # Train XGBoost
        if HAS_XGB:
            xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val)
            oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
            test_preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_FOLDS

            xgb_auc = roc_auc_score(y_val, oof_preds_xgb[val_idx])
            print(f"    XGBoost AUC: {xgb_auc:.4f}")

        # Combined fold AUC
        if HAS_LGB and HAS_XGB:
            combined_val = 0.5 * oof_preds_lgb[val_idx] + 0.5 * oof_preds_xgb[val_idx]
        elif HAS_LGB:
            combined_val = oof_preds_lgb[val_idx]
        else:
            combined_val = oof_preds_xgb[val_idx]

        fold_auc = roc_auc_score(y_val, combined_val)
        fold_aucs.append(fold_auc)
        print(f"    Ensemble AUC: {fold_auc:.4f}")

    # Final ensemble
    if HAS_LGB and HAS_XGB:
        oof_preds = 0.5 * oof_preds_lgb + 0.5 * oof_preds_xgb
        test_preds = 0.5 * test_preds_lgb + 0.5 * test_preds_xgb
    elif HAS_LGB:
        oof_preds = oof_preds_lgb
        test_preds = test_preds_lgb
    else:
        oof_preds = oof_preds_xgb
        test_preds = test_preds_xgb

    # Calculate overall metrics
    overall_auc = roc_auc_score(y, oof_preds)
    ap = average_precision_score(y, oof_preds)

    metrics = {
        'cv_auc_mean': np.mean(fold_aucs),
        'cv_auc_std': np.std(fold_aucs),
        'overall_auc': overall_auc,
        'average_precision': ap,
        'feature_cols': feature_cols,
    }

    print(f"\n  CV AUC: {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']:.4f})")
    print(f"  Overall AUC: {metrics['overall_auc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")

    # Create full oof predictions array with original train index
    full_oof_preds = np.full(len(train_df), np.nan)
    full_oof_preds[labeled_mask.values] = oof_preds

    return full_oof_preds, test_preds, metrics


# =============================================================================
# COST-AWARE THRESHOLD OPTIMIZATION
# =============================================================================

def optimize_thresholds(y_true: np.ndarray, y_pred: np.ndarray,
                        cost_fn: float = COST_FALSE_NEGATIVE,
                        cost_fp: float = COST_FALSE_POSITIVE_AUTO_BLOCK,
                        cost_review: float = COST_MANUAL_REVIEW) -> Tuple[float, float, float]:
    """
    Find optimal thresholds for three-way decision:
    - Below t_low: auto-pass (predict 0)
    - Between t_low and t_high: manual review
    - Above t_high: auto-block (predict 1)

    Returns: (t_low, t_high, min_cost)
    """
    print("\nOptimizing decision thresholds...")

    best_cost = float('inf')
    best_t_low = 0.5
    best_t_high = 0.5

    # Grid search over threshold pairs
    thresholds = np.linspace(0.01, 0.99, 99)

    for t_low in thresholds:
        for t_high in thresholds:
            if t_high <= t_low:
                continue

            # Calculate cost
            cost = 0

            for y, p in zip(y_true, y_pred):
                if p <= t_low:
                    # Auto-pass: cost if actually cheating (FN)
                    if y == 1:
                        cost += cost_fn
                elif p >= t_high:
                    # Auto-block: cost if not cheating (FP)
                    if y == 0:
                        cost += cost_fp
                else:
                    # Manual review: always incurs review cost
                    cost += cost_review

            if cost < best_cost:
                best_cost = cost
                best_t_low = t_low
                best_t_high = t_high

    print(f"  Optimal thresholds: t_low={best_t_low:.3f}, t_high={best_t_high:.3f}")
    print(f"  Minimum cost: {best_cost:.2f}")

    # Print decision breakdown
    auto_pass = (y_pred <= best_t_low).sum()
    auto_block = (y_pred >= best_t_high).sum()
    manual_review = len(y_pred) - auto_pass - auto_block

    print(f"  Decision breakdown: auto-pass={auto_pass}, manual-review={manual_review}, auto-block={auto_block}")

    return best_t_low, best_t_high, best_cost


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def print_feature_importance(model, feature_names: List[str], top_n: int = 20):
    """Print top feature importances."""
    if hasattr(model, 'feature_importance'):
        importances = model.feature_importance(importance_type='gain')
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return

    feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])

    print(f"\nTop {top_n} Feature Importances:")
    for i, (name, imp) in enumerate(feat_imp[:top_n]):
        print(f"  {i+1:2d}. {name}: {imp:.4f}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cheating Detection Model")
    parser.add_argument("--train", default="sample-data/train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="sample-data/test.csv", help="Path to test.csv")
    parser.add_argument("--graph", default="sample-data/social_graph.csv", help="Path to social_graph.csv")
    parser.add_argument("--output", default="submission_advanced.csv", help="Output submission path")
    parser.add_argument("--no-graph", action="store_true", help="Skip graph features")
    parser.add_argument("--no-pseudo", action="store_true", help="Skip pseudo-labeling")
    args = parser.parse_args()

    # Auto-detect data paths
    for base in ["", "sample-data/", "data/"]:
        if Path(f"{base}train.csv").exists():
            if args.train == "sample-data/train.csv":
                args.train = f"{base}train.csv"
            if args.test == "sample-data/test.csv":
                args.test = f"{base}test.csv"
            if args.graph == "sample-data/social_graph.csv":
                args.graph = f"{base}social_graph.csv"
            break

    print("=" * 60)
    print("CHEATING DETECTION MODEL")
    print("=" * 60)

    # Check dependencies
    if not HAS_LGB and not HAS_XGB:
        print("\nError: Neither LightGBM nor XGBoost is installed.")
        print("Please run: pip install lightgbm xgboost")
        return 1

    # Load data
    train, test, graph = load_data(args.train, args.test, args.graph)

    # Print class distribution
    labeled = train[train['is_cheating'].notna()]
    n_cheaters = (labeled['is_cheating'] == 1).sum()
    n_clean = (labeled['is_cheating'] == 0).sum()
    print(f"\nLabeled data: {len(labeled):,} rows ({n_cheaters:,} cheaters, {n_clean:,} clean)")
    print(f"  Cheating rate: {n_cheaters / len(labeled) * 100:.2f}%")

    high_conf_clean = (train['high_conf_clean'] == 1.0).sum()
    print(f"High-confidence clean (unlabeled): {high_conf_clean:,} rows")

    # Feature engineering
    print("\nEngineering features...")
    train = engineer_features(train)
    test = engineer_features(test)

    # Graph features
    if not args.no_graph and graph is not None:
        train, test = build_graph_features(train, test, graph)

    # Train ensemble
    oof_preds, test_preds, metrics = train_ensemble(
        train, test,
        use_pseudo_labels=not args.no_pseudo
    )

    # Optimize thresholds on validation data
    labeled_mask = train['is_cheating'].notna()
    y_true = train.loc[labeled_mask, 'is_cheating'].values
    y_oof = oof_preds[labeled_mask.values]

    t_low, t_high, min_cost = optimize_thresholds(y_true, y_oof)

    # Create submission
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': test_preds
    })

    submission.to_csv(args.output, index=False)
    print(f"\nSubmission saved to: {args.output}")
    print(f"  Prediction range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  Prediction mean: {test_preds.mean():.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"CV AUC: {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']:.4f})")
    print(f"Overall AUC: {metrics['overall_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Optimal thresholds: t_low={t_low:.3f}, t_high={t_high:.3f}")
    print(f"Minimum cost (on train): {min_cost:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
