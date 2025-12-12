#!/usr/bin/env python3
"""
Cheating Detection V4 - MAXIMUM ACCURACY
==========================================
Advanced techniques:
- Multi-hop graph propagation (2-hop, 3-hop)
- Label propagation on graph
- PageRank-style risk scoring
- Connected component analysis
- Advanced feature engineering (interactions, target encoding)
- Multi-model ensemble (LightGBM + XGBoost + CatBoost)
- Stacking with meta-learner
- Iterative pseudo-labeling with confidence thresholds
- Cost-aware threshold optimization
"""

import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# Try importing gradient boosting libraries
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not available")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available")

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False

FEATURE_COLS = [f"feature_{i:03d}" for i in range(1, 19)]
RANDOM_STATE = 42
N_FOLDS = 5

# Cost structure from competition
COST_FN = 600    # False negative (cheater passes)
COST_FP_BLOCK = 300   # False positive in auto-block
COST_FP_REVIEW = 150  # False positive in manual review
COST_TP_REVIEW = 5    # True positive requiring review
COST_CORRECT = 0      # Correct decisions


def load_data(train_path: str, test_path: str, graph_path: str):
    """Load all data files."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    graph = pd.read_csv(graph_path) if Path(graph_path).exists() else None
    
    # Stats
    n_labeled = train['is_cheating'].notna().sum()
    n_cheaters = (train['is_cheating'] == 1).sum()
    n_clean = (train['is_cheating'] == 0).sum()
    n_unlabeled = train['is_cheating'].isna().sum()
    n_high_conf = (train['high_conf_clean'] == 1).sum()
    
    print(f"  Train total: {len(train):,}")
    print(f"  - Labeled: {n_labeled:,} (cheaters: {n_cheaters:,}, clean: {n_clean:,})")
    print(f"  - Unlabeled: {n_unlabeled:,} (high_conf_clean: {n_high_conf:,})")
    print(f"  Test: {len(test):,}")
    print(f"  Graph edges: {len(graph) if graph is not None else 0:,}")
    
    return train, test, graph


def build_adjacency(graph_df: pd.DataFrame) -> Dict[str, set]:
    """Build adjacency list from edge dataframe."""
    adj = defaultdict(set)
    for _, row in graph_df.iterrows():
        adj[row['user_a']].add(row['user_b'])
        adj[row['user_b']].add(row['user_a'])
    return adj


def compute_multi_hop_features(
    all_users: np.ndarray,
    adj: Dict[str, set],
    cheater_set: set,
    clean_set: set,
    base_rate: float
) -> pd.DataFrame:
    """Compute 1-hop, 2-hop, and 3-hop graph features."""
    
    features = []
    
    for user in all_users:
        neighbors_1 = adj.get(user, set())
        degree = len(neighbors_1)
        
        # 1-hop stats
        cheater_1hop = sum(1 for n in neighbors_1 if n in cheater_set)
        clean_1hop = sum(1 for n in neighbors_1 if n in clean_set)
        labeled_1hop = cheater_1hop + clean_1hop
        
        # 2-hop neighbors (excluding user and 1-hop)
        neighbors_2 = set()
        for n1 in neighbors_1:
            neighbors_2.update(adj.get(n1, set()))
        neighbors_2 -= neighbors_1
        neighbors_2.discard(user)
        
        cheater_2hop = sum(1 for n in neighbors_2 if n in cheater_set)
        clean_2hop = sum(1 for n in neighbors_2 if n in clean_set)
        labeled_2hop = cheater_2hop + clean_2hop
        
        # 3-hop neighbors
        neighbors_3 = set()
        for n2 in neighbors_2:
            neighbors_3.update(adj.get(n2, set()))
        neighbors_3 -= neighbors_2
        neighbors_3 -= neighbors_1
        neighbors_3.discard(user)
        
        cheater_3hop = sum(1 for n in neighbors_3 if n in cheater_set)
        clean_3hop = sum(1 for n in neighbors_3 if n in clean_set)
        labeled_3hop = cheater_3hop + clean_3hop
        
        # Rates
        rate_1hop = cheater_1hop / labeled_1hop if labeled_1hop > 0 else np.nan
        rate_2hop = cheater_2hop / labeled_2hop if labeled_2hop > 0 else np.nan
        rate_3hop = cheater_3hop / labeled_3hop if labeled_3hop > 0 else np.nan
        
        # Weighted combined rate (closer hops matter more)
        weights = [0.6, 0.3, 0.1]
        rates = [rate_1hop, rate_2hop, rate_3hop]
        valid_weights = [w for w, r in zip(weights, rates) if not np.isnan(r)]
        valid_rates = [r for r in rates if not np.isnan(r)]
        combined_rate = np.average(valid_rates, weights=valid_weights) if valid_rates else np.nan
        
        features.append({
            'user_hash': user,
            'graph_degree': degree,
            'graph_log_degree': np.log1p(degree),
            
            # 1-hop features
            'graph_cheater_1hop': cheater_1hop,
            'graph_clean_1hop': clean_1hop,
            'graph_labeled_1hop': labeled_1hop,
            'graph_rate_1hop': rate_1hop,
            'graph_has_cheater_1hop': int(cheater_1hop > 0),
            
            # 2-hop features
            'graph_neighbors_2hop': len(neighbors_2),
            'graph_cheater_2hop': cheater_2hop,
            'graph_clean_2hop': clean_2hop,
            'graph_labeled_2hop': labeled_2hop,
            'graph_rate_2hop': rate_2hop,
            'graph_has_cheater_2hop': int(cheater_2hop > 0),
            
            # 3-hop features
            'graph_neighbors_3hop': len(neighbors_3),
            'graph_cheater_3hop': cheater_3hop,
            'graph_rate_3hop': rate_3hop,
            
            # Combined
            'graph_combined_rate': combined_rate,
            'graph_total_cheater_reach': cheater_1hop + cheater_2hop + cheater_3hop,
        })
    
    return pd.DataFrame(features)


def label_propagation(
    all_users: np.ndarray,
    adj: Dict[str, set],
    known_labels: Dict[str, float],
    n_iterations: int = 10,
    alpha: float = 0.5
) -> Dict[str, float]:
    """
    Simple label propagation algorithm.
    Returns smoothed probabilities for all users.
    """
    # Initialize scores
    scores = {}
    for user in all_users:
        if user in known_labels:
            scores[user] = known_labels[user]
        else:
            scores[user] = 0.5  # neutral
    
    for _ in range(n_iterations):
        new_scores = {}
        for user in all_users:
            neighbors = adj.get(user, set())
            if not neighbors:
                new_scores[user] = scores[user]
                continue
            
            # Average of neighbors
            neighbor_avg = np.mean([scores.get(n, 0.5) for n in neighbors])
            
            if user in known_labels:
                # Labeled nodes: weighted average with original label
                new_scores[user] = alpha * known_labels[user] + (1 - alpha) * neighbor_avg
            else:
                # Unlabeled nodes: take neighbor average
                new_scores[user] = neighbor_avg
        
        scores = new_scores
    
    return scores


def pagerank_risk(
    all_users: np.ndarray,
    adj: Dict[str, set],
    cheater_set: set,
    damping: float = 0.85,
    n_iterations: int = 20
) -> Dict[str, float]:
    """
    PageRank-style risk propagation.
    Risk flows from known cheaters through the network.
    """
    n_users = len(all_users)
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    
    # Initialize: cheaters start with high risk
    risk = np.zeros(n_users)
    for user in cheater_set:
        if user in user_to_idx:
            risk[user_to_idx[user]] = 1.0
    
    for _ in range(n_iterations):
        new_risk = np.zeros(n_users)
        
        for user in all_users:
            idx = user_to_idx[user]
            neighbors = adj.get(user, set())
            
            if not neighbors:
                new_risk[idx] = risk[idx]
                continue
            
            # Receive risk from neighbors
            incoming = sum(
                risk[user_to_idx[n]] / len(adj.get(n, {user}))
                for n in neighbors
                if n in user_to_idx
            )
            
            if user in cheater_set:
                # Cheaters maintain high risk
                new_risk[idx] = 0.5 + 0.5 * damping * incoming
            else:
                new_risk[idx] = damping * incoming
        
        # Normalize
        max_risk = new_risk.max()
        if max_risk > 0:
            new_risk = new_risk / max_risk
        
        risk = new_risk
    
    return {user: risk[user_to_idx[user]] for user in all_users}


def find_connected_components(
    all_users: np.ndarray,
    adj: Dict[str, set]
) -> Tuple[Dict[str, int], Dict[int, int]]:
    """Find connected components and their sizes."""
    visited = set()
    user_to_component = {}
    component_sizes = {}
    component_id = 0
    
    for start_user in all_users:
        if start_user in visited:
            continue
        
        # BFS
        queue = [start_user]
        component_users = []
        
        while queue:
            user = queue.pop(0)
            if user in visited:
                continue
            visited.add(user)
            component_users.append(user)
            
            for neighbor in adj.get(user, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        # Assign component
        for user in component_users:
            user_to_component[user] = component_id
        component_sizes[component_id] = len(component_users)
        component_id += 1
    
    return user_to_component, component_sizes


def build_graph_features_advanced(
    train: pd.DataFrame,
    test: pd.DataFrame,
    graph_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build comprehensive graph features."""
    
    if graph_df is None:
        return train, test
    
    print("\n" + "=" * 60)
    print("BUILDING ADVANCED GRAPH FEATURES")
    print("=" * 60)
    
    # Get all users and labels
    all_users = pd.concat([train['user_hash'], test['user_hash']]).unique()
    labeled = train[train['is_cheating'].notna()]
    cheater_set = set(labeled[labeled['is_cheating'] == 1]['user_hash'])
    clean_set = set(labeled[labeled['is_cheating'] == 0]['user_hash'])
    base_rate = len(cheater_set) / (len(cheater_set) + len(clean_set))
    
    print(f"  Users in dataset: {len(all_users):,}")
    print(f"  Cheaters: {len(cheater_set):,}, Clean: {len(clean_set):,}")
    print(f"  Base rate: {base_rate:.4f}")
    
    # Build adjacency
    print("  Building adjacency list...")
    adj = build_adjacency(graph_df)
    print(f"  Users with edges: {len(adj):,}")
    
    # Multi-hop features
    print("  Computing multi-hop features...")
    hop_features = compute_multi_hop_features(all_users, adj, cheater_set, clean_set, base_rate)
    
    # Label propagation
    print("  Running label propagation...")
    known_labels = {row['user_hash']: row['is_cheating'] 
                    for _, row in labeled.iterrows()}
    lp_scores = label_propagation(all_users, adj, known_labels, n_iterations=15, alpha=0.6)
    hop_features['graph_label_prop_score'] = hop_features['user_hash'].map(lp_scores)
    
    # PageRank risk
    print("  Computing PageRank risk...")
    pr_scores = pagerank_risk(all_users, adj, cheater_set, damping=0.85, n_iterations=25)
    hop_features['graph_pagerank_risk'] = hop_features['user_hash'].map(pr_scores)
    
    # Connected components
    print("  Finding connected components...")
    user_to_comp, comp_sizes = find_connected_components(all_users, adj)
    hop_features['graph_component_id'] = hop_features['user_hash'].map(user_to_comp)
    hop_features['graph_component_size'] = hop_features['graph_component_id'].map(comp_sizes)
    hop_features['graph_log_component_size'] = np.log1p(hop_features['graph_component_size'])
    
    # Component-level cheater rate
    comp_cheater_counts = defaultdict(int)
    comp_labeled_counts = defaultdict(int)
    for user in all_users:
        comp = user_to_comp.get(user)
        if comp is not None:
            if user in cheater_set:
                comp_cheater_counts[comp] += 1
                comp_labeled_counts[comp] += 1
            elif user in clean_set:
                comp_labeled_counts[comp] += 1
    
    comp_rates = {
        comp: comp_cheater_counts[comp] / comp_labeled_counts[comp] 
        if comp_labeled_counts[comp] > 0 else np.nan
        for comp in comp_sizes
    }
    hop_features['graph_component_cheat_rate'] = hop_features['graph_component_id'].map(comp_rates)
    
    # Neighbor degree stats
    print("  Computing neighbor statistics...")
    neighbor_stats = []
    for user in all_users:
        neighbors = adj.get(user, set())
        if neighbors:
            degrees = [len(adj.get(n, set())) for n in neighbors]
            neighbor_stats.append({
                'user_hash': user,
                'graph_neighbor_degree_mean': np.mean(degrees),
                'graph_neighbor_degree_max': np.max(degrees),
                'graph_neighbor_degree_std': np.std(degrees) if len(degrees) > 1 else 0,
            })
        else:
            neighbor_stats.append({
                'user_hash': user,
                'graph_neighbor_degree_mean': 0,
                'graph_neighbor_degree_max': 0,
                'graph_neighbor_degree_std': 0,
            })
    
    neighbor_df = pd.DataFrame(neighbor_stats)
    hop_features = hop_features.merge(neighbor_df, on='user_hash', how='left')
    
    # Clustering coefficient approximation (triangle density)
    print("  Computing clustering coefficients...")
    clustering = []
    for user in all_users:
        neighbors = list(adj.get(user, set()))
        if len(neighbors) < 2:
            clustering.append({'user_hash': user, 'graph_clustering': 0})
            continue
        
        # Count edges between neighbors (sample if too many)
        if len(neighbors) > 100:
            neighbors = list(np.random.choice(neighbors, 100, replace=False))
        
        triangles = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if n2 in adj.get(n1, set()):
                    triangles += 1
        
        max_triangles = len(neighbors) * (len(neighbors) - 1) / 2
        cc = triangles / max_triangles if max_triangles > 0 else 0
        clustering.append({'user_hash': user, 'graph_clustering': cc})
    
    clustering_df = pd.DataFrame(clustering)
    hop_features = hop_features.merge(clustering_df, on='user_hash', how='left')
    
    # Merge to train/test
    print("  Merging features...")
    graph_cols = [c for c in hop_features.columns if c != 'user_hash']
    train = train.merge(hop_features, on='user_hash', how='left')
    test = test.merge(hop_features, on='user_hash', how='left')
    
    print(f"  Added {len(graph_cols)} graph features")
    
    return train, test


def engineer_features_advanced(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Advanced feature engineering."""
    df = df.copy()
    
    # Missingness features
    df['missing_count'] = df[FEATURE_COLS].isna().sum(axis=1)
    df['missing_ratio'] = df['missing_count'] / len(FEATURE_COLS)
    
    for col in FEATURE_COLS:
        df[f'{col}_missing'] = df[col].isna().astype(int)
    
    # Aggregations
    numeric = df[FEATURE_COLS]
    df['feature_mean'] = numeric.mean(axis=1)
    df['feature_std'] = numeric.std(axis=1)
    df['feature_min'] = numeric.min(axis=1)
    df['feature_max'] = numeric.max(axis=1)
    df['feature_range'] = df['feature_max'] - df['feature_min']
    df['feature_skew'] = numeric.skew(axis=1)
    df['feature_sum'] = numeric.sum(axis=1)
    
    # Specific feature transforms
    for col in FEATURE_COLS:
        vals = df[col].fillna(0)
        
        # Log transform for skewed features
        if col in ['feature_010', 'feature_015', 'feature_016']:
            df[f'{col}_log'] = np.log1p(vals.clip(lower=0))
        
        # Square for potentially quadratic relationships
        df[f'{col}_sq'] = vals ** 2
    
    # Binary feature combinations
    binary_cols = ['feature_007', 'feature_011', 'feature_013', 'feature_014']
    for i, c1 in enumerate(binary_cols):
        for c2 in binary_cols[i+1:]:
            v1 = df[c1].fillna(0)
            v2 = df[c2].fillna(0)
            df[f'{c1}_{c2}_and'] = (v1 * v2).astype(int)
            df[f'{c1}_{c2}_or'] = ((v1 + v2) > 0).astype(int)
    
    # Ratio features
    for c1, c2 in [('feature_002', 'feature_003'), ('feature_004', 'feature_006'),
                    ('feature_008', 'feature_009')]:
        v1 = df[c1].fillna(0)
        v2 = df[c2].fillna(1).replace(0, 1)
        df[f'{c1}_div_{c2}'] = v1 / v2
    
    # Interaction with graph features
    if 'graph_rate_1hop' in df.columns:
        df['graph_rate_1hop_x_missing'] = df['graph_rate_1hop'].fillna(0) * df['missing_count']
        df['graph_pagerank_x_feat_mean'] = df.get('graph_pagerank_risk', 0) * df['feature_mean'].fillna(0)
    
    return df


def compute_cost_score(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute the competition cost metric.
    Returns: (best_cost, best_t_low, best_t_high)
    """
    best_cost = float('inf')
    best_t_low = 0.0
    best_t_high = 1.0
    
    # Grid search over thresholds
    for t_low in np.arange(0.05, 0.5, 0.02):
        for t_high in np.arange(t_low + 0.05, 0.95, 0.02):
            cost = 0
            for yt, yp in zip(y_true, y_pred):
                if yp < t_low:
                    # Auto-pass
                    if yt == 1:
                        cost += COST_FN  # False negative
                elif yp > t_high:
                    # Auto-block
                    if yt == 0:
                        cost += COST_FP_BLOCK  # False positive
                else:
                    # Manual review
                    if yt == 0:
                        cost += COST_FP_REVIEW  # FP in review
                    else:
                        cost += COST_TP_REVIEW  # TP in review
            
            if cost < best_cost:
                best_cost = cost
                best_t_low = t_low
                best_t_high = t_high
    
    return best_cost, best_t_low, best_t_high


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    scale_pos: float
) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train LightGBM model."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 10,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'seed': RANDOM_STATE,
        'scale_pos_weight': scale_pos,
        'n_jobs': -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params, train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)
    
    return model, train_pred, val_pred


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos: float
) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train XGBoost model."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': RANDOM_STATE,
        'scale_pos_weight': scale_pos,
        'n_jobs': -1,
        'verbosity': 0,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    val_pred = model.predict(dval)
    train_pred = model.predict(dtrain)
    
    return model, train_pred, val_pred


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos: float
) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train CatBoost model."""
    model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        scale_pos_weight=scale_pos,
        random_seed=RANDOM_STATE,
        verbose=0,
        early_stopping_rounds=100,
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    
    val_pred = model.predict_proba(X_val)[:, 1]
    train_pred = model.predict_proba(X_train)[:, 1]
    
    return model, train_pred, val_pred


def train_ensemble(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_pseudo_iterations: int = 3
) -> np.ndarray:
    """Train ensemble of models with pseudo-labeling."""
    
    print("\n" + "=" * 60)
    print("TRAINING ENSEMBLE")
    print("=" * 60)
    
    # Use high_conf_clean as pseudo-negatives
    pseudo_mask = (train_df['high_conf_clean'] == 1.0) & (train_df['is_cheating'].isna())
    train_df = train_df.copy()
    train_df.loc[pseudo_mask, 'is_cheating'] = 0.0
    print(f"  Added {pseudo_mask.sum():,} high-confidence pseudo-negatives")
    
    # Get labeled data
    labeled = train_df[train_df['is_cheating'].notna()].copy()
    
    # Feature columns
    exclude = {'user_hash', 'is_cheating', 'high_conf_clean', 'graph_component_id'}
    feature_cols = [c for c in labeled.columns 
                    if c not in exclude 
                    and labeled[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    print(f"  Features: {len(feature_cols)}")
    
    X = labeled[feature_cols].values
    y = labeled['is_cheating'].values
    X_test = test_df[feature_cols].values
    
    # Fill NaN
    X = np.nan_to_num(X, nan=-999)
    X_test = np.nan_to_num(X_test, nan=-999)
    
    # Storage for predictions
    n_models = sum([HAS_LGB, HAS_XGB, HAS_CB])
    if n_models == 0:
        raise RuntimeError("No gradient boosting library available!")
    
    oof_preds = {name: np.zeros(len(labeled)) for name in ['lgb', 'xgb', 'cb'] if eval(f'HAS_{name.upper()}')}
    test_preds = {name: np.zeros(len(test_df)) for name in ['lgb', 'xgb', 'cb'] if eval(f'HAS_{name.upper()}')}
    
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1) * 1.5
        
        # Train models
        if HAS_LGB:
            model_lgb, _, pred_lgb = train_lightgbm(X_tr, y_tr, X_val, y_val, feature_cols, scale_pos)
            oof_preds['lgb'][val_idx] = pred_lgb
            test_preds['lgb'] += model_lgb.predict(X_test) / N_FOLDS
            print(f"    LGB AUC: {roc_auc_score(y_val, pred_lgb):.4f}")
        
        if HAS_XGB:
            model_xgb, _, pred_xgb = train_xgboost(X_tr, y_tr, X_val, y_val, scale_pos)
            oof_preds['xgb'][val_idx] = pred_xgb
            test_preds['xgb'] += model_xgb.predict(xgb.DMatrix(X_test)) / N_FOLDS
            print(f"    XGB AUC: {roc_auc_score(y_val, pred_xgb):.4f}")
        
        if HAS_CB:
            model_cb, _, pred_cb = train_catboost(X_tr, y_tr, X_val, y_val, scale_pos)
            oof_preds['cb'][val_idx] = pred_cb
            test_preds['cb'] += model_cb.predict_proba(X_test)[:, 1] / N_FOLDS
            print(f"    CB AUC: {roc_auc_score(y_val, pred_cb):.4f}")
    
    # Combine OOF predictions
    print("\n  Combining models...")
    oof_stack = np.column_stack(list(oof_preds.values()))
    test_stack = np.column_stack(list(test_preds.values()))
    
    # Simple average
    final_oof = np.mean(oof_stack, axis=1)
    final_test = np.mean(test_stack, axis=1)
    
    # Train meta-learner on OOF
    print("  Training stacking meta-learner...")
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(oof_stack, y)
    
    final_oof_meta = meta.predict_proba(oof_stack)[:, 1]
    final_test_meta = meta.predict_proba(test_stack)[:, 1]
    
    # Evaluate
    auc_simple = roc_auc_score(y, final_oof)
    auc_meta = roc_auc_score(y, final_oof_meta)
    
    print(f"\n  Simple average AUC: {auc_simple:.4f}")
    print(f"  Meta-learner AUC: {auc_meta:.4f}")
    
    # Use better one
    if auc_meta > auc_simple:
        final_test = final_test_meta
        final_oof = final_oof_meta
        print("  Using meta-learner predictions")
    else:
        print("  Using simple average predictions")
    
    # Compute cost score
    best_cost, t_low, t_high = compute_cost_score(y, final_oof)
    print(f"\n  Best cost score: -{best_cost:,.0f}")
    print(f"  Optimal thresholds: t_low={t_low:.3f}, t_high={t_high:.3f}")
    
    # Feature importance (from last LGB model)
    if HAS_LGB:
        imp = model_lgb.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:15]
        print("\n  Top 15 Features:")
        for i, (name, val) in enumerate(feat_imp):
            print(f"    {i+1:2d}. {name}: {val:.0f}")
    
    return final_test


def calibrate_predictions(preds: np.ndarray) -> np.ndarray:
    """Apply isotonic/Platt calibration if needed."""
    # Simple clipping to ensure valid probabilities
    preds = np.clip(preds, 0.001, 0.999)
    return preds


def main():
    parser = argparse.ArgumentParser(description="Cheating Detection V4 - Maximum Accuracy")
    parser.add_argument("--train", default="sample-data/train.csv")
    parser.add_argument("--test", default="sample-data/test.csv")
    parser.add_argument("--graph", default="sample-data/social_graph.csv")
    parser.add_argument("--output", default="submission_v4.csv")
    args = parser.parse_args()
    
    # Auto-detect data paths
    for base in ["", "sample-data/", "data/"]:
        if Path(f"{base}train.csv").exists():
            args.train = f"{base}train.csv"
            args.test = f"{base}test.csv"
            args.graph = f"{base}social_graph.csv"
            break
    
    print("=" * 60)
    print("CHEATING DETECTOR V4 - MAXIMUM ACCURACY")
    print("=" * 60)
    print(f"LightGBM: {'✓' if HAS_LGB else '✗'}")
    print(f"XGBoost:  {'✓' if HAS_XGB else '✗'}")
    print(f"CatBoost: {'✓' if HAS_CB else '✗'}")
    
    # Load data
    train, test, graph = load_data(args.train, args.test, args.graph)
    
    # Build graph features
    if graph is not None:
        train, test = build_graph_features_advanced(train, test, graph)
    
    # Feature engineering
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    train = engineer_features_advanced(train, is_train=True)
    test = engineer_features_advanced(test, is_train=False)
    print(f"  Total features: {len([c for c in train.columns if c not in ['user_hash', 'is_cheating', 'high_conf_clean']])}")
    
    # Train ensemble
    test_preds = train_ensemble(train, test)
    
    # Calibrate
    test_preds = calibrate_predictions(test_preds)
    
    # Save submission
    submission = pd.DataFrame({
        'user_hash': test['user_hash'],
        'prediction': test_preds
    })
    submission.to_csv(args.output, index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Saved: {args.output}")
    print(f"Predictions range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"Predictions mean: {test_preds.mean():.4f}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

