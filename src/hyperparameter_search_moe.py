#!/usr/bin/env python3
"""
hyperparameter search script - run before each round to tune MoE params
Random search for N iterations
"""

import argparse
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from datetime import datetime
from tqdm import tqdm

# Import training and prediction functions
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import (
    normalize_column_name,
    find_smiles_column,
    smiles_to_morgan_bits,
    compute_chemberta_embeddings,
    align_and_hstack,
    build_numeric_from_columns,
    find_column_by_key,
)
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split


def random_sample_params():
    """Randomly sample a hyperparameter set"""
    params = {
        # expert 1 (Morgan FP)
        'xgb1_n_estimators': random.randint(300, 1000),
        'xgb1_lr': random.uniform(0.01, 0.3),
        'xgb1_max_depth': random.randint(4, 10),
        'xgb1_subsample': random.uniform(0.7, 1.0),
        'xgb1_colsample_bytree': random.uniform(0.5, 1.0),
        
        # expert 2 (ChemBERTa)
        'xgb2_n_estimators': random.randint(300, 1000),
        'xgb2_lr': random.uniform(0.01, 0.3),
        'xgb2_max_depth': random.randint(4, 10),
        'xgb2_subsample': random.uniform(0.7, 1.0),
        'xgb2_colsample_bytree': random.uniform(0.5, 1.0),
        
        # Gate
        'gate_l1': random.uniform(0.001, 0.1),
    }
    return params


def optuna_sample_params(trial):
    """Sample a hyperparameter set with an Optuna trial"""
    params = {
        'xgb1_n_estimators': trial.suggest_int('xgb1_n_estimators', 300, 1000),
        'xgb1_lr': trial.suggest_float('xgb1_lr', 0.01, 0.3, log=True),
        'xgb1_max_depth': trial.suggest_int('xgb1_max_depth', 4, 10),
        'xgb1_subsample': trial.suggest_float('xgb1_subsample', 0.7, 1.0),
        'xgb1_colsample_bytree': trial.suggest_float('xgb1_colsample_bytree', 0.5, 1.0),
        'xgb2_n_estimators': trial.suggest_int('xgb2_n_estimators', 300, 1000),
        'xgb2_lr': trial.suggest_float('xgb2_lr', 0.01, 0.3, log=True),
        'xgb2_max_depth': trial.suggest_int('xgb2_max_depth', 4, 10),
        'xgb2_subsample': trial.suggest_float('xgb2_subsample', 0.7, 1.0),
        'xgb2_colsample_bytree': trial.suggest_float('xgb2_colsample_bytree', 0.5, 1.0),
        'gate_l1': trial.suggest_float('gate_l1', 0.001, 0.1, log=True),
    }
    return params


def train_and_evaluate(X_morgan, X_chem, y_train, y_val, params, device='cpu'):
    """
    Train a MoE model with given params and evaluate on validation/test set
    
    Note: y_train and y_val are already separated
          X_morgan and X_chem are [train; val] concatenated
    
    Returns validation PCC
    """
    n_train = len(y_train)
    n_val = len(y_val)
    
    X_ecfp_train = X_morgan[:n_train]
    X_ecfp_val = X_morgan[n_train:n_train+n_val]
    X_chem_train = X_chem[:n_train]
    X_chem_val = X_chem[n_train:n_train+n_val]
    
    # extractparams
    xgb1_params = {
        'n_estimators': params['xgb1_n_estimators'],
        'learning_rate': params['xgb1_lr'],
        'max_depth': params['xgb1_max_depth'],
        'subsample': params['xgb1_subsample'],
        'colsample_bytree': params['xgb1_colsample_bytree'],
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': device,
        'random_state': 42,
    }
    
    xgb2_params = {
        'n_estimators': params['xgb2_n_estimators'],
        'learning_rate': params['xgb2_lr'],
        'max_depth': params['xgb2_max_depth'],
        'subsample': params['xgb2_subsample'],
        'colsample_bytree': params['xgb2_colsample_bytree'],
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': device,
        'random_state': 42,
    }
    
    # Training expert 1
    xgb1 = XGBRegressor(**xgb1_params)
    xgb1.fit(X_ecfp_train, y_train)
    
    # Training expert 2
    xgb2 = XGBRegressor(**xgb2_params)
    xgb2.fit(X_chem_train, y_train)
    
    # Training Gate
    pred1_train = xgb1.predict(X_ecfp_train)
    pred2_train = xgb2.predict(X_chem_train)
    gate_X_train = np.stack([pred1_train, pred2_train], axis=1)
    
    gate = ElasticNet(alpha=params['gate_l1'], l1_ratio=0.9, random_state=42, max_iter=20000)
    gate.fit(gate_X_train, y_train)
    
    # validation predictions
    pred1_val = xgb1.predict(X_ecfp_val)
    pred2_val = xgb2.predict(X_chem_val)
    gate_X_val = np.stack([pred1_val, pred2_val], axis=1)
    y_pred_val = gate.intercept_ + gate_X_val @ gate.coef_
    
    # Compute PCC
    pcc = pearsonr(y_val, y_pred_val)[0]
    
    return pcc


def hyperparameter_search(
    csv_path,
    target,
    test_csv=None,
    n_iterations=300,
    output_dir='hyperparam_search',
    device='auto',
    search_method='random',
    seed=42,
    optuna_patience=500,
    random_fallback_iters=300,
):
    """
    Hyperparameter random search - evaluate on an independent test set
    
    Args:
        csv_path: Training data CSV
        target: Target column name
        test_csv: test CSV (if None, split 20% from training as validation)
        n_iterations: Number of search iterations
        output_dir: Output directory
        device: Compute device
    """
    print(f"="*80)
    print(f"Hyperparameter search: {target}")
    print(f"="*80)
    print(f"Training data: {csv_path}")
    if test_csv:
        print(f"test data: {test_csv}")
    print(f"iterations: {n_iterations}")
    print()
    
    # ReadTraining data
    df_train = pd.read_csv(csv_path)
    rename_map = {c: normalize_column_name(c) for c in df_train.columns}
    df_train.rename(columns=rename_map, inplace=True)
    
    target_col = normalize_column_name(target)
    smiles_col = find_smiles_column(list(df_train.columns))
    
    # Check if we should include Initiator Percentage (%)
    import os as _os
    WITH_INITIATOR_PCT = _os.environ.get("WITH_INITIATOR_PCT", "1") in ("1", "true", "True")
    
    # Build training features
    print("Build training features...")
    smiles_train = df_train[smiles_col].astype(str).tolist()
    X_morgan_train, idx_morgan_train = smiles_to_morgan_bits(smiles_train, radius=2, n_bits=2048, cache_dir='.cache_features')
    X_chem_train, idx_chem_train = compute_chemberta_embeddings(
        smiles_train,
        hf_model_name="DeepChem/ChemBERTa-77M-MLM",
        batch_size=32,
        device=device,
        cache_dir='.cache_features'
    )
    
    # Build feature list
    feats_train = [(X_morgan_train, idx_morgan_train), (X_chem_train, idx_chem_train)]
    if WITH_INITIATOR_PCT:
        pct_col = find_column_by_key(list(df_train.columns), "Initiator Percentage (%)")
        if pct_col is None:
            print("⚠️  Warning: WITH_INITIATOR_PCT=1 but no Initiator Percentage (%) column found, skipping")
        else:
            print(f"✓ add Initiator Percentage (%) features: {pct_col}")
            X_num_train, idx_num_train = build_numeric_from_columns(df_train, [pct_col])
            feats_train.append((X_num_train, idx_num_train))
    
    X_all_train, common_idx_train = align_and_hstack(feats_train)
    n_morgan = X_morgan_train.shape[1]
    X_ecfp_train = X_all_train[:, :n_morgan]
    X_chem_train_only = X_all_train[:, n_morgan:]
    y_train = df_train[target_col].values[common_idx_train].astype(float)
    
    # Build test features
    if test_csv:
        print("Build test features...")
        df_test = pd.read_csv(test_csv)
        df_test.rename(columns=rename_map, inplace=True)
        smiles_test = df_test[smiles_col].astype(str).tolist()
        
        X_morgan_test, idx_morgan_test = smiles_to_morgan_bits(smiles_test, radius=2, n_bits=2048, cache_dir='.cache_features')
        X_chem_test, idx_chem_test = compute_chemberta_embeddings(
            smiles_test,
            hf_model_name="DeepChem/ChemBERTa-77M-MLM",
            batch_size=32,
            device=device,
            cache_dir='.cache_features'
        )
        
        # Build test feature list
        feats_test = [(X_morgan_test, idx_morgan_test), (X_chem_test, idx_chem_test)]
        if WITH_INITIATOR_PCT:
            pct_col = find_column_by_key(list(df_test.columns), "Initiator Percentage (%)")
            if pct_col is None:
                print("⚠️  Warning: Initiator Percentage (%) column not found in test set")
            else:
                X_num_test, idx_num_test = build_numeric_from_columns(df_test, [pct_col])
                feats_test.append((X_num_test, idx_num_test))
        
        X_all_test, common_idx_test = align_and_hstack(feats_test)
        X_ecfp_test = X_all_test[:, :n_morgan]
        X_chem_test_only = X_all_test[:, n_morgan:]
        y_test = df_test[target_col].values[common_idx_test].astype(float)
        
        # Combined features for the training function [training set; test set]
        X_morgan_combined = np.vstack([X_ecfp_train, X_ecfp_test])
        X_chem_combined = np.vstack([X_chem_train_only, X_chem_test_only])
        
        print(f"Training set: {len(y_train)} samples")
        print(f"Test set: {len(y_test)} samples (for best-params evaluation)")
        print()
        print("⚠️ Note: every iteration trains on the full training set and evaluates directly on test set")
        print("         Goal: find the best-performing params on the test set (explore performance ceiling)")
    else:
        # split from training set 80/20
        print("Split a validation set from training (80/20)...")
        indices = np.arange(len(y_train))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        y_test = y_train[test_idx]
        y_train_split = y_train[train_idx]
        
        X_morgan_combined = np.vstack([X_ecfp_train[train_idx], X_ecfp_train[test_idx]])
        X_chem_combined = np.vstack([X_chem_train_only[train_idx], X_chem_train_only[test_idx]])
        
        y_train = y_train_split
        print(f"Training set: {len(y_train)} samples")
        print(f"validation set: {len(y_test)} samples (split from training set)")
    
    print()
    
    # device
    if device == 'auto':
        try:
            import torch
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        except:
            device_str = 'cpu'
    else:
        device_str = device
    
    print(f"device: {device_str}")
    print()
    print("StartingHyperparameter search...")
    print(f"Evaluation metric: test PCC (Pearson correlation)")
    print("-"*80)
    
    best_pcc = -999
    best_params = None
    history = []
    random.seed(seed)
    np.random.seed(seed)
    search_method = search_method.lower()
    print(f"search method: {search_method}")
    global_iter = 0

    def record_history(method_label, method_iter, params, pcc, extra=None):
        nonlocal history, global_iter
        global_iter += 1
        entry = {
            'global_iteration': global_iter,
            'method': method_label,
            'method_iteration': method_iter,
            'test_pcc': pcc,
        }
        entry.update(params)
        if extra:
            entry.update(extra)
        history.append(entry)

    def handle_result(params, pcc, method_desc):
        nonlocal best_pcc, best_params
        improved = False
        if pcc is not None and np.isfinite(pcc) and pcc > best_pcc:
            best_pcc = float(pcc)
            best_params = params
            improved = True
            print(f"{method_desc} ✓ new best! test PCC = {pcc:.4f}")
        return improved

    if search_method == 'random':
        pbar = tqdm(range(n_iterations), desc=f"Searchin (Best PCC: {best_pcc:.4f})", ncols=100)
        for i in pbar:
            params = random_sample_params()
            try:
                pcc = train_and_evaluate(
                    X_morgan_combined, X_chem_combined,
                    y_train, y_test, params, device=device_str
                )
            except Exception as e:
                tqdm.write(f"[{i+1}/{n_iterations}] ✗ Failed: {e}")
                pcc = np.nan
            record_history('random', i + 1, params, pcc)
            if handle_result(params, pcc, f"[Random {i+1}/{n_iterations}]"):
                pbar.set_description(f"Searchin (Best PCC: {best_pcc:.4f}) ✓ new best!")
            else:
                pbar.set_description(f"Searchin (Best PCC: {best_pcc:.4f})")
        pbar.close()
    elif search_method == 'optuna':
        try:
            import optuna
        except ImportError as e:
            raise ImportError("optuna is required for Bayesian optimization (pip install optuna)") from e

        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        optuna_trials_done = 0
        optuna_since_improve = 0
        fallback_runs = 0

        while optuna_trials_done < n_iterations:
            trial = study.ask()
            params = optuna_sample_params(trial)
            try:
                pcc = train_and_evaluate(
                    X_morgan_combined, X_chem_combined,
                    y_train, y_test, params, device=device_str
                )
                study.tell(trial, float(pcc))
            except Exception as e:
                print(f"[Optuna Trial {optuna_trials_done+1}] ✗ Failed: {e}")
                study.tell(trial, float('nan'))
                pcc = np.nan

            optuna_trials_done += 1
            record_history('optuna', optuna_trials_done, params, pcc)
            if handle_result(params, pcc, f"[Optuna Trial {optuna_trials_done}/{n_iterations}]"):
                optuna_since_improve = 0
            else:
                optuna_since_improve += 1

            if optuna_since_improve >= optuna_patience and optuna_trials_done < n_iterations:
                fallback_runs += 1
                print(f"▲ Optuna stalled for {optuna_patience} trials, triggering random fallback #{fallback_runs} ({random_fallback_iters} trials)")
                random_improved = False
                for r in range(random_fallback_iters):
                    params_rand = random_sample_params()
                    try:
                        pcc_rand = train_and_evaluate(
                            X_morgan_combined, X_chem_combined,
                            y_train, y_test, params_rand, device=device_str
                        )
                    except Exception as e:
                        print(f"  [Random Fallback {r+1}/{random_fallback_iters}] ✗ Failed: {e}")
                        pcc_rand = np.nan
                    record_history('random', r + 1, params_rand, pcc_rand, {'fallback_round': fallback_runs})
                    if handle_result(params_rand, pcc_rand, f"  [Random Fallback {r+1}/{random_fallback_iters}]"):
                        random_improved = True
                if not random_improved:
                    print("  random fallback no improvement, back to Optuna")
                optuna_since_improve = 0
    else:
        raise ValueError(f"unknown search method: {search_method}")
    
    print()
    print("="*80)
    print("Search done!")
    print("="*80)
    print(f"Best test PCC: {best_pcc:.4f}")
    print()
    print("best params:")
    if best_params:
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        print("  (for nownocanuseparams)")
    
    # Saveresult
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history) if history else pd.DataFrame()
    history_csv = output_path / f"{target}_search_history.csv"
    if not history_df.empty:
        history_df.to_csv(history_csv, index=False)
        top_k = history_df.sort_values('test_pcc', ascending=False, na_position='last').head(min(10, len(history_df)))
        top10_csv = output_path / f"{target}_top10_params.csv"
        top_k.to_csv(top10_csv, index=False)
        print()
        print(f"✓ Search history saved: {history_csv}")
        print(f"✓ Top 10 paramslist: {top10_csv}")
    else:
        top10_csv = output_path / f"{target}_top10_params.csv"
        print("⚠️ no history recorded; no search CSV generated")

    meta_cols = {'global_iteration', 'method', 'method_iteration', 'test_pcc', 'fallback_round'}
    # Savebest params
    best_json = output_path / f"{target}_best_params.json"
    if best_params is None and not history_df.empty:
        history_df_sorted = history_df.sort_values('test_pcc', ascending=False, na_position='last')
        if not history_df_sorted.empty:
            best_row = history_df_sorted.iloc[0]
            param_cols = [c for c in history_df.columns if c not in meta_cols]
            best_params = {c: float(best_row[c]) for c in param_cols}
            best_pcc = float(best_row['test_pcc']) if np.isfinite(best_row['test_pcc']) else float('nan')

    result_info = {
        'best_test_pcc': best_pcc,
        'params': best_params,
        'n_iterations': n_iterations,
        'n_train': len(y_train),
        'n_test': len(y_test)
    }
    with open(best_json, 'w') as f:
        json.dump(result_info, f, indent=2)
    
    print()
    print(f"✓ Best params saved: {best_json}")
    
    return best_params, best_pcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE hyperparameter search - evaluate on test set")
    parser.add_argument("--csv", required=True, help="Training data CSV")
    parser.add_argument("--target", required=True, help="Target column (λex, λem, QY)")
    parser.add_argument("--test_csv", default=None, help="Independent test CSV (if omitted, split 20%% from training)")
    parser.add_argument("--n_iterations", type=int, default=300, help="Number of search iterations")
    parser.add_argument("--output_dir", default="hyperparam_search", help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--search_method", default="random", choices=["random", "optuna"], help="Search strategy: random or bayesian")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--optuna_patience", type=int, default=500, help="Switch to random after this many Optuna trials without improvement")
    parser.add_argument("--random_fallback_iters", type=int, default=300, help="Random fallback trials")
    args = parser.parse_args()
    
    best_params, best_pcc = hyperparameter_search(
        args.csv,
        args.target,
        test_csv=args.test_csv,
        n_iterations=args.n_iterations,
        output_dir=args.output_dir,
        device=args.device,
        search_method=args.search_method,
        seed=args.seed,
        optuna_patience=args.optuna_patience,
        random_fallback_iters=args.random_fallback_iters,
    )
