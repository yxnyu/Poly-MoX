#!/usr/bin/env python3
"""
RegMixcore module: data augmentation and SOTA methods
Refactored from train_models.py
Provides: RegMix augmentation, MoE fusion, Optuna tuning
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import json


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics"""
    r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"pearson_r": float(r), "r2": float(r2), "mae": float(mae), "rmse": rmse}


def regmix_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    mix_ratio: float = 0.5,
    k_neighbors: int = 5,
    aug_factor: float = 2.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RegMix data augmentation: mixed augmentation method for regression tasks
    
    params:
        X: Input feature matrix [N, D]
        y: Target values [N,]
        mix_ratio: Beta distribution parameters, controlling randomness of mixing ratio
        k_neighbors: number of neighbors per sample
        aug_factor: augmentation factor, generated data size = original data size * aug_factor
        random_state: random seed
    
    Returns:
        X_aug: augmented feature matrix
        y_aug: augmented target values
    """
    np.random.seed(random_state)
    n_samples = len(X)
    n_aug_samples = int(n_samples * aug_factor)
    
    # standardize features to improve neighbor search
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # build kNN search
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_samples), metric='euclidean')
    nbrs.fit(X_scaled)
    
    X_aug_list = []
    y_aug_list = []
    
    # asperoriginalsamplesgenerateaugmentationsamples
    samples_per_original = max(1, n_aug_samples // n_samples)
    
    for i in range(n_samples):
        # find k nearest neighbors of current sample
        distances, indices = nbrs.kneighbors([X_scaled[i]])
        neighbor_indices = indices[0][1:]  # exclude self (index 0)
        
        # if not enough neighbors, use all available neighbors
        if len(neighbor_indices) == 0:
            neighbor_indices = [i]  # if no neighborss, use self
        
        # generate multiple augmentations for the current sample
        for _ in range(samples_per_original):
            # randomly choose a neighbor
            if len(neighbor_indices) > 0:
                j = np.random.choice(neighbor_indices)
            else:
                j = i
            
            # sample a mixing coefficient from Beta distribution
            lambda_mix = np.random.beta(mix_ratio, mix_ratio)
            lambda_mix = np.clip(lambda_mix, 0.1, 0.9)  # avoid extreme values
            
            # mixingfeaturesandlabel
            x_mixed = lambda_mix * X[i] + (1 - lambda_mix) * X[j]
            y_mixed = lambda_mix * y[i] + (1 - lambda_mix) * y[j]
            
            X_aug_list.append(x_mixed)
            y_aug_list.append(y_mixed)
    
    # if generatedd n_samples is insufficient, fill with extra samples
    while len(X_aug_list) < n_aug_samples:
        i = np.random.randint(0, n_samples)
        distances, indices = nbrs.kneighbors([X_scaled[i]])
        neighbor_indices = indices[0][1:]
        
        if len(neighbor_indices) > 0:
            j = np.random.choice(neighbor_indices)
        else:
            j = i
            
        lambda_mix = np.random.beta(mix_ratio, mix_ratio)
        lambda_mix = np.clip(lambda_mix, 0.1, 0.9)
        
        x_mixed = lambda_mix * X[i] + (1 - lambda_mix) * X[j]
        y_mixed = lambda_mix * y[i] + (1 - lambda_mix) * y[j]
        
        X_aug_list.append(x_mixed)
        y_aug_list.append(y_mixed)
    
    X_aug = np.array(X_aug_list[:n_aug_samples])
    y_aug = np.array(y_aug_list[:n_aug_samples])
    
    return X_aug, y_aug


def cross_validate_with_regmix(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    use_regmix: bool = False,
    regmix_params: Optional[Dict] = None,
) -> dict:
    """Cross-validation with RegMix augmentation"""
    
    # Detect GPU
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass
    
    device_str = "cuda" if gpu_available else "cpu"
    
    if regmix_params is None:
        regmix_params = {
            "mix_ratio": 0.5,
            "k_neighbors": 5,
            "aug_factor": 2.0
        }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    xgb_oof = np.zeros_like(y, dtype=float)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # apply RegMix data augmentation (only on training set)
        if use_regmix:
            try:
                X_aug, y_aug = regmix_augmentation(
                    X_train, y_train,
                    mix_ratio=regmix_params["mix_ratio"],
                    k_neighbors=regmix_params["k_neighbors"],
                    aug_factor=regmix_params["aug_factor"],
                    random_state=random_state
                )
                # Merge augmented data with original training data
                X_train = np.concatenate([X_train, X_aug], axis=0)
                y_train = np.concatenate([y_train, y_aug], axis=0)
            except Exception as e:
                print(f"RegMix augmentation failed; using original data: {e}")
        
        # XGBoostTraining
        xgb = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            device=device_str,
            random_state=random_state,
        )
        xgb.fit(X_train, y_train)
        xgb_oof[test_idx] = xgb.predict(X_test)
    
    return compute_metrics(y, xgb_oof)


def run_moe_fusion(
    X_morgan: np.ndarray,
    X_chem: np.ndarray, 
    y: np.ndarray,
    random_state: int = 42,
    gate_l1: float = 1e-3,
) -> Dict[str, float]:
    """
    Mixture-of-expertsfusion: 
    Expert 1: Morgan fingerprint + XGBoost
    expert2: ChemBERTavector + XGBoost  
    Gate: sparse linear (L1 regularization)
    """
    
    # Detect GPU
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass
    device_str = "cuda" if gpu_available else "cpu"
    
    # 5-fold CV: train two experts
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_morgan = np.zeros_like(y, dtype=float)
    oof_chem = np.zeros_like(y, dtype=float)
    
    for tr, te in kf.split(X_morgan):
        # Expert 1: Morgan fingerprint + XGBoost
        xgb1 = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            device=device_str,
            random_state=random_state,
        )
        xgb1.fit(X_morgan[tr], y[tr])
        oof_morgan[te] = xgb1.predict(X_morgan[te])
        
        # expert2: ChemBERTa + XGBoost
        xgb2 = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            device=device_str,
            random_state=random_state,
        )
        xgb2.fit(X_chem[tr], y[tr])
        oof_chem[te] = xgb2.predict(X_chem[te])
    
    # sparseGate: useexpert outputsasInputfeaturesfit target
    gate_X = np.stack([oof_morgan, oof_chem], axis=1)  # [N,2]
    gate = ElasticNet(alpha=float(gate_l1), l1_ratio=0.9, random_state=random_state)
    gate.fit(gate_X, y)
    
    # Final prediction (linear fusion)
    oof_final = gate.intercept_ + gate_X @ gate.coef_.astype(float)
    
    return compute_metrics(y, oof_final)


def optuna_xgb_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 600,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Baseline XGBoost + Optuna hyperparameter tuning"""
    try:
        import optuna
    except Exception as e:
        raise ImportError("optuna is required for hyperparameter tuning") from e
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "max_leaves": trial.suggest_int("max_leaves", 0, 256),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda",
            "random_state": random_state,
        }
        
        # 5-foldcross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_scores = []
        
        for train_idx, val_idx in kf.split(X):
            model = XGBRegressor(**params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            r2_scores.append(r2_score(y[val_idx], pred))
        
        return float(np.mean(r2_scores))
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=True)
    
    return {
        "best_params": study.best_params,
        "best_r2": float(study.best_value),
        "n_samples": len(y),
        "mode": "baseline"
    }


def optuna_xgb_regmix(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 600,
    random_state: int = 42,
) -> Dict[str, Any]:
    """RegMix-augmented XGBoost + Optuna hyperparameter tuning"""
    try:
        import optuna
    except Exception as e:
        raise ImportError("optuna is required for hyperparameter tuning") from e
    
    def objective(trial):
        # XGBoost超params
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "max_leaves": trial.suggest_int("max_leaves", 0, 256),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda",
            "random_state": random_state,
        }
        
        # RegMix超params
        regmix_params = {
            "mix_ratio": trial.suggest_float("regmix_mix_ratio", 0.1, 0.9),
            "k_neighbors": trial.suggest_int("regmix_k_neighbors", 3, 10),
            "aug_factor": trial.suggest_float("regmix_aug_factor", 1.2, 3.0)
        }
        
        # 5-foldcross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # applyRegMixaugmentation
            try:
                X_aug, y_aug = regmix_augmentation(
                    X_train, y_train,
                    mix_ratio=regmix_params["mix_ratio"],
                    k_neighbors=regmix_params["k_neighbors"],
                    aug_factor=regmix_params["aug_factor"],
                    random_state=random_state
                )
                X_train_combined = np.concatenate([X_train, X_aug], axis=0)
                y_train_combined = np.concatenate([y_train, y_aug], axis=0)
            except Exception:
                # ifRegMixfailed, use original data
                X_train_combined = X_train
                y_train_combined = y_train
            
            model = XGBRegressor(**params)
            model.fit(X_train_combined, y_train_combined)
            pred = model.predict(X_val)
            r2_scores.append(r2_score(y_val, pred))
        
        return float(np.mean(r2_scores))
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=True)
    
    return {
        "best_params": study.best_params,
        "best_r2": float(study.best_value),
        "n_samples": len(y),
        "mode": "regmix_enhanced"
    }


def optuna_moe(
    X_morgan: np.ndarray,
    X_chem: np.ndarray,
    y: np.ndarray,
    n_trials: int = 600,
    random_state: int = 42,
    storage: str = None,
    study_name: str = None,
    target: str = None,
) -> Dict[str, Any]:
    """MoE + Optuna hyperparameter tuning (supports SQLite storage for resume)"""
    try:
        import optuna
    except Exception as e:
        raise ImportError("optuna is required for hyperparameter tuning") from e
    
    # Detect GPU
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass
    device_str = "cuda" if gpu_available else "cpu"
    
    def objective(trial):
        # XGB expert 1 params
        params1 = {
            "n_estimators": trial.suggest_int("xgb1_n_estimators", 300, 1600),
            "learning_rate": trial.suggest_float("xgb1_lr", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("xgb1_max_depth", 3, 10),
            "subsample": trial.suggest_float("xgb1_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb1_colsample_bytree", 0.5, 1.0),
        }
        # XGB expert 2 params
        params2 = {
            "n_estimators": trial.suggest_int("xgb2_n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("xgb2_lr", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("xgb2_max_depth", 3, 8),
            "subsample": trial.suggest_float("xgb2_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb2_colsample_bytree", 0.5, 1.0),
        }
        gate_alpha = trial.suggest_float("gate_l1", 1e-4, 1e-2, log=True)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        oof_morgan = np.zeros_like(y, dtype=float)
        oof_chem = np.zeros_like(y, dtype=float)
        
        for tr, te in kf.split(X_morgan):
            xgb1 = XGBRegressor(
                **params1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                device=device_str,
                random_state=random_state,
            )
            xgb1.fit(X_morgan[tr], y[tr])
            oof_morgan[te] = xgb1.predict(X_morgan[te])
            
            xgb2 = XGBRegressor(
                **params2,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                device=device_str,
                random_state=random_state,
            )
            xgb2.fit(X_chem[tr], y[tr])
            oof_chem[te] = xgb2.predict(X_chem[te])
        
        gate_X = np.stack([oof_morgan, oof_chem], axis=1)
        gate = ElasticNet(alpha=float(gate_alpha), l1_ratio=0.9, random_state=random_state)
        gate.fit(gate_X, y)
        oof_final = gate.intercept_ + gate_X @ gate.coef_.astype(float)
        
        return float(r2_score(y, oof_final))
    
    if storage:
        study = optuna.create_study(
            direction="maximize",
            storage=storage,
            study_name=study_name or "moe_study",
            load_if_exists=True,
        )
        # Record reproducibility metadata so dump_study_best can recover them
        study.set_user_attr("seed", int(random_state))
        study.set_user_attr("n_samples", int(len(y)))
        if target is not None:
            study.set_user_attr("target", target)
        done = len(study.trials)
        remain = max(0, int(n_trials) - done)
        print(f"[Optuna] storage={storage} study={study_name or 'moe_study'} completed {done} trials, {remain} remaining")
        if remain > 0:
            study.optimize(objective, n_trials=remain, show_progress_bar=True)
    else:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=True)

    return {"best_params": study.best_params, "best_r2": float(study.best_value), "n_samples": int(len(y))}


