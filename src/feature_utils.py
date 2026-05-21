#!/usr/bin/env python3
"""
Feature engineering utility module
Provides: Morgan fingerprints, ChemBERTa embeddings, data preprocessing
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import os
import hashlib
import json


def normalize_column_name(name: str) -> str:
    """Normalize column name"""
    if name is None:
        return ""
    name = name.replace("\xa0", " ")
    name = " ".join(name.split())
    return name


def simplify_column_key(name: str) -> str:
    """Simplify column name for matching"""
    name = normalize_column_name(name)
    return "".join(name.split()).lower()


def find_column_by_key(columns: List[str], key: str) -> Optional[str]:
    """Find column by simplified key"""
    simplified_map = {simplify_column_key(c): c for c in columns}
    k = simplify_column_key(key)
    return simplified_map.get(k)


def find_smiles_column(columns: List[str]) -> str:
    """Find the SMILES column"""
    simplified_map = {simplify_column_key(c): c for c in columns}
    candidates = [
        "initiator+anhydride+epoxy+[*]",
        "initiator+anhydride+epoxy[*]",
        "smiles",
        "structure",
    ]
    for key in candidates:
        if key in simplified_map:
            return simplified_map[key]
    
    # Fallback: column containing +anhydride+epoxy
    for c in columns:
        s = simplify_column_key(c)
        if "+anhydride+epoxy" in s:
            return c
    
    raise ValueError("No usable SMILES column found. Make sure the CSV has 'Initiator +anhydride+epoxy+[*]' or a SMILES column.")


def smiles_to_morgan_bits(
    smiles_list: List[str], 
    radius: int = 2, 
    n_bits: int = 2048, 
    cache_dir: Optional[str] = None
) -> Tuple[np.ndarray, List[int]]:
    """Convert SMILES to Morgan fingerprints"""
    # lazy import RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem.rdfingerprintGenerator import getMorganGenerator
        from rdkit import dataStructs
    except Exception as e:
        raise ImportError("Need RDKit to compute Morgan fingerprints. Please install RDKit in the container or use ChemBERTa mode.") from e
    
    # disk cache
    cache_path = None
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            hasher = hashlib.md5()
            hasher.update("\n".join([str(s) for s in smiles_list]).encode("utf-8"))
            hasher.update(f"r={radius},b={n_bits}".encode("utf-8"))
            cache_path = os.path.join(cache_dir, f"morgan_r{radius}_b{n_bits}_" + hasher.hexdigest() + ".npz")
            if os.path.exists(cache_path):
                npz = np.load(cache_path, allow_pickle=False)
                return npz["X"], npz["valid_idx"].tolist()
        except Exception:
            cache_path = None
    
    feature_list: List[np.ndarray] = []
    valid_indices: List[int] = []
    
    for idx, smi in enumerate(smiles_list):
        if pd.isna(smi):
            continue
        smi_str = str(smi).strip()
        if not smi_str:
            continue
        mol = Chem.MolFromSmiles(smi_str)
        if mol is None:
            continue
        
        gen = getMorganGenerator(radius=int(radius), fpSize=int(n_bits), includeChirality=True)
        fp = gen.getfingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        dataStructs.ConvertToNumpyArray(fp, arr)
        feature_list.append(arr)
        valid_indices.append(idx)
    
    if not feature_list:
        raise ValueError("Could not generate any Morgan fingerprints from the provided SMILES; please check the data format.")
    
    X = np.stack(feature_list, axis=0)
    
    # Savecache
    if cache_path:
        try:
            np.savez_compressed(cache_path, X=X, valid_idx=np.array(valid_indices, dtype=np.int64))
        except Exception:
            pass
    
    return X, valid_indices


def compute_chemberta_embeddings(
    smiles_list: List[str],
    hf_model_name: str = "DeepChem/ChemBERTa-77M-MLM",
    batch_size: int = 32,
    max_length: int = 256,
    device: str = "auto",
    pooling: str = "mean",
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Use ChemBERTa to encode SMILES as sentence vectors"""
    
    def _resolve_device(preferred: str = "auto") -> str:
        try:
            import torch
        except Exception:
            return "cpu"
        if preferred == "cpu":
            return "cpu"
        if preferred == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _mean_pooling(last_hidden_state, attention_mask):
        import torch
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = torch.sum(masked, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom
    
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    dev = _resolve_device(device)
    
    # disk cache
    cache_path = None
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_key = {
                "hf": hf_model_name,
                "max_len": max_length,
                "pooling": pooling,
                "device": str(device),
                "smiles": [str(s) for s in smiles_list],
            }
            hasher = hashlib.md5(json.dumps(cache_key, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            cache_path = os.path.join(cache_dir, f"chemberta_{hasher.hexdigest()}.npz")
            if os.path.exists(cache_path):
                npz = np.load(cache_path, allow_pickle=False)
                return npz["X"], npz["valid_idx"].tolist()
        except Exception:
            cache_path = None
    
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModel.from_pretrained(hf_model_name, use_safetensors=True, add_pooling_layer=False)
    model.to(dev)
    model.eval()
    
    valid_idx: List[int] = []
    texts: List[str] = []
    
    for i, smi in enumerate(smiles_list):
        if pd.isna(smi):
            continue
        s = str(smi).strip()
        if not s:
            continue
        valid_idx.append(i)
        texts.append(s)
    
    if not texts:
        raise ValueError("SMILES 列as空, no法proceed ChemBERTa 编码")
    
    all_embeds: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        
        with torch.no_grad():
            out = model(**enc)
            last_hidden = out.last_hidden_state  # [B, L, H]
            
            if pooling == "mean":
                pooled = _mean_pooling(last_hidden, enc["attention_mask"])
            elif pooling == "cls":
                pooled = last_hidden[:, 0, :]
            else:
                pooled = _mean_pooling(last_hidden, enc["attention_mask"])
        
        all_embeds.append(pooled.detach().cpu().numpy())
    
    X = np.concatenate(all_embeds, axis=0)
    
    # Savecache
    if cache_path:
        try:
            np.savez_compressed(cache_path, X=X, valid_idx=np.array(valid_idx, dtype=np.int64))
        except Exception:
            pass
    
    return X, valid_idx


def build_numeric_from_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[np.ndarray, List[int]]:
    """fromnumeric列Buildfeatures"""
    if not columns:
        raise ValueError("必须提供至few一个numeric列")
    
    valid_mask = np.ones(len(df), dtype=bool)
    for c in columns:
        valid_mask &= df[c].notna().values
    
    idx = [int(i) for i, v in enumerate(valid_mask) if v]
    if not idx:
        raise ValueError("numeric列novalidsamples")
    
    data_cols = []
    for c in columns:
        col = df.loc[idx, c].astype(float).values.reshape(-1, 1)
        data_cols.append(col)
    
    X = np.concatenate(data_cols, axis=1)
    return X, idx


def align_and_hstack(features: List[Tuple[np.ndarray, List[int]]]) -> Tuple[np.ndarray, List[int]]:
    """Align multiple feature matrices by row and concatenate horizontally"""
    if not features:
        raise ValueError("No features to concatenate")
    
    index_sets = [set(idx) for _, idx in features]
    common_idx = sorted(set.intersection(*index_sets))
    if not common_idx:
        raise ValueError("No common valid samples across features")
    
    aligned_parts: List[np.ndarray] = []
    for Xi, idx_i in features:
        pos_map = {idx: i for i, idx in enumerate(idx_i)}
        aligned = np.stack([Xi[pos_map[i]] for i in common_idx], axis=0)
        aligned_parts.append(aligned)
    
    X = np.concatenate(aligned_parts, axis=1)
    return X, common_idx


