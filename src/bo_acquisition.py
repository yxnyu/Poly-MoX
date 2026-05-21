"""BO acquisition + diversity-aware batch selection for active learning.

Implements the methodology in the paper:
  - Stochastic BO acquisition: Thompson Sampling, EI, UCB
  - Morgan-fingerprint Tanimoto diversity
  - phase-wise weighted composite scoring with greedy batch construction
  - Subspace restriction (phosphine filter for Round 3)
  - Synthesizability filter (human-in-the-loop placeholder)
  - MoE-compatible σ proxy via expert disagreement

All scoring functions operate on numpy arrays for batch efficiency.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from feature_utils import (
    find_smiles_column,
    normalize_column_name,
    smiles_to_morgan_bits,
)


# ──────────────────────────────────────────────────────────────────────
# Tanimoto similarity & diversity scoring
# ──────────────────────────────────────────────────────────────────────

def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Tanimoto coefficient between two binary fingerprints."""
    fp1 = np.asarray(fp1, dtype=bool)
    fp2 = np.asarray(fp2, dtype=bool)
    inter = int(np.sum(fp1 & fp2))
    union = int(np.sum(fp1 | fp2))
    return inter / union if union else 0.0


def tanimoto_matrix(fps: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto similarity matrix for stack of binary fingerprints."""
    fps = fps.astype(np.int32)
    sizes = fps.sum(axis=1)
    inter = fps @ fps.T
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim


def diversity_score(
    candidate_fps: np.ndarray,
    selected_fps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """d_r(x; S) = min_{x' in S} (1 - Tanimoto(x, x')).

    if S is empty, returns all 1.0 (maximum diversity).
    """
    if selected_fps is None or len(selected_fps) == 0:
        return np.ones(len(candidate_fps))
    cand = candidate_fps.astype(np.int32)
    sel = selected_fps.astype(np.int32)
    cs = cand.sum(axis=1)
    ss = sel.sum(axis=1)
    inter = cand @ sel.T
    union = cs[:, None] + ss[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return (1.0 - sim).min(axis=1)


# ──────────────────────────────────────────────────────────────────────
# Min-max normalization
# ──────────────────────────────────────────────────────────────────────

def minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]; constant array → all 0.5."""
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


# ──────────────────────────────────────────────────────────────────────
# Acquisition functions
# ──────────────────────────────────────────────────────────────────────

def acquisition_ei(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
    xi: float = 0.0,
) -> np.ndarray:
    """Expected Improvement (maximization)."""
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def acquisition_ucb(mu: np.ndarray, sigma: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Upper Confidence Bound: μ + β σ."""
    return np.asarray(mu, dtype=float) + beta * np.asarray(sigma, dtype=float)


def acquisition_thompson_sample(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_samples: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Batched Thompson Sampling.

    Approximates the surrogate posterior at each candidate as N(μ, σ²) with
    independent draws (the MoE doesn't expose a joint posterior). Returns the
    sampled function values; downstream caller picks argmax for greedy.
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    samples = rng.normal(loc=mu, scale=sigma, size=(n_samples, len(mu)))
    return samples[0] if n_samples == 1 else samples


# ──────────────────────────────────────────────────────────────────────
# σ proxy for MoE (no native posterior)
# ──────────────────────────────────────────────────────────────────────

def expert_disagreement_sigma(
    pred_morgan: np.ndarray,
    pred_chem: np.ndarray,
) -> np.ndarray:
    """Half the |Morgan-expert - ChemBERTa-expert| as σ proxy.

    Captures regions where the two experts disagree, which usually signals
    extrapolation outside the training distribution.
    """
    return 0.5 * np.abs(np.asarray(pred_morgan) - np.asarray(pred_chem))


# ──────────────────────────────────────────────────────────────────────
# Composite score & phase-wise weighted batch selection
# ──────────────────────────────────────────────────────────────────────

# Diversity weight per active-learning round (paper §Morgan-fingerprint diversity).
PHASE_WEIGHTS = {1: 0.10, 2: 0.25, 3: 0.25}


def composite_score(a_norm: np.ndarray, d_norm: np.ndarray, w_r: float) -> np.ndarray:
    """s_r(x; S) = (1 - w_r) ã_r(x) + w_r d̃_r(x; S)."""
    return (1.0 - w_r) * a_norm + w_r * d_norm


def greedy_batch_select(
    acq_scores: np.ndarray,
    fps: np.ndarray,
    pool_ids: Sequence,
    batch_size: int = 12,
    w_r: float = 0.10,
) -> List:
    """Greedy batch construction maximizing the phase-wise composite score.

    Args:
        acq_scores: raw acquisition score per candidate (higher = better).
        fps:        binary Morgan fingerprints, shape (N_pool, D).
        pool_ids:   identifier (e.g. molecule ID) per candidate.
        batch_size: B in the paper (default 12).
        w_r:        diversity weight for this round.

    Returns:
        list of selected IDs (length ≤ batch_size).
    """
    N = len(acq_scores)
    a_norm = minmax_normalize(acq_scores)
    selected_idx: List[int] = []
    remaining = set(range(N))

    while len(selected_idx) < batch_size and remaining:
        rem = np.fromiter(remaining, dtype=int)
        if not selected_idx:
            d = np.ones(len(rem))            # first pick: any candidate is "fully diverse"
        else:
            sel_fps = fps[selected_idx]
            d = diversity_score(fps[rem], sel_fps)
        d_norm = minmax_normalize(d) if len(rem) > 1 else np.ones_like(d)
        s = composite_score(a_norm[rem], d_norm, w_r)
        pick = int(rem[np.argmax(s)])
        selected_idx.append(pick)
        remaining.remove(pick)

    return [pool_ids[i] for i in selected_idx]


# ──────────────────────────────────────────────────────────────────────
# Subspace restriction & synthesizability filter
# ──────────────────────────────────────────────────────────────────────

def restrict_to_phosphine(
    df: pd.DataFrame,
    initiator_col: Optional[str] = None,
) -> pd.DataFrame:
    """Round-3 subspace: keep candidates whose Initiator contains phosphorus.

    Heuristic match for ``P(`` / ``[P`` / ``P+`` patterns in the Initiator SMILES.
    """
    df = df.copy()
    df.rename(columns={c: normalize_column_name(c) for c in df.columns}, inplace=True)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    if initiator_col is None:
        for c in df.columns:
            cnorm = c.lower()
            if cnorm.startswith("initiator") and "anhydride" not in cnorm and "percentage" not in cnorm:
                initiator_col = c
                break
    if initiator_col is None or initiator_col not in df.columns:
        raise ValueError("Initiator column not found")

    pattern = r"P\(|\[P|P\+"
    mask = df[initiator_col].astype(str).str.contains(pattern, regex=True, na=False)
    return df[mask].reset_index(drop=True)


def synth_filter_topk(
    ranked_ids: Sequence,
    feasible_set: Optional[Sequence] = None,
    K: int = 24,
    batch_size: int = 12,
) -> List:
    """Human-in-the-loop synthesizability filter.

    Takes the top-K ranked IDs, keeps those in ``feasible_set`` (expert reviewed),
    returns the first ``batch_size`` survivors. When ``feasible_set`` is None, all
    top-K are assumed feasible (placeholder behavior).
    """
    top_k = list(ranked_ids[:K])
    if feasible_set is None:
        return top_k[:batch_size]
    feasible = set(feasible_set)
    return [x for x in top_k if x in feasible][:batch_size]


# ──────────────────────────────────────────────────────────────────────
# End-to-end round selection
# ──────────────────────────────────────────────────────────────────────

def select_round_batch(
    pool_df: pd.DataFrame,
    pred_mu: np.ndarray,
    pred_sigma: Optional[np.ndarray] = None,
    y_best: Optional[float] = None,
    round_idx: int = 1,
    batch_size: int = 12,
    acquisition: str = "thompson",
    fp_radius: int = 2,
    fp_bits: int = 2048,
    cache_dir: str = ".cache_features",
    seed: Optional[int] = None,
) -> List:
    """Run one round of acquisition + diversity-aware batch selection.

    Args:
        pool_df:     candidate dataFrame (must include SMILES + Training data columns).
        pred_mu:     mean prediction per row of ``pool_df``.
        pred_sigma:  σ per row (required for EI/UCB; for TS, falls back to
                     0.1 * std(μ) if omitted).
        y_best:      best observed value (for EI).
        round_idx:   1/2/3 - selects diversity weight w_r from PHASE_WEIGHTS.
        acquisition: "thompson" | "ei" | "ucb".

    Returns:
        list of selected molecule IDs.
    """
    df = pool_df.copy()
    df.rename(columns={c: normalize_column_name(c) for c in df.columns}, inplace=True)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    smiles_col = find_smiles_column(list(df.columns))
    smiles = df[smiles_col].astype(str).tolist()
    pool_ids = df["Training data"].astype(int).tolist()

    fps, idx_kept = smiles_to_morgan_bits(
        smiles, radius=fp_radius, n_bits=fp_bits, cache_dir=cache_dir
    )

    mu_aligned = np.asarray(pred_mu)[idx_kept]
    if pred_sigma is None:
        sigma_aligned = np.full_like(mu_aligned, mu_aligned.std() * 0.1 + 1e-3)
    else:
        sigma_aligned = np.asarray(pred_sigma)[idx_kept]
    aligned_ids = [pool_ids[i] for i in idx_kept]

    if acquisition == "thompson":
        acq = acquisition_thompson_sample(mu_aligned, sigma_aligned, seed=seed)
    elif acquisition == "ei":
        if y_best is None:
            raise ValueError("y_best required for EI")
        acq = acquisition_ei(mu_aligned, sigma_aligned, y_best)
    elif acquisition == "ucb":
        acq = acquisition_ucb(mu_aligned, sigma_aligned)
    else:
        raise ValueError(f"unknown acquisition: {acquisition}")

    w_r = PHASE_WEIGHTS.get(round_idx, 0.25)
    return greedy_batch_select(
        acq_scores=acq,
        fps=fps,
        pool_ids=aligned_ids,
        batch_size=batch_size,
        w_r=w_r,
    )
