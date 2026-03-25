from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

MARKET_STATES = [
    "RANGE",
    "TREND_UP",
    "TREND_DOWN",
    "EXPANSION_UP",
    "EXPANSION_DOWN",
    "POST_SWEEP_REVERSAL",
    "BREAKOUT_ACCEPTED",
    "BREAKOUT_REJECTED",
    "REBALANCING_TO_FVG",
    "HIGH_VOLATILITY_PANIC",
    "LOW_VOLATILITY_COMPRESSION",
]


@dataclass
class TransitionModel:
    transition_probs: Dict[Tuple[str, str, str, str, str, str], Dict[str, float]]
    state_return_stats: Dict[Tuple[str, str, str], Dict[str, float]]
    state_range_stats: Dict[Tuple[str, str, str], Dict[str, float]]


def _bucket_liquidity(x_rel: float) -> str:
    if pd.isna(x_rel):
        return "UNK"
    if x_rel < 0.75:
        return "NEAR"
    if x_rel < 2.5:
        return "MID"
    return "FAR"


def _breakout_ctx(row: pd.Series) -> str:
    if row.get("breakout_rejected", False):
        return "REJECTED"
    if row.get("breakout_up", False) or row.get("breakout_down", False):
        return "ACCEPTED"
    return "NONE"


def infer_market_states(features_df: pd.DataFrame) -> pd.DataFrame:
    out = features_df.copy()
    out["state"] = "RANGE"
    compression_q70 = float(out["compression_score"].quantile(0.7))

    for i, row in out.iterrows():
        state = "RANGE"
        if row["vol_regime_bucket"] == "HIGH" and row["expansion_score"] > 1.7:
            state = "HIGH_VOLATILITY_PANIC"
        elif row["vol_regime_bucket"] == "LOW" and row["compression_score"] > compression_q70:
            state = "LOW_VOLATILITY_COMPRESSION"
        elif row.get("recent_sweep_flag", 0) and row.get("breakout_rejected", False):
            state = "POST_SWEEP_REVERSAL"
        elif row.get("breakout_rejected", False):
            state = "BREAKOUT_REJECTED"
        elif row.get("breakout_up", False) or row.get("breakout_down", False):
            state = "BREAKOUT_ACCEPTED"
        elif row.get("fvg_context", 0) == 1 and row.get("distance_to_nearest_open_fvg", np.nan) < 1.25:
            state = "REBALANCING_TO_FVG"
        elif row["trend_context"] == "TREND_UP" and row["expansion_score"] > 1.3:
            state = "EXPANSION_UP"
        elif row["trend_context"] == "TREND_DOWN" and row["expansion_score"] > 1.3:
            state = "EXPANSION_DOWN"
        elif row["trend_context"] == "TREND_UP":
            state = "TREND_UP"
        elif row["trend_context"] == "TREND_DOWN":
            state = "TREND_DOWN"
        out.at[i, "state"] = state

    return out


def build_transition_key(state: str, row: pd.Series) -> Tuple[str, str, str, str, str, str]:
    vol_bucket = str(row.get("vol_regime_bucket", "UNK"))
    liq_rel = min(float(row.get("distance_to_nearest_buy_liquidity", np.nan)), float(row.get("distance_to_nearest_sell_liquidity", np.nan)))
    liq_bucket = _bucket_liquidity(liq_rel)
    breakout_bucket = _breakout_ctx(row)
    sweep_bucket = "SWEEP" if row.get("recent_sweep_flag", 0) else "NO_SWEEP"
    session = str(row.get("session_name", "UNK"))
    return state, vol_bucket, liq_bucket, breakout_bucket, sweep_bucket, session


def learn_transition_model(state_df: pd.DataFrame) -> TransitionModel:
    keys = []
    for _, row in state_df.iterrows():
        keys.append(build_transition_key(row["state"], row))
    next_states = state_df["state"].shift(-1)

    matrix: Dict[Tuple[str, str, str, str, str, str], Dict[str, float]] = {}
    for i, key in enumerate(keys[:-1]):
        nxt = next_states.iloc[i]
        if pd.isna(nxt):
            continue
        matrix.setdefault(key, {})
        matrix[key][str(nxt)] = matrix[key].get(str(nxt), 0.0) + 1.0

    for key, counts in matrix.items():
        total = sum(counts.values())
        matrix[key] = {k: float(v / total) for k, v in counts.items()}

    ret_stats: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    rng_stats: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for state in MARKET_STATES:
        for vol_bucket in ["LOW", "NORMAL", "HIGH", "UNK"]:
            for session in ["ASIA", "LONDON_OPEN", "NEW_YORK", "LATE_SESSION", "UNK"]:
                sub = state_df[(state_df["state"] == state) & (state_df["vol_regime_bucket"] == vol_bucket) & (state_df["session_name"] == session)]
                key = (state, vol_bucket, session)
                if sub.empty:
                    ret_stats[key] = {"mu": float(state_df["log_return"].mean()), "sigma": float(state_df["log_return"].std(ddof=0) + 1e-8)}
                    rng_stats[key] = {"mu": float(state_df["range"].mean()), "sigma": float(state_df["range"].std(ddof=0) + 1e-8)}
                else:
                    ret_stats[key] = {"mu": float(sub["log_return"].mean()), "sigma": float(sub["log_return"].std(ddof=0) + 1e-8)}
                    rng_stats[key] = {"mu": float(sub["range"].mean()), "sigma": float(sub["range"].std(ddof=0) + 1e-8)}

    return TransitionModel(matrix, ret_stats, rng_stats)


def sample_next_state(current_state: str, row_ctx: pd.Series, transition_probs: Dict[Tuple[str, str, str, str, str, str], Dict[str, float]], rng: np.random.Generator) -> str:
    key = build_transition_key(current_state, row_ctx)
    row = transition_probs.get(key)
    if row is None:
        # fallback coarse on state only
        candidates = {k2: v for k, d in transition_probs.items() if k[0] == current_state for k2, v in d.items()}
        if not candidates:
            return "RANGE"
        total = sum(candidates.values())
        states = list(candidates.keys())
        probs = np.array([candidates[s] / total for s in states], dtype=float)
        return str(rng.choice(states, p=probs))

    states = list(row.keys())
    probs = np.array([row[s] for s in states], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(states, p=probs))
