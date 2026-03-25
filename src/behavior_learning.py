from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import logging

import numpy as np
import pandas as pd

from src.feature_engineering import add_market_features
from src.fvg_engine import FVGBook
from src.liquidity_map import LiquidityMap
from src.state_engine import TransitionModel, infer_market_states, learn_transition_model


Key3 = Tuple[str, str, str]


@dataclass
class LearnedBehavior:
    transition_model: TransitionModel
    session_profiles: Dict[str, Dict[str, float]]
    sweep_stats: Dict[str, float]
    fvg_stats: Dict[str, float]
    volatility_stats: Dict[str, float]
    conditional_returns: Dict[Key3, Dict[str, float]]
    conditional_ranges: Dict[Key3, Dict[str, float]]
    conditional_wicks: Dict[Key3, Dict[str, float]]
    market_profile: Dict[str, float]
    historical_contexts: pd.DataFrame
    context_index: Dict[str, list[int]]
    feature_df: pd.DataFrame

    def find_similar_contexts(self, current_context: Dict[str, float], state: str, n_neighbors: int = 10) -> list[int]:
        if state not in self.context_index:
            return []
        candidates = self.context_index[state]
        if len(candidates) <= n_neighbors:
            return candidates

        features = [
            "realized_vol",
            "distance_to_nearest_buy_liquidity",
            "distance_to_nearest_sell_liquidity",
            "compression_score",
            "expansion_score",
            "recent_sweep_strength",
        ]
        distances: list[tuple[float, int]] = []
        for idx in candidates:
            hist_row = self.historical_contexts.iloc[idx]
            d = 0.0
            for f in features:
                d += (float(current_context.get(f, 0.0)) - float(hist_row.get(f, 0.0))) ** 2
            distances.append((d, idx))
        distances.sort(key=lambda x: x[0])
        return [idx for _, idx in distances[:n_neighbors]]


def _continuation_rate(series: pd.Series, horizon: int = 3) -> float:
    if len(series) < horizon + 1:
        return 0.0
    ok = 0
    n = 0
    for i in range(len(series) - horizon):
        direction = np.sign(series.iloc[i])
        if direction == 0:
            continue
        future = series.iloc[i + 1 : i + horizon + 1]
        if np.sign(future.sum()) == direction:
            ok += 1
        n += 1
    return float(ok / n) if n else 0.0


def learn_behavior(df: pd.DataFrame) -> LearnedBehavior:
    logger = logging.getLogger(__name__)
    feat = add_market_features(df)

    # Strictement incrémental: query avant update (pas d'info future)
    lmap = LiquidityMap()
    fvg = FVGBook()

    feat = feat.reset_index(drop=True)
    rolling_high = feat["high"].rolling(30, min_periods=5).max().reset_index(drop=True)
    rolling_low = feat["low"].rolling(30, min_periods=5).min().reset_index(drop=True)
    work = feat.copy()
    work["rolling_high"] = rolling_high
    work["rolling_low"] = rolling_low
    initial_len = len(work)
    work_valid = work.dropna(subset=["rolling_high", "rolling_low"])
    dropped_len = initial_len - len(work_valid)
    logger.info("[LEARN] feature rows initial=%s aligned=%s dropped=%s", initial_len, len(work_valid), dropped_len)

    liq_by_idx: Dict[int, float] = {}
    dist_buy_by_idx: Dict[int, float] = {}
    dist_sell_by_idx: Dict[int, float] = {}
    sweep_flag_by_idx: Dict[int, int] = {}
    sweep_side_by_idx: Dict[int, int] = {}
    sweep_strength_by_idx: Dict[int, float] = {}
    d_to_fvg_by_idx: Dict[int, float] = {}
    open_fvg_by_idx: Dict[int, int] = {}
    fvg_ctx_by_idx: Dict[int, int] = {}

    for idx, row in work_valid.iterrows():
        local_scale = float(feat["range"].iloc[max(0, idx - 30) : idx + 1].mean() or 1.0)

        liq = lmap.nearest_distances(float(row["close"]), local_scale=local_scale)
        sw = lmap.detect_sweep(idx, float(row["high"]), float(row["low"]), float(row["close"]))

        d = fvg.nearest_open_distance(float(row["close"]))
        dist_buy_by_idx[idx] = liq["distance_to_nearest_buy_liquidity"]
        dist_sell_by_idx[idx] = liq["distance_to_nearest_sell_liquidity"]
        liq_by_idx[idx] = liq["nearest_liquidity_strength"]
        sweep_flag_by_idx[idx] = int(sw["recent_sweep_flag"])
        sweep_side_by_idx[idx] = int(sw["recent_sweep_side"])
        sweep_strength_by_idx[idx] = float(sw["recent_sweep_strength"])
        d_to_fvg_by_idx[idx] = float(d / max(local_scale, 1e-8)) if not np.isnan(d) else np.nan
        open_fvg_by_idx[idx] = int(fvg.open_count_nearby(float(row["close"]), threshold=local_scale * 1.5))
        fvg_ctx_by_idx[idx] = int(not np.isnan(d))

        # Update maps only after extracting features at t
        lmap.ingest_feature_row(idx, row, float(row["rolling_high"]), float(row["rolling_low"]))
        lmap.age_and_decay(idx)
        fvg.detect_new(work, idx, state=row.get("trend_context", ""), session=row.get("session_name", ""))
        fvg.update_fill(idx, float(row["high"]), float(row["low"]))

    feat["distance_to_nearest_buy_liquidity"] = pd.Series(dist_buy_by_idx).reindex(feat.index).fillna(3.0)
    feat["distance_to_nearest_sell_liquidity"] = pd.Series(dist_sell_by_idx).reindex(feat.index).fillna(3.0)
    feat["nearest_liquidity_strength"] = pd.Series(liq_by_idx).reindex(feat.index).fillna(0.0)
    feat["recent_sweep_flag"] = pd.Series(sweep_flag_by_idx).reindex(feat.index).fillna(0).astype(int)
    feat["recent_sweep_side"] = pd.Series(sweep_side_by_idx).reindex(feat.index).fillna(0).astype(int)
    feat["recent_sweep_strength"] = pd.Series(sweep_strength_by_idx).reindex(feat.index).fillna(0.0)
    feat["distance_to_nearest_open_fvg"] = pd.Series(d_to_fvg_by_idx).reindex(feat.index).fillna(3.0)
    feat["open_fvg_count_nearby"] = pd.Series(open_fvg_by_idx).reindex(feat.index).fillna(0).astype(int)
    feat["fvg_context"] = pd.Series(fvg_ctx_by_idx).reindex(feat.index).fillna(0).astype(int)

    states = infer_market_states(feat)
    tm = learn_transition_model(states)

    cond_returns: Dict[Key3, Dict[str, float]] = {}
    cond_ranges: Dict[Key3, Dict[str, float]] = {}
    cond_wicks: Dict[Key3, Dict[str, float]] = {}

    for state in states["state"].unique():
        for vol in states["vol_regime_bucket"].unique():
            for sess in states["session_name"].unique():
                sub = states[(states["state"] == state) & (states["vol_regime_bucket"] == vol) & (states["session_name"] == sess)]
                key = (str(state), str(vol), str(sess))
                base = states[states["state"] == state]
                use = sub if not sub.empty else base
                cond_returns[key] = {
                    "mu": float(use["log_return"].mean()),
                    "sigma": float(use["log_return"].std(ddof=0) + 1e-8),
                    "cont_1": _continuation_rate(use["log_return"], 1),
                    "cont_3": _continuation_rate(use["log_return"], 3),
                    "cont_5": _continuation_rate(use["log_return"], 5),
                }
                cond_ranges[key] = {
                    "mu": float(use["range"].mean()),
                    "sigma": float(use["range"].std(ddof=0) + 1e-8),
                }
                cond_wicks[key] = {
                    "upper_mu": float(use["wick_upper"].mean()),
                    "lower_mu": float(use["wick_lower"].mean()),
                    "upper_sigma": float(use["wick_upper"].std(ddof=0) + 1e-8),
                    "lower_sigma": float(use["wick_lower"].std(ddof=0) + 1e-8),
                }

    sweep_mask_buy = states["recent_sweep_side"] == 1
    sweep_mask_sell = states["recent_sweep_side"] == -1
    sweep_stats = {
        "sweep_frequency": float(states["recent_sweep_flag"].mean()),
        "continuation_after_sweep_buy": _continuation_rate(states.loc[sweep_mask_buy, "log_return"], 3),
        "reversal_after_sweep_buy": 1.0 - _continuation_rate(states.loc[sweep_mask_buy, "log_return"], 3),
        "continuation_after_sweep_sell": _continuation_rate(states.loc[sweep_mask_sell, "log_return"], 3),
        "reversal_after_sweep_sell": 1.0 - _continuation_rate(states.loc[sweep_mask_sell, "log_return"], 3),
        "overshoot_mean": float(states.loc[states["recent_sweep_flag"] == 1, "recent_sweep_strength"].mean() if (states["recent_sweep_flag"] == 1).any() else 0.0),
    }

    session_profiles = (
        states.groupby("session_name")
        .agg(realized_vol_mean=("realized_vol", "mean"), range_mean=("range", "mean"), sweep_freq=("recent_sweep_flag", "mean"), breakout_freq=("breakout_up", lambda x: float(x.mean())), continuation_rate=("log_return", lambda x: _continuation_rate(x, 3)))
        .to_dict(orient="index")
    )

    volatility_stats = {
        "log_return_mu": float(states["log_return"].mean()),
        "log_return_sigma": float(states["log_return"].std(ddof=0)),
        "range_mean": float(states["range"].mean()),
        "body_ratio_mean": float(states["body_ratio"].mean()),
        "wick_upper_mean": float(states["wick_upper"].mean()),
        "wick_lower_mean": float(states["wick_lower"].mean()),
    }

    market_profile = {
        "expansion_freq": float((states["expansion_score"] > 1.25).mean()),
        "compression_freq": float((states["compression_score"] > states["compression_score"].quantile(0.7)).mean()),
        "breakout_accepted_freq": float(((states["breakout_up"]) | (states["breakout_down"])).mean()),
        "breakout_rejected_freq": float(states["breakout_rejected"].mean()),
        "equal_high_freq": float(states["equal_high"].mean()),
        "equal_low_freq": float(states["equal_low"].mean()),
        "near_fvg_freq": float((states["distance_to_nearest_open_fvg"] < 1.25).mean()),
        "near_liquidity_freq": float((states[["distance_to_nearest_buy_liquidity", "distance_to_nearest_sell_liquidity"]].min(axis=1) < 1.0).mean()),
    }

    context_features = [
        "realized_vol",
        "distance_to_nearest_buy_liquidity",
        "distance_to_nearest_sell_liquidity",
        "compression_score",
        "expansion_score",
        "recent_sweep_strength",
        "log_return",
    ]
    historical_contexts = states[context_features + ["state"]].copy().reset_index(drop=True)
    context_index: Dict[str, list[int]] = {}
    for state in historical_contexts["state"].unique():
        context_index[str(state)] = historical_contexts[historical_contexts["state"] == state].index.tolist()

    return LearnedBehavior(
        tm,
        session_profiles,
        sweep_stats,
        fvg.stats(total_bars=len(states)),
        volatility_stats,
        cond_returns,
        cond_ranges,
        cond_wicks,
        market_profile,
        historical_contexts,
        context_index,
        states,
    )
