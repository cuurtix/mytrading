from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.feature_engineering import add_market_features
from src.fvg_engine import FVGBook
from src.liquidity_map import build_liquidity_map
from src.state_engine import TransitionModel, infer_market_states, learn_transition_model


@dataclass
class LearnedBehavior:
    transition_model: TransitionModel
    session_profiles: Dict[str, Dict[str, float]]
    sweep_stats: Dict[str, float]
    fvg_stats: Dict[str, float]
    volatility_stats: Dict[str, float]
    conditional_returns: Dict[str, Dict[str, float]]
    conditional_ranges: Dict[str, Dict[str, float]]
    conditional_wicks: Dict[str, Dict[str, float]]
    feature_df: pd.DataFrame


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
    feat = add_market_features(df)

    # Carte de liquidité persistante
    lmap = build_liquidity_map(feat)
    dist_buy, dist_sell, liq_strength = [], [], []
    sweep_flag, sweep_side, sweep_strength = [], [], []

    for i, row in feat.iterrows():
        lm_stats = lmap.nearest_distances(float(row["close"]))
        sw = lmap.detect_sweep(i, float(row["high"]), float(row["low"]), float(row["close"]))
        lmap.age_and_decay(i)

        dist_buy.append(lm_stats["distance_to_nearest_buy_liquidity"])
        dist_sell.append(lm_stats["distance_to_nearest_sell_liquidity"])
        liq_strength.append(lm_stats["nearest_liquidity_strength"])
        sweep_flag.append(sw["recent_sweep_flag"])
        sweep_side.append(sw["recent_sweep_side"])
        sweep_strength.append(sw["recent_sweep_strength"])

    feat["distance_to_nearest_buy_liquidity"] = pd.Series(dist_buy).fillna(feat["range"].mean())
    feat["distance_to_nearest_sell_liquidity"] = pd.Series(dist_sell).fillna(feat["range"].mean())
    feat["nearest_liquidity_strength"] = liq_strength
    feat["recent_sweep_flag"] = sweep_flag
    feat["recent_sweep_side"] = sweep_side
    feat["recent_sweep_strength"] = sweep_strength

    # Cycle de vie FVG
    fvg = FVGBook()
    d_to_fvg = []
    open_fvg_count = []
    fvg_context = []
    for i, row in feat.iterrows():
        fvg.detect_new(feat, i, state=row.get("trend_context", ""), session=row.get("session_name", ""))
        fvg.update_fill(i, float(row["high"]), float(row["low"]))
        d = fvg.nearest_open_distance(float(row["close"]))
        d_to_fvg.append(d)
        open_fvg_count.append(fvg.open_count_nearby(float(row["close"]), threshold=float(feat["range"].mean() * 1.5)))
        fvg_context.append(int(not np.isnan(d)))

    feat["distance_to_nearest_open_fvg"] = pd.Series(d_to_fvg).fillna(feat["range"].mean() * 3)
    feat["open_fvg_count_nearby"] = open_fvg_count
    feat["fvg_context"] = fvg_context

    states = infer_market_states(feat)
    tm = learn_transition_model(states)

    # Statistiques conditionnelles riches
    cond_returns, cond_ranges, cond_wicks = {}, {}, {}
    for state in states["state"].unique():
        sub = states[states["state"] == state]
        cond_returns[state] = {
            "mu": float(sub["log_return"].mean()),
            "sigma": float(sub["log_return"].std(ddof=0) + 1e-8),
            "cont_1": _continuation_rate(sub["log_return"], 1),
            "cont_3": _continuation_rate(sub["log_return"], 3),
            "cont_5": _continuation_rate(sub["log_return"], 5),
        }
        cond_ranges[state] = {
            "mu": float(sub["range"].mean()),
            "sigma": float(sub["range"].std(ddof=0) + 1e-8),
        }
        cond_wicks[state] = {
            "upper_mu": float(sub["wick_upper"].mean()),
            "lower_mu": float(sub["wick_lower"].mean()),
            "upper_sigma": float(sub["wick_upper"].std(ddof=0) + 1e-8),
            "lower_sigma": float(sub["wick_lower"].std(ddof=0) + 1e-8),
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

    fvg_stats = fvg.stats()

    session_profiles = (
        states.groupby("session_name")
        .agg(
            realized_vol_mean=("realized_vol", "mean"),
            range_mean=("range", "mean"),
            sweep_freq=("recent_sweep_flag", "mean"),
            breakout_freq=("breakout_up", lambda x: float(x.mean())),
            continuation_rate=("log_return", lambda x: _continuation_rate(x, 3)),
            reversal_rate=("log_return", lambda x: 1.0 - _continuation_rate(x, 3)),
            fvg_creation_rate=("fvg_context", "mean"),
        )
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

    return LearnedBehavior(
        transition_model=tm,
        session_profiles=session_profiles,
        sweep_stats=sweep_stats,
        fvg_stats=fvg_stats,
        volatility_stats=volatility_stats,
        conditional_returns=cond_returns,
        conditional_ranges=cond_ranges,
        conditional_wicks=cond_wicks,
        feature_df=states,
    )
