from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # --- Macro guardrails ---
    global_gold_reference_daily_notional: float = 3.27e11

    # --- Order impact ---
    impact_activation_threshold: float = 5e-6
    impact_y: float = 0.55
    impact_clip_abs: float = 0.02
    spread_participation_multiplier: float = 0.45

    # --- Residual pressure ---
    pending_impact_decay: float = 0.5

    # --- Liquidity / capacity ---
    min_liquidity_capacity: float = 5e5
    min_local_executable_notional: float = 2e5
    liquidity_strength_to_notional: float = 8e6
    local_volume_weight: float = 0.70
    liquidity_capacity_weight: float = 0.30

    # --- Sweep / cascade ---
    sweep_trigger_threshold: float = 0.55
    cascade_trigger_threshold: float = 0.45
    cascade_cap: float = 0.004
    cascade_multiplier: float = 0.002

    # --- FVG rebalance damping ---
    fvg_rebalance_distance_threshold: float = 1.25
    fvg_rebalance_damping: float = 0.85
