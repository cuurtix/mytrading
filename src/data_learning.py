from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.behavior_learning import LearnedBehavior, learn_behavior
from src.data_ingestion import IngestionConfig, IngestionReport, detect_timeframe_seconds, merge_compatible_datasets, scan_data_sources


@dataclass
class CalibrationBundle:
    merged_df: pd.DataFrame
    learned: LearnedBehavior
    logs: List[str]
    timeframe_seconds: int
    timeframe_label: str


def learn_from_report(report: IngestionReport, cfg: IngestionConfig | None = None) -> CalibrationBundle:
    cfg = cfg or IngestionConfig()
    if not report.datasets:
        raise ValueError("Aucun dataset exploitable")

    merged = merge_compatible_datasets(report.datasets, report=report, mode=cfg.selection_strategy, max_total_rows=cfg.max_total_rows)
    if merged.empty:
        raise ValueError("Aucun dataset compatible (timeframe)")

    tf_seconds, tf_label = detect_timeframe_seconds(merged["datetime"])
    learned = learn_behavior(merged)

    logs = list(report.logs)
    logs.append(f"timeframe détecté: {tf_label} ({tf_seconds}s)")
    logs.append("calibration terminée")

    return CalibrationBundle(
        merged_df=merged,
        learned=learned,
        logs=logs,
        timeframe_seconds=tf_seconds,
        timeframe_label=tf_label,
    )


def learn_from_root(root_path: str, cfg: IngestionConfig | None = None) -> CalibrationBundle:
    cfg = cfg or IngestionConfig()
    report = scan_data_sources(root_path, cfg=cfg)
    return learn_from_report(report, cfg=cfg)


def bundle_summary(bundle: CalibrationBundle) -> Dict[str, float]:
    return {
        "timeframe_seconds": bundle.timeframe_seconds,
        "rows": int(len(bundle.merged_df)),
        "sweep_frequency": bundle.learned.sweep_stats["sweep_frequency"],
        "fvg_fill_complete_rate": bundle.learned.fvg_stats.get("fvg_fill_complete_rate", 0.0),
        "log_return_sigma": bundle.learned.volatility_stats["log_return_sigma"],
    }
