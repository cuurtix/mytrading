from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

import pandas as pd

from src.behavior_learning import LearnedBehavior, learn_behavior
from src.data_ingestion import IngestionConfig, IngestionReport, detect_timeframe_seconds, scan_data_sources


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

    groups = defaultdict(list)
    for ds in report.datasets:
        groups[ds.timeframe_label].append(ds)
    if not groups:
        report.debug.update({"phase": "learn_behavior", "datasets_retained": 0, "rows_merged": 0, "timeframe": "unknown", "ok": False, "error": "no_valid_dataset_retained"})
        raise ValueError("Aucun dataset compatible (timeframe)")

    best_tf, best_group = max(groups.items(), key=lambda kv: sum(len(x.dataframe) for x in kv[1]))
    merged = (
        pd.concat([x.dataframe for x in best_group], ignore_index=True)
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )
    if len(merged) > cfg.max_total_rows:
        merged = merged.tail(cfg.max_total_rows).reset_index(drop=True)
    report.debug.update({"datasets_retained": len(best_group), "rows_merged": len(merged), "timeframe": best_tf})
    if merged.empty:
        report.debug.update({"phase": "learn_behavior", "ok": False, "error": "merged_dataset_empty"})
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
