from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

import pandas as pd
import numpy as np

from src.behavior_learning import LearnedBehavior, learn_behavior
from src.data_ingestion import IngestionConfig, IngestionReport, detect_timeframe_seconds, scan_data_sources


@dataclass
class CalibrationBundle:
    merged_df: pd.DataFrame
    learned: LearnedBehavior
    logs: List[str]
    timeframe_seconds: int
    timeframe_label: str
    htf_context: Dict[str, Dict[str, float]]
    htf_contexts: Dict[str, Dict[str, float]]


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
    htf_context: Dict[str, Dict[str, float]] = {}
    for tf, ds_list in groups.items():
        total_rows = sum(len(x.dataframe) for x in ds_list)
        if total_rows < 20:
            continue
        df_last = max(ds_list, key=lambda x: x.dataframe["datetime"].max()).dataframe
        last = float(df_last["close"].iloc[-1])
        mean_price = float(np.mean([float(ds.dataframe["close"].mean()) for ds in ds_list]))
        high_max = float(max(float(ds.dataframe["high"].max()) for ds in ds_list))
        low_min = float(min(float(ds.dataframe["low"].min()) for ds in ds_list))
        vol_mean = float(np.mean([float(ds.dataframe["close"].pct_change().std()) for ds in ds_list]))
        range_mean = float(np.mean([float((ds.dataframe["high"] - ds.dataframe["low"]).mean()) for ds in ds_list]))
        htf_context[tf] = {
            "bias": float(np.sign(last - mean_price)),
            "vol": vol_mean,
            "dist_high": float((high_max - last) / max(last, 1e-8)),
            "dist_low": float((last - low_min) / max(last, 1e-8)),
            "range": range_mean,
        }
    merged_list: List[pd.DataFrame] = []
    for ds in best_group:
        seg = ds.dataframe.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
        if len(seg) > cfg.max_total_rows:
            seg = seg.tail(cfg.max_total_rows).reset_index(drop=True)
        merged_list.append(seg)
    report.debug.update({"datasets_retained": len(best_group), "rows_merged": int(sum(len(x) for x in merged_list)), "timeframe": best_tf})
    if not merged_list:
        report.debug.update({"phase": "learn_behavior", "ok": False, "error": "merged_dataset_empty"})
        raise ValueError("Aucun dataset compatible (timeframe)")

    min_learn_rows = min(cfg.min_rows_per_dataset, 20)
    learn_pairs = [(seg, learn_behavior(seg)) for seg in merged_list if len(seg) >= min_learn_rows]
    if not learn_pairs:
        report.debug.update({"phase": "learn_behavior", "ok": False, "error": "no_segment_large_enough"})
        raise ValueError("Aucun segment suffisant pour apprentissage")
    merged, learned = max(learn_pairs, key=lambda x: len(x[0]))
    tf_seconds, tf_label = detect_timeframe_seconds(merged["datetime"])

    logs = list(report.logs)
    logs.append(f"timeframe détecté: {tf_label} ({tf_seconds}s)")
    logs.append("calibration terminée")
    report.debug.update({"phase": "learn_behavior", "ok": True, "error": None, "timeframe": tf_label})

    return CalibrationBundle(
        merged_df=merged,
        learned=learned,
        logs=logs,
        timeframe_seconds=tf_seconds,
        timeframe_label=tf_label,
        htf_context=htf_context,
        htf_contexts=htf_context,
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
