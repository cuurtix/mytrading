from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import io
from pathlib import Path
from typing import Any, Dict, List, Tuple
import difflib
import json
import zipfile

import numpy as np
import pandas as pd

TIMESTAMP_ALIASES = {"timestamp", "time", "datetime", "date", "open_time", "close_time"}
OPEN_ALIASES = {"open", "o"}
HIGH_ALIASES = {"high", "h"}
LOW_ALIASES = {"low", "l"}
CLOSE_ALIASES = {"close", "c"}
VOLUME_ALIASES = {"volume", "basevolume", "tick_volume", "real_volume", "quotevolume", "usdtvolume"}
VOLUME_PRIORITY = ["volume", "basevolume", "quotevolume", "usdtvolume", "tick_volume", "real_volume"]
SUPPORTED_EXTS = {".zip", ".csv", ".xlsx", ".xls"}
COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "date", "datetime", "open_time"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "last"],
    "volume": ["volume", "vol", "v"],
}
VERBOSE_INGESTION = True


@dataclass(frozen=True)
class IngestionConfig:
    max_files: int = 64
    max_total_rows: int = 300_000
    max_rows_per_dataset: int = 80_000
    min_rows_per_dataset: int = 100
    selection_strategy: str = "balanced"  # balanced | most_recent | largest


@dataclass
class NormalizedDataset:
    name: str
    source_type: str
    sheet_name: str | None
    timeframe_seconds: int
    timeframe_label: str
    mapped_columns: Dict[str, str]
    dataframe: pd.DataFrame


@dataclass
class IngestionReport:
    datasets: List[NormalizedDataset] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    root_scanned: str = ""
    total_files_found: int = 0
    supported_files_found: int = 0
    retained_files: List[str] = field(default_factory=list)
    ignored_files: List[Dict[str, str]] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def log(self, message: str) -> None:
        self.logs.append(message)


def make_ingestion_debug() -> Dict[str, Any]:
    return {
        "files_detected": 0,
        "datasets_retained": 0,
        "rows_merged": 0,
        "retained_files": [],
        "ignored_files": [],
        "ingestion_logs": [],
        "timeframe": "unknown",
        "phase": "scan",
        "ok": False,
        "error": None,
    }


def log_ingestion(debug: Dict[str, Any], level: str, file_path: str | Path, message: str, **extra: Any) -> None:
    entry = {"level": level, "file": str(file_path), "message": message, **extra}
    debug["ingestion_logs"].append(entry)
    if level in ("warning", "error"):
        debug["ignored_files"].append({"file": str(file_path), "reason": message, **extra})


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _best_match(col: str, aliases: set[str]) -> bool:
    if col in aliases:
        return True
    return bool(difflib.get_close_matches(col, list(aliases), n=1, cutoff=0.8))


def _map_columns(df: pd.DataFrame) -> Dict[str, str]:
    normalized = {_normalize_col(c): c for c in df.columns}
    cols = list(normalized.keys())

    def pick(aliases: set[str]) -> str | None:
        for c in cols:
            if _best_match(c, aliases):
                return c
        return None

    ts = pick(TIMESTAMP_ALIASES)
    o = pick(OPEN_ALIASES)
    h = pick(HIGH_ALIASES)
    l = pick(LOW_ALIASES)
    c = pick(CLOSE_ALIASES)
    if not all([ts, o, h, l, c]):
        raise ValueError("Colonnes OHLC/timestamp introuvables après mapping robuste")

    volumes = [v for v in VOLUME_PRIORITY if v in cols]
    if not volumes:
        volumes = [c_ for c_ in cols if _best_match(c_, VOLUME_ALIASES)]

    mapping = {
        "timestamp": normalized[ts],
        "open": normalized[o],
        "high": normalized[h],
        "low": normalized[l],
        "close": normalized[c],
    }
    for vol in volumes:
        mapping[vol] = normalized[vol]

    primary = next((v for v in VOLUME_PRIORITY if v in mapping), volumes[0] if volumes else None)
    if primary is not None:
        mapping["volume"] = mapping[primary]
    return mapping


def _detect_header_and_read_excel(io_obj, sheet_name: str) -> pd.DataFrame:
    for header_row in range(0, 8):
        try:
            if hasattr(io_obj, "seek"):
                io_obj.seek(0)
            df = pd.read_excel(io_obj, sheet_name=sheet_name, header=header_row)
            _map_columns(df)
            return df
        except Exception:
            continue
    raise ValueError(f"Impossible de détecter un en-tête valide pour {sheet_name}")


def _parse_timestamp(series: pd.Series) -> Tuple[pd.Series, str]:
    raw = pd.to_numeric(series, errors="coerce")
    if raw.notna().mean() > 0.95:
        median = raw.dropna().median()
        if median > 1e12:
            return pd.to_datetime(raw, unit="ms", utc=True, errors="coerce"), "epoch_ms"
        if median > 1e9:
            return pd.to_datetime(raw, unit="s", utc=True, errors="coerce"), "epoch_s"
    return pd.to_datetime(series, utc=True, errors="coerce"), "text_datetime"


def detect_timeframe_seconds(timestamps: pd.Series) -> Tuple[int, str]:
    diffs = timestamps.sort_values().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 60, "1m"
    mode_sec = int(diffs.round().mode().iloc[0])
    mapping = {60: "1m", 180: "3m", 300: "5m", 900: "15m", 1800: "30m", 3600: "1h", 14400: "4h", 86400: "1d"}
    return mode_sec, mapping.get(mode_sec, f"{mode_sec}s")


def sanitize_ohlc(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    essential = ["datetime", "open", "high", "low", "close", "volume"]
    before = len(df)
    df = df.dropna(subset=essential).copy()
    valid = (
        (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
        & (df["high"] >= df["low"])
    )
    df = df.loc[valid].copy()
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    return df, before - len(df)


def normalize_ohlc_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    rename_map: Dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        for col in out.columns:
            if col in aliases and target not in rename_map.values():
                rename_map[col] = target
                break
    out = out.rename(columns=rename_map)

    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"missing_columns:{missing}")

    raw_ts = pd.to_numeric(out["timestamp"], errors="coerce")
    if raw_ts.notna().mean() > 0.95:
        median = raw_ts.dropna().median()
        if median > 1e12:
            out["timestamp"] = pd.to_datetime(raw_ts, unit="ms", errors="coerce", utc=True)
        elif median > 1e9:
            out["timestamp"] = pd.to_datetime(raw_ts, unit="s", errors="coerce", utc=True)
        else:
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    else:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    else:
        out["volume"] = 0.0

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    out = out[(out["high"] >= out["low"])]
    out = out[(out["open"] >= out["low"]) & (out["open"] <= out["high"])]
    out = out[(out["close"] >= out["low"]) & (out["close"] <= out["high"])]
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return out


def infer_timeframe(df: pd.DataFrame) -> str:
    if len(df) < 3:
        raise ValueError("not_enough_rows_for_timeframe")
    deltas = df["timestamp"].sort_values().diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        raise ValueError("no_positive_timestamp_delta")
    step = float(deltas.mode().iloc[0])
    mapping = {60: "1m", 180: "3m", 300: "5m", 900: "15m", 1800: "30m", 3600: "1h", 14400: "4h", 86400: "1d"}
    nearest = min(mapping.keys(), key=lambda x: abs(x - step))
    if abs(nearest - step) > max(1.0, nearest * 0.1):
        raise ValueError(f"unknown_timeframe_step:{step}")
    return mapping[nearest]


def split_by_continuity(df: pd.DataFrame, timeframe_seconds: int) -> Tuple[List[pd.DataFrame], Dict[str, int]]:
    stats = {"gaps_detected": 0, "gaps_filled": 0, "rows_removed": 0}
    if df.empty or timeframe_seconds <= 0:
        return [df], stats

    time_diffs = df["datetime"].diff().dt.total_seconds()
    large_gaps = time_diffs > (timeframe_seconds * 3)
    stats["gaps_detected"] = int(large_gaps.sum())
    if stats["gaps_detected"] == 0:
        return [df], stats

    gap_indices = df[large_gaps].index.tolist()
    segments: List[Tuple[int, int]] = []
    start = 0
    for gap_idx in gap_indices:
        if gap_idx > start:
            segments.append((start, gap_idx - 1))
        start = gap_idx
    if start < len(df):
        segments.append((start, len(df) - 1))
    if not segments:
        return [df], stats
    out = []
    for s, e in segments:
        seg = df.iloc[s : e + 1].copy().reset_index(drop=True)
        if not seg.empty:
            out.append(seg)
    return out, stats


def _normalize_dataframe(df: pd.DataFrame, source_name: str, sheet_name: str | None, source_type: str, report: IngestionReport | None = None, cfg: IngestionConfig | None = None) -> NormalizedDataset:
    cfg = cfg or IngestionConfig()
    mapping = _map_columns(df)
    if report:
        report.log(f"[PARSE] mapped columns {source_name}{'::'+sheet_name if sheet_name else ''}: {mapping}")

    out = pd.DataFrame()
    out["datetime"], ts_kind = _parse_timestamp(df[mapping["timestamp"]])
    if report:
        report.log(f"[PARSE] timestamp mode={ts_kind} source={source_name}{'::'+sheet_name if sheet_name else ''}")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(df[mapping[col]], errors="coerce")

    out, removed = sanitize_ohlc(out)
    if len(out) > cfg.max_rows_per_dataset:
        out = out.tail(cfg.max_rows_per_dataset).reset_index(drop=True)
    timeframe_seconds, timeframe_label = detect_timeframe_seconds(out["datetime"])
    segments, continuity_stats = split_by_continuity(out, timeframe_seconds)
    if segments:
        valid = [seg for seg in segments if len(seg) >= cfg.min_rows_per_dataset]
        out = valid[0] if valid else segments[0]
    timeframe_seconds, timeframe_label = detect_timeframe_seconds(out["datetime"])
    if report:
        report.log(f"[PARSE] timeframe={timeframe_label} ({timeframe_seconds}s) source={source_name}{'::'+sheet_name if sheet_name else ''}")
        report.log(f"[PARSE] invalid rows removed={removed} source={source_name}{'::'+sheet_name if sheet_name else ''}")
        if continuity_stats["gaps_detected"] > 0:
            report.log(f"[CONTINUITY] source={source_name}{'::'+sheet_name if sheet_name else ''} gaps={continuity_stats['gaps_detected']} rows_removed={continuity_stats['rows_removed']}")

    return NormalizedDataset(source_name, source_type, sheet_name, timeframe_seconds, timeframe_label, {k: str(v) for k, v in mapping.items()}, out)


def read_excel_dataset(file_or_bytes: str | Path | bytes, source_name: str, sheet_name: str, report: IngestionReport | None = None, cfg: IngestionConfig | None = None) -> NormalizedDataset:
    io_obj = BytesIO(file_or_bytes) if isinstance(file_or_bytes, bytes) else file_or_bytes
    raw = _detect_header_and_read_excel(io_obj, sheet_name)
    return _normalize_dataframe(raw, source_name=source_name, sheet_name=sheet_name, source_type="excel", report=report, cfg=cfg)


def read_zip_members(path: str | Path) -> List[Tuple[str, pd.DataFrame]]:
    members: List[Tuple[str, pd.DataFrame]] = []
    with zipfile.ZipFile(path, "r") as z:
        for name in z.namelist():
            lname = name.lower()
            if lname.endswith(".csv"):
                with z.open(name) as f:
                    df = pd.read_csv(io.BytesIO(f.read()))
                    members.append((name, df))
            elif lname.endswith(".xlsx") or lname.endswith(".xls"):
                data = io.BytesIO(z.read(name))
                xls = pd.ExcelFile(data)
                for sheet in xls.sheet_names:
                    try:
                        raw = _detect_header_and_read_excel(data, sheet)
                        members.append((f"{name}::{sheet}", raw))
                    except Exception:
                        continue
    return members


def read_zip_file(path: str | Path) -> List[pd.DataFrame]:
    return [df for _, df in read_zip_members(path)]


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_ohlc_dataframe(df)


def read_zip_datasets(zip_path: str | Path, cfg: IngestionConfig | None = None) -> IngestionReport:
    cfg = cfg or IngestionConfig()
    debug = make_ingestion_debug()
    report = IngestionReport(debug=debug)
    try:
        raw_dfs = read_zip_file(zip_path)
        for i, raw in enumerate(raw_dfs):
            rows_raw = len(raw)
            clean = normalize_ohlc(raw)
            if len(clean) > cfg.max_rows_per_dataset:
                clean = clean.tail(cfg.max_rows_per_dataset).reset_index(drop=True)
            if len(clean) < cfg.min_rows_per_dataset:
                log_ingestion(debug, "warning", f"{zip_path}::csv_{i}", "dataset_too_small", rows_raw=rows_raw, rows_clean=len(clean))
                continue
            tf = infer_timeframe(clean)
            tf_seconds, _ = detect_timeframe_seconds(clean["timestamp"])
            report.datasets.append(
                NormalizedDataset(str(zip_path), "zip_csv", f"csv_{i}", tf_seconds, tf, {}, clean.rename(columns={"timestamp": "datetime"}))
            )
            debug["retained_files"].append({"file": f"{zip_path}::csv_{i}", "timeframe": tf, "rows": len(clean)})
            log_ingestion(debug, "info", f"{zip_path}::csv_{i}", "normalized", rows_raw=rows_raw, rows_clean=len(clean))
    except Exception as e:
        log_ingestion(debug, "error", zip_path, "zip_read_failed", error=str(e))
    return report


def _ordered_supported_files(root: Path, strategy: str) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file()]
    if strategy == "most_recent":
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    if strategy == "largest":
        return sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    # balanced
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)


def scan_data_sources(root_path: str | Path, cfg: IngestionConfig | None = None) -> IngestionReport:
    cfg = cfg or IngestionConfig()
    import os

    root = Path(root_path).expanduser().resolve()
    debug = make_ingestion_debug()
    report = IngestionReport(root_scanned=str(root), debug=debug)
    datasets: List[NormalizedDataset] = []
    debug["phase"] = "scan"

    processed_supported = 0
    for walk_root, _, files in os.walk(root):
        for file in files:
            path = Path(walk_root) / file
            debug["files_detected"] += 1
            log_ingestion(debug, "info", path, "file_detected", suffix=path.suffix.lower())
            try:
                if file.endswith(".zip"):
                    if processed_supported >= cfg.max_files:
                        log_ingestion(debug, "warning", path, "max_files_limit_reached", limit=cfg.max_files)
                        continue
                    processed_supported += 1
                    raw_members = read_zip_members(path)
                elif file.endswith(".csv"):
                    if processed_supported >= cfg.max_files:
                        log_ingestion(debug, "warning", path, "max_files_limit_reached", limit=cfg.max_files)
                        continue
                    processed_supported += 1
                    raw_members = [(path.name, pd.read_csv(path))]
                elif file.endswith(".xlsx") or file.endswith(".xls"):
                    if processed_supported >= cfg.max_files:
                        log_ingestion(debug, "warning", path, "max_files_limit_reached", limit=cfg.max_files)
                        continue
                    processed_supported += 1
                    xls = pd.ExcelFile(path)
                    raw_members = []
                    for sheet in xls.sheet_names:
                        try:
                            raw = _detect_header_and_read_excel(path, sheet)
                            raw_members.append((f"{path.name}::{sheet}", raw))
                        except Exception as e:
                            log_ingestion(debug, "warning", f"{path}::{sheet}", "excel_sheet_invalid", error=str(e))
                else:
                    log_ingestion(debug, "warning", path, "unsupported_extension")
                    continue

                for member_name, raw in raw_members:
                    rows_raw = len(raw)
                    try:
                        clean = normalize_ohlc(raw)
                        rows_clean = len(clean)
                        log_ingestion(debug, "info", f"{path}::{member_name}", "normalized", rows_raw=rows_raw, rows_clean=rows_clean)
                    except Exception as e:
                        log_ingestion(debug, "error", f"{path}::{member_name}", "normalize_failed", rows_raw=rows_raw, error=str(e))
                        continue
                    if len(clean) < cfg.min_rows_per_dataset:
                        log_ingestion(debug, "warning", f"{path}::{member_name}", "dataset_too_small", rows_clean=len(clean))
                        continue
                    if len(clean) > cfg.max_rows_per_dataset:
                        clean = clean.tail(cfg.max_rows_per_dataset).reset_index(drop=True)
                    try:
                        tf = infer_timeframe(clean)
                    except Exception as e:
                        log_ingestion(debug, "warning", f"{path}::{member_name}", "timeframe_inference_failed", error=str(e))
                        continue
                    tf_seconds, _ = detect_timeframe_seconds(clean["timestamp"])
                    segs, seg_stats = split_by_continuity(clean.rename(columns={"timestamp": "datetime"}), tf_seconds)
                    for seg_i, seg in enumerate(segs):
                        if len(seg) < cfg.min_rows_per_dataset:
                            continue
                        ds_name = f"{path}::{member_name}::seg{seg_i}"
                        ds_type = "csv" if file.endswith(".csv") else ("excel" if file.endswith(".xlsx") or file.endswith(".xls") else "zip_mixed")
                        datasets.append(NormalizedDataset(str(ds_name), ds_type, None, tf_seconds, tf, {}, seg))
                        debug["retained_files"].append({"file": str(ds_name), "timeframe": tf, "rows": len(seg), "gaps_detected": seg_stats["gaps_detected"]})
            except Exception as e:
                log_ingestion(debug, "error", path, "read_failed", error=str(e))

    report.datasets = datasets
    report.total_files_found = debug["files_detected"]
    report.supported_files_found = len([x for x in debug["ingestion_logs"] if x["message"] == "file_detected"])
    report.retained_files = [x["file"] for x in debug["retained_files"]]
    report.ignored_files = debug["ignored_files"]
    debug["datasets_retained"] = len(datasets)
    if debug["datasets_retained"] == 0:
        debug["ok"] = False
        debug["error"] = "no_valid_dataset_retained"
    if VERBOSE_INGESTION:
        print(json.dumps(debug["ingestion_logs"][-20:], ensure_ascii=False, indent=2, default=str))
    return report


def merge_compatible_datasets(datasets: List[NormalizedDataset], report: IngestionReport | None = None, mode: str = "balanced", max_total_rows: int | None = None) -> pd.DataFrame:
    if not datasets:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    grouped: Dict[int, List[NormalizedDataset]] = {}
    for d in datasets:
        grouped.setdefault(d.timeframe_seconds, []).append(d)

    def score(tf: int) -> float:
        ds = grouped[tf]
        rows = sum(len(x.dataframe) for x in ds)
        if mode == "dataset_count":
            return float(len(ds))
        if mode == "row_count":
            return float(rows)
        return float(len(ds) + rows / 10_000.0)

    target_tf = max(grouped, key=score)
    selected = grouped[target_tf]

    if report:
        report.log(f"[LEARN] datasets retained for merge={len(selected)} target_tf={target_tf}")
        for tf, ds_list in grouped.items():
            if tf != target_tf:
                report.log(f"[LEARN] timeframe ignored tf={tf} count={len(ds_list)}")

    merged = pd.concat([d.dataframe.assign(source=d.name, sheet=d.sheet_name) for d in selected], ignore_index=True)
    merged = merged.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    if max_total_rows and len(merged) > max_total_rows:
        merged = merged.tail(max_total_rows)
        if report:
            report.log(f"[LEARN] merged rows clipped to {max_total_rows}")
    if report:
        report.log(f"[LEARN] merged rows={len(merged)}")
    return merged.reset_index(drop=True)
