from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import difflib
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

    def log(self, message: str) -> None:
        self.logs.append(message)


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _best_match(col: str, aliases: set[str]) -> bool:
    if col in aliases:
        return True
    matches = difflib.get_close_matches(col, list(aliases), n=1, cutoff=0.8)
    return bool(matches)


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

    primary = next((v for v in VOLUME_PRIORITY if v in mapping), None)
    if primary is None and volumes:
        primary = volumes[0]
    if primary is not None:
        mapping["volume"] = mapping[primary]

    return mapping


def _detect_header_and_read_excel(io_obj, sheet_name: str) -> pd.DataFrame:
    for header_row in range(0, 8):
        try:
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
    removed = before - len(df)
    return df, removed


def _normalize_dataframe(df: pd.DataFrame, source_name: str, sheet_name: str | None, source_type: str, report: IngestionReport | None = None) -> NormalizedDataset:
    mapping = _map_columns(df)
    if report:
        report.log(f"colonnes mappées [{source_name}{'::'+sheet_name if sheet_name else ''}]: {mapping}")

    out = pd.DataFrame()
    out["datetime"], ts_kind = _parse_timestamp(df[mapping["timestamp"]])
    if report:
        report.log(f"timestamp détecté comme {ts_kind} [{source_name}{'::'+sheet_name if sheet_name else ''}]")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(df[mapping[col]], errors="coerce")

    for vcol in ["basevolume", "quotevolume", "usdtvolume", "tick_volume", "real_volume"]:
        if vcol in mapping:
            out[vcol] = pd.to_numeric(df[mapping[vcol]], errors="coerce")

    out, removed = sanitize_ohlc(out)
    timeframe_seconds, timeframe_label = detect_timeframe_seconds(out["datetime"])
    if report:
        report.log(f"timeframe détecté: {timeframe_label} ({timeframe_seconds}s) [{source_name}{'::'+sheet_name if sheet_name else ''}]")
        report.log(f"lignes invalides supprimées: {removed} [{source_name}{'::'+sheet_name if sheet_name else ''}]")

    return NormalizedDataset(source_name, source_type, sheet_name, timeframe_seconds, timeframe_label, {k: str(v) for k, v in mapping.items()}, out)


def read_excel_dataset(file_or_bytes: str | Path | bytes, source_name: str, sheet_name: str, report: IngestionReport | None = None) -> NormalizedDataset:
    io_obj = BytesIO(file_or_bytes) if isinstance(file_or_bytes, bytes) else file_or_bytes
    raw = _detect_header_and_read_excel(io_obj, sheet_name)
    return _normalize_dataframe(raw, source_name=source_name, sheet_name=sheet_name, source_type="excel", report=report)


def read_zip_datasets(zip_path: str | Path) -> IngestionReport:
    report = IngestionReport()
    report.log(f"zip trouvé: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            lower = info.filename.lower()
            if not (lower.endswith(".xlsx") or lower.endswith(".xls") or lower.endswith(".csv")):
                report.log(f"ignoré (format non supporté): {info.filename}")
                continue

            data = zf.read(info.filename)
            report.log(f"fichier trouvé dans zip: {info.filename}")
            if lower.endswith(".csv"):
                try:
                    raw = pd.read_csv(BytesIO(data))
                    ds = _normalize_dataframe(raw, source_name=info.filename, sheet_name=None, source_type="zip_csv", report=report)
                    report.datasets.append(ds)
                    report.log(f"dataset normalisé: {info.filename}")
                except Exception as exc:
                    report.log(f"échec lecture csv {info.filename}: {exc}")
                continue

            try:
                xls = pd.ExcelFile(BytesIO(data))
                report.log(f"workbook ouvert: {info.filename}")
                for sheet in xls.sheet_names:
                    try:
                        report.log(f"feuille lue: {info.filename}::{sheet}")
                        ds = read_excel_dataset(data, source_name=info.filename, sheet_name=sheet, report=report)
                        report.datasets.append(ds)
                        report.log(f"dataset normalisé: {info.filename}::{sheet}")
                    except Exception as exc:
                        report.log(f"feuille ignorée {info.filename}::{sheet}: {exc}")
            except Exception as exc:
                report.log(f"échec workbook {info.filename}: {exc}")

    return report


def scan_data_sources(root_path: str | Path) -> IngestionReport:
    root = Path(root_path)
    report = IngestionReport()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        try:
            if lower.endswith(".zip"):
                zip_report = read_zip_datasets(path)
                report.datasets.extend(zip_report.datasets)
                report.logs.extend(zip_report.logs)
            elif lower.endswith(".csv"):
                raw = pd.read_csv(path)
                ds = _normalize_dataframe(raw, source_name=str(path), sheet_name=None, source_type="csv", report=report)
                report.datasets.append(ds)
                report.log(f"dataset normalisé: {path}")
            elif lower.endswith(".xlsx") or lower.endswith(".xls"):
                xls = pd.ExcelFile(path)
                report.log(f"workbook ouvert: {path}")
                for sheet in xls.sheet_names:
                    report.log(f"feuille lue: {path}::{sheet}")
                    ds = read_excel_dataset(path, source_name=str(path), sheet_name=sheet, report=report)
                    report.datasets.append(ds)
                    report.log(f"dataset normalisé: {path}::{sheet}")
        except Exception as exc:
            report.log(f"source ignorée {path}: {exc}")

    return report


def merge_compatible_datasets(datasets: List[NormalizedDataset], report: IngestionReport | None = None, mode: str = "timeframe") -> pd.DataFrame:
    if not datasets:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    grouped: Dict[int, List[NormalizedDataset]] = {}
    for d in datasets:
        grouped.setdefault(d.timeframe_seconds, []).append(d)

    target_tf = max(grouped, key=lambda k: len(grouped[k]))
    selected = grouped[target_tf]

    if report:
        report.log(f"fusion: timeframe cible {target_tf}s avec {len(selected)} datasets")
        for tf, ds_list in grouped.items():
            if tf != target_tf:
                report.log(f"dataset ignoré pour fusion (timeframe incompatible {tf}s): {len(ds_list)}")

    merged = pd.concat([d.dataframe.assign(source=d.name, sheet=d.sheet_name) for d in selected], ignore_index=True)
    merged = merged.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    if report:
        report.log(f"dataset fusionné: {len(merged)} lignes")
    return merged.reset_index(drop=True)
