from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict
from collections import Counter

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src.data_ingestion import IngestionConfig, IngestionReport, NormalizedDataset, scan_data_sources
from src.data_learning import CalibrationBundle, learn_from_report
from src.game_engine import TradingGameEngine

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "web"
DATA_ROOT = ROOT / "data"

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
logger = logging.getLogger(__name__)

state_lock = threading.Lock()
server_state: Dict[str, Any] = {
    "status": "loading",  # loading | ready | failed
    "phase": "boot",
    "error": None,
    "fallback_used": False,
    "load_time_ms": None,
    "files_detected": 0,
    "datasets_retained": 0,
    "rows_merged": 0,
    "timeframe": "unknown",
    "scan_time_ms": None,
    "learn_time_ms": None,
    "retained_files": [],
    "ignored_files": [],
    "ingestion_logs_tail": [],
    "data_mode": "real",
    "fallback_reason": None,
    "session_timezone_mode": "utc",
}

engine: TradingGameEngine | None = None
cached_bundle: CalibrationBundle | None = None
# Increase max_files to avoid prematurely stopping ingestion when a single ZIP contains many shards.
INGEST_CFG = IngestionConfig(max_files=4096, max_total_rows=300_000, max_rows_per_dataset=80_000, selection_strategy="balanced")


def _count_supported_files(root: Path) -> int:
    supported = {".csv", ".xlsx", ".xls", ".zip"}
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in supported)


def _fallback_bundle() -> CalibrationBundle:
    dt = pd.date_range("2025-01-01", periods=1200, freq="1min", tz="UTC")
    rng = pd.Series(range(len(dt)), dtype=float)
    close = 2020.0 + (rng * 0.01) + pd.Series((rng % 17 - 8) * 0.03, dtype=float)
    df = pd.DataFrame(
        {
            "datetime": dt,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.35,
            "low": close - 0.35,
            "close": close,
            "volume": 900 + (rng % 40) * 15,
        }
    )
    ds = NormalizedDataset("fallback_synth", "generated", None, 60, "1m", {}, df)
    return learn_from_report(IngestionReport(datasets=[ds], logs=["fallback dataset generated in-memory"]))


def _set_state(**kwargs: Any) -> None:
    with state_lock:
        server_state.update(kwargs)


def _initialize_runtime() -> None:
    global cached_bundle, engine
    logger.info("[BOOT] background initialization started")
    start = time.perf_counter()
    _set_state(status="loading", phase="scan_data", error=None)
    files_detected = _count_supported_files(DATA_ROOT)
    report = IngestionReport()
    try:
        t_scan0 = time.perf_counter()
        report = scan_data_sources(str(DATA_ROOT), cfg=INGEST_CFG)
        scan_ms = int((time.perf_counter() - t_scan0) * 1000)
        _set_state(phase="learn_behavior", files_detected=files_detected)
        if not report.datasets:
            bundle = _fallback_bundle()
            learn_ms = None
            used_fallback = True
            _set_state(fallback_reason=f"Aucun dataset valide trouvé dans {DATA_ROOT}")
        else:
            total_rows = sum(len(ds.dataframe) for ds in report.datasets)
            if total_rows < 100:
                raise ValueError(f"Données insuffisantes: {total_rows} lignes trouvées (minimum 100)")
            t_learn0 = time.perf_counter()
            bundle = learn_from_report(report, cfg=INGEST_CFG)
            learn_ms = int((time.perf_counter() - t_learn0) * 1000)
            used_fallback = False
    except Exception as exc:
        scan_ms = None
        learn_ms = None
        try:
            bundle = _fallback_bundle()
            used_fallback = True
            _set_state(error=f"Échec chargement données: {exc}. Utilisation fallback synthétique.", fallback_reason=str(exc))
        except Exception as fallback_exc:
            _set_state(
                status="failed",
                phase="failed",
                error=f"Échec initialisation runtime: {fallback_exc}",
                fallback_reason=str(exc),
                load_time_ms=int((time.perf_counter() - start) * 1000),
            )
            logger.exception("[BOOT] runtime initialization failed")
            return

    with state_lock:
        cached_bundle = bundle
        engine = TradingGameEngine(bundle)
        server_state.update(
            {
                "status": "ready",
                "phase": "ready",
                "fallback_used": used_fallback,
                "load_time_ms": int((time.perf_counter() - start) * 1000),
                "datasets_retained": int(len(report.datasets)) if not used_fallback else 1,
                "rows_merged": int(len(bundle.merged_df)),
                "timeframe": bundle.timeframe_label,
                "scan_time_ms": scan_ms,
                "learn_time_ms": learn_ms,
                "retained_files": list(getattr(report, "debug", {}).get("retained_files", [])) if not used_fallback else [],
                "ignored_files": list(getattr(report, "debug", {}).get("ignored_files", []))[:80] if not used_fallback else [],
                "ingestion_logs_tail": list(getattr(report, "debug", {}).get("ingestion_logs", []))[-80:] if not used_fallback else ["fallback used"],
                "data_mode": "fallback" if used_fallback else "real",
            }
        )
    logger.info(
        "[BOOT] runtime state ready (mode=%s, fallback=%s, rows=%s)",
        server_state.get("data_mode"),
        server_state.get("fallback_used"),
        server_state.get("rows_merged"),
    )


def _start_background_init() -> None:
    logger.info("[BOOT] launching initialization thread")
    threading.Thread(target=_initialize_runtime, daemon=True).start()


def _ensure_engine_ready() -> TradingGameEngine:
    with state_lock:
        current = server_state["status"]
        e = engine
    if current != "ready" or e is None:
        raise RuntimeError("Engine not ready")
    return e


def _debug_payload() -> Dict[str, Any]:
    with state_lock:
        ignored = server_state["ignored_files"]
        retained = server_state["retained_files"]
        rejection_summary = Counter()
        for item in ignored:
            rejection_summary[item.get("reason", "unknown")] += 1
        return {
            "status": server_state["status"],
            "phase": server_state["phase"],
            "files_detected": server_state["files_detected"],
            "datasets_retained": server_state["datasets_retained"],
            "rows_merged": server_state["rows_merged"],
            "timeframe": server_state["timeframe"],
            "fallback_used": server_state["fallback_used"],
            "load_time_ms": server_state["load_time_ms"],
            "scan_time_ms": server_state["scan_time_ms"],
            "learn_time_ms": server_state["learn_time_ms"],
            "retained_files": retained[:20],
            "ignored_files": ignored[:20],
            "ingestion_logs_tail": server_state["ingestion_logs_tail"][-20:],
            "rejection_summary": dict(rejection_summary),
            "error": server_state["error"],
            "data_mode": server_state["data_mode"],
            "fallback_reason": server_state["fallback_reason"],
            "engine_timezone": "UTC",
            "display_timezone": "local_browser",
            "session_timezone_mode": server_state.get("session_timezone_mode", "utc"),
        }


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/health")
def health():
    d = _debug_payload()
    return jsonify({"ok": d["status"] == "ready", **d})


@app.get("/api/init_status")
def init_status():
    return jsonify({"ok": True, **_debug_payload()})


@app.post("/api/init")
def init_game():
    with state_lock:
        status = server_state["status"]
        e = engine
    if status != "ready" or e is None:
        return jsonify({"ok": False, "loading": True, "retryable": True, "reason": "initialization in progress", "debug": _debug_payload()})

    snap = e.snapshot()
    snap["data_mode"] = _debug_payload()["data_mode"]
    initial_candles = e.history.tail(250).to_dict(orient="records")
    return jsonify({
        "ok": True,
        "timeframe": e.bundle.timeframe_label,
        "data_mode": _debug_payload()["data_mode"],
        "disable_auto_run": bool(_debug_payload()["data_mode"] == "fallback"),
        "initial_candles": initial_candles,
        "snapshot": snap,
        "debug": _debug_payload(),
        **snap,
    })


@app.post("/api/step")
def step():
    try:
        e = _ensure_engine_ready()
    except RuntimeError:
        return jsonify({"ok": False, "reason": "Engine not ready", "debug": _debug_payload()})
    out = e.step_market()
    out["data_mode"] = _debug_payload()["data_mode"]
    if isinstance(out.get("snapshot"), dict):
        out["snapshot"]["data_mode"] = out["data_mode"]
    return jsonify(out)


@app.post("/api/order")
def order():
    try:
        e = _ensure_engine_ready()
    except RuntimeError:
        return jsonify({"ok": False, "reason": "Engine not ready", "debug": _debug_payload()})
    data: Dict[str, Any] = request.json or {}
    return jsonify(e.place_order(side=data.get("side", "buy"), size=float(data.get("size", 1.0)), leverage=int(data.get("leverage", 50))))


@app.post("/api/close")
def close():
    try:
        e = _ensure_engine_ready()
    except RuntimeError:
        return jsonify({"ok": False, "reason": "Engine not ready", "debug": _debug_payload()})
    data = request.json or {}
    fraction = float(data.get("fraction", 1.0))
    return jsonify(e.close_fraction(fraction))


@app.post("/api/deposit")
def deposit():
    try:
        e = _ensure_engine_ready()
    except RuntimeError:
        return jsonify({"ok": False, "reason": "Engine not ready", "debug": _debug_payload()})
    amt = float((request.json or {}).get("amount", 0.0))
    return jsonify(e.deposit(amt))


@app.post("/api/withdraw")
def withdraw():
    try:
        e = _ensure_engine_ready()
    except RuntimeError:
        return jsonify({"ok": False, "reason": "Engine not ready", "debug": _debug_payload()})
    amt = float((request.json or {}).get("amount", 0.0))
    return jsonify(e.withdraw(amt))


@app.post("/api/reset")
def reset():
    global engine
    with state_lock:
        bundle = cached_bundle
    if bundle is None:
        return jsonify({"ok": False, "reason": "Engine not ready", "debug": _debug_payload()})
    engine = TradingGameEngine(bundle)
    return jsonify({"ok": True, "snapshot": engine.snapshot()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _start_background_init()
    app.run(host="127.0.0.1", port=8000, debug=False)
