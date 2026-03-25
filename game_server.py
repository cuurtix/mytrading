from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src.data_ingestion import IngestionConfig, IngestionReport, NormalizedDataset, scan_data_sources
from src.data_learning import CalibrationBundle, learn_from_report
from src.game_engine import TradingGameEngine

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "web"
DATA_ROOT = ROOT / "data"

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")

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
}

engine: TradingGameEngine | None = None
cached_bundle: CalibrationBundle | None = None
INGEST_CFG = IngestionConfig(max_files=96, max_total_rows=300_000, max_rows_per_dataset=80_000, selection_strategy="balanced")


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
            raise ValueError("No valid dataset found under data/")
        t_learn0 = time.perf_counter()
        bundle = learn_from_report(report, cfg=INGEST_CFG)
        learn_ms = int((time.perf_counter() - t_learn0) * 1000)
        used_fallback = False
    except Exception as exc:
        scan_ms = None
        learn_ms = None
        bundle = _fallback_bundle()
        used_fallback = True
        _set_state(error=f"Primary calibration failed, fallback used: {exc}")

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
                "retained_files": list(getattr(report, "retained_files", [])) if not used_fallback else [],
                "ignored_files": list(getattr(report, "ignored_files", []))[:80] if not used_fallback else [],
                "ingestion_logs_tail": list(getattr(report, "logs", []))[-80:] if not used_fallback else ["fallback used"],
            }
        )


def _ensure_engine_ready() -> TradingGameEngine:
    with state_lock:
        current = server_state["status"]
        e = engine
    if current != "ready" or e is None:
        raise RuntimeError("Engine not ready")
    return e


def _debug_payload() -> Dict[str, Any]:
    with state_lock:
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
            "retained_files": server_state["retained_files"],
            "ignored_files": server_state["ignored_files"],
            "ingestion_logs_tail": server_state["ingestion_logs_tail"],
            "error": server_state["error"],
        }


def _start_background_init() -> None:
    t = threading.Thread(target=_initialize_runtime, daemon=True)
    t.start()


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
        return jsonify({"ok": False, "loading": True, "reason": "initialization in progress", "debug": _debug_payload()})

    snap = e.snapshot()
    initial_candles = e.history.tail(250).to_dict(orient="records")
    return jsonify({
        "ok": True,
        "timeframe": e.bundle.timeframe_label,
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
    return jsonify(e.step_market())


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


_start_background_init()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
