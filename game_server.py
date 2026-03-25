from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src.data_ingestion import IngestionReport, NormalizedDataset
from src.data_learning import learn_from_report, learn_from_root
from src.game_engine import TradingGameEngine

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "web"
DATA_ROOT = ROOT / "data"

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")

engine: TradingGameEngine | None = None


def _fallback_engine() -> TradingGameEngine:
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
    bundle = learn_from_report(IngestionReport(datasets=[ds], logs=["fallback dataset generated in-memory"]))
    return TradingGameEngine(bundle)


def _ensure_engine() -> TradingGameEngine:
    global engine
    if engine is None:
        try:
            bundle = learn_from_root(str(DATA_ROOT))
            engine = TradingGameEngine(bundle)
        except Exception:
            engine = _fallback_engine()
    return engine


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.post("/api/init")
def init_game():
    global engine
    try:
        bundle = learn_from_root(str(DATA_ROOT))
        engine = TradingGameEngine(bundle)
    except Exception:
        engine = _fallback_engine()
    snap = engine.snapshot()
    initial_candles = engine.history.tail(250).to_dict(orient="records")
    return jsonify({"ok": True, "timeframe": engine.bundle.timeframe_label, "initial_candles": initial_candles, "snapshot": snap, **snap})


@app.post("/api/step")
def step():
    e = _ensure_engine()
    return jsonify(e.step_market())


@app.post("/api/order")
def order():
    e = _ensure_engine()
    data: Dict[str, Any] = request.json or {}
    return jsonify(e.place_order(side=data.get("side", "buy"), size=float(data.get("size", 1.0)), leverage=int(data.get("leverage", 50))))


@app.post("/api/close")
def close():
    e = _ensure_engine()
    data = request.json or {}
    fraction = float(data.get("fraction", 1.0))
    return jsonify(e.close_fraction(fraction))


@app.post("/api/deposit")
def deposit():
    e = _ensure_engine()
    amt = float((request.json or {}).get("amount", 0.0))
    return jsonify(e.deposit(amt))


@app.post("/api/withdraw")
def withdraw():
    e = _ensure_engine()
    amt = float((request.json or {}).get("amount", 0.0))
    return jsonify(e.withdraw(amt))


@app.post("/api/reset")
def reset():
    e = _ensure_engine()
    return jsonify(e.reset())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
