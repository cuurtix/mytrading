from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory

from src.data_learning import learn_from_root
from src.game_engine import TradingGameEngine

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "web"
DATA_ROOT = ROOT / "data"

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")

engine: TradingGameEngine | None = None


def _ensure_engine() -> TradingGameEngine:
    global engine
    if engine is None:
        bundle = learn_from_root(str(DATA_ROOT))
        engine = TradingGameEngine(bundle)
    return engine


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.post("/api/init")
def init_game():
    global engine
    bundle = learn_from_root(str(DATA_ROOT))
    engine = TradingGameEngine(bundle)
    m = engine._mark_to_market(float(engine.history["close"].iloc[-1]))
    return jsonify({"ok": True, "metrics": m, "timeframe": engine.bundle.timeframe_label})


@app.post("/api/step")
def step():
    e = _ensure_engine()
    return jsonify(e.step_market())


@app.post("/api/order")
def order():
    e = _ensure_engine()
    data: Dict[str, Any] = request.json or {}
    res = e.place_order(side=data.get("side", "buy"), size=float(data.get("size", 1.0)), leverage=int(data.get("leverage", 50)))
    return jsonify({"ok": True, "execution": res})


@app.post("/api/close")
def close():
    e = _ensure_engine()
    data = request.json or {}
    fraction = float(data.get("fraction", 1.0))
    res = e.close_fraction(fraction)
    return jsonify({"ok": True, **res})


@app.post("/api/deposit")
def deposit():
    e = _ensure_engine()
    amt = float((request.json or {}).get("amount", 0.0))
    e.deposit(amt)
    return jsonify({"ok": True})


@app.post("/api/withdraw")
def withdraw():
    e = _ensure_engine()
    amt = float((request.json or {}).get("amount", 0.0))
    ok = e.withdraw(amt)
    return jsonify({"ok": ok})


@app.post("/api/reset")
def reset():
    e = _ensure_engine()
    e.reset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
