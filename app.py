from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_learning import calibrate_from_dataframe, load_ohlcv_csv
from src.simulator import SyntheticXAUUSDMarket
from src.trading import Account, Position, compute_fee, compute_slippage, compute_spread

st.set_page_config(page_title="Simulateur XAUUSD synthétique", layout="wide")
st.title("Moteur de marché synthétique XAUUSD")
st.caption("Microstructure explicite: OFI, liquidité, sweep, FOMO, impact, volatilité, sessions.")

if "sim_df" not in st.session_state:
    st.session_state.sim_df = pd.DataFrame()
if "account" not in st.session_state:
    st.session_state.account = Account(balance=10000.0)

st.sidebar.header("Configuration")
timezone = st.sidebar.text_input("Timezone (IANA)", value="UTC")
steps = st.sidebar.slider("Nombre de pas synthétiques", 100, 5000, 1000, 100)

uploaded = st.file_uploader("Charger CSV OHLCV XAUUSD", type=["csv"])

hist_df = None
if uploaded:
    tmp = Path("data") / uploaded.name
    tmp.write_bytes(uploaded.getvalue())
    hist_df = load_ohlcv_csv(tmp)
    st.success(f"Données chargées: {len(hist_df)} bougies")

if hist_df is not None:
    calib = calibrate_from_dataframe(hist_df).__dict__
    st.subheader("Calibration extraite")
    st.json(calib)

    if st.button("Lancer simulation synthétique"):
        engine = SyntheticXAUUSDMarket(calibration=calib, timezone=timezone)
        sim_df = engine.run(hist_df, n_steps=steps)
        st.session_state.sim_df = sim_df

if not st.session_state.sim_df.empty:
    sim = st.session_state.sim_df
    st.subheader("Prix simulé")
    st.line_chart(sim.set_index("datetime")["close"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Dernier prix", f"{sim['close'].iloc[-1]:.2f}")
    c2.metric("Sweeps", int(sim["sweep"].sum()))
    c3.metric("FOMO moyen", f"{sim['fomo'].mean():.3f}")

    st.subheader("Trading")
    side = st.selectbox("Direction", ["long", "short"])
    size = st.number_input("Taille", min_value=0.01, value=1.0, step=0.1)
    leverage = st.selectbox("Levier", [1, 10, 50, 100, 200], index=1)
    sl = st.number_input("Stop Loss", value=float(sim["close"].iloc[-1] - 5.0))
    tp = st.number_input("Take Profit", value=float(sim["close"].iloc[-1] + 8.0))

    px = float(sim["close"].iloc[-1])
    vol = float(sim["close"].pct_change().std())
    liq = float(sim["volume"].iloc[-1])
    spread = compute_spread(volatility=vol, liquidity=liq)

    if st.button("Ouvrir position"):
        slip = compute_slippage(order_size=size, liquidity=liq, volatility=vol)
        fill = px + spread / 2 + slip if side == "long" else px - spread / 2 - slip
        pos = Position(side=side, entry=fill, size=size, stop_loss=sl, take_profit=tp, leverage=leverage)
        fee = compute_fee(pos.position_value())
        pos.fee_paid = fee
        ok = st.session_state.account.open_position(pos, current_price=px)
        if ok:
            st.success(f"Position {side} ouverte @ {fill:.2f}, frais {fee:.2f}")
        else:
            st.error("Marge insuffisante")

    account = st.session_state.account
    liquidation_pnls = account.enforce_liquidation(px)
    if liquidation_pnls:
        st.warning(f"Liquidation forcée exécutée: {len(liquidation_pnls)} positions")

    st.subheader("Compte")
    st.write(
        {
            "balance": round(account.balance, 2),
            "equity": round(account.equity(px), 2),
            "margin": round(account.margin_used(), 2),
            "free_margin": round(account.free_margin(px), 2),
            "positions_ouvertes": len(account.positions),
            "spread_estime": round(spread, 4),
        }
    )

    to_close = []
    for i, p in enumerate(account.positions):
        pnl = p.unrealized_pnl(px)
        st.write(f"#{i} {p.side} entry={p.entry:.2f} size={p.size:.2f} pnl={pnl:.2f}")
        if (p.side == "long" and (px <= p.stop_loss or px >= p.take_profit)) or (
            p.side == "short" and (px >= p.stop_loss or px <= p.take_profit)
        ):
            to_close.append(i)

    for idx in reversed(to_close):
        closed = account.close_position(idx, px, fee=0.0)
        st.info(f"Position clôturée automatiquement, PnL={closed:.2f}")

    with st.expander("Détails microstructure"):
        st.dataframe(sim.tail(200))
