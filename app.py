from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from src.data_ingestion import IngestionReport, read_zip_datasets, scan_data_sources
from src.data_learning import bundle_summary, learn_from_report
from src.simulator import SimulationConfig, SyntheticMarketSimulator

st.set_page_config(page_title="Moteur XAUUSD data-driven", layout="wide")
st.title("Moteur synthétique XAUUSD (appris sur vos données)")

if "bundle" not in st.session_state:
    st.session_state.bundle = None
if "sim_df" not in st.session_state:
    st.session_state.sim_df = pd.DataFrame()

st.sidebar.header("Sources de données")
uploaded = st.sidebar.file_uploader("Charger .csv/.xlsx/.xls/.zip", type=["csv", "xlsx", "xls", "zip"])
local_root = st.sidebar.text_input("Ou dossier local (optionnel)", value="")

report = IngestionReport()


def _append_report(src: IngestionReport) -> None:
    report.datasets.extend(src.datasets)
    report.logs.extend(src.logs)


if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / uploaded.name
        fpath.write_bytes(uploaded.getvalue())
        if suffix == ".zip":
            _append_report(read_zip_datasets(fpath))
        else:
            _append_report(scan_data_sources(td))

if local_root:
    try:
        _append_report(scan_data_sources(local_root))
    except Exception as exc:
        st.error(f"Erreur scan local: {exc}")

st.subheader("Datasets détectés")
if report.datasets:
    meta = [
        {
            "source": d.name,
            "type": d.source_type,
            "sheet": d.sheet_name,
            "rows": len(d.dataframe),
            "timeframe": d.timeframe_label,
            "mapped_columns": d.mapped_columns,
        }
        for d in report.datasets
    ]
    st.dataframe(pd.DataFrame(meta))
else:
    st.info("Aucune source détectée pour le moment.")

if st.button("Apprendre le comportement marché"):
    try:
        bundle = learn_from_report(report)
        st.session_state.bundle = bundle
        st.success("Apprentissage terminé")
    except Exception as exc:
        st.error(str(exc))

if st.session_state.bundle is not None:
    bundle = st.session_state.bundle
    st.subheader("Statistiques apprises")
    st.json(bundle_summary(bundle))

    st.subheader("États appris")
    states = bundle.learned.feature_df["state"].value_counts(normalize=True).rename("freq").reset_index()
    states.columns = ["state", "freq"]
    st.dataframe(states)

    st.subheader("Stats conditionnelles")
    st.write("Returns conditionnels par état")
    st.dataframe(pd.DataFrame(bundle.learned.conditional_returns).T)
    st.write("Ranges conditionnels par état")
    st.dataframe(pd.DataFrame(bundle.learned.conditional_ranges).T)

    with st.expander("Logs explicites"):
        for log in bundle.logs:
            st.write("-", log)

    steps = st.slider("Nombre de bougies synthétiques", 100, 5000, 800, 100)
    seed = st.number_input("Seed", value=11, step=1)

    if st.button("Lancer simulation calibrée"):
        sim = SyntheticMarketSimulator(bundle.learned, timeframe_seconds=bundle.timeframe_seconds)
        st.session_state.sim_df = sim.run(bundle.merged_df, SimulationConfig(seed=int(seed), n_steps=steps))

if not st.session_state.sim_df.empty:
    sim_df = st.session_state.sim_df
    st.subheader("Prix synthétique")
    st.line_chart(sim_df.set_index("datetime")["close"])
    st.dataframe(sim_df.tail(200))
