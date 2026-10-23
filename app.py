"""
NeuroScope — Brain Aging & Clinical Dementia Status Explorer

A Streamlit dashboard for exploring the relationship between structural
brain measurements and recorded dementia status in the OASIS-1 dataset.

⚠️  This is an educational and exploratory research tool.
    It is NOT a medical diagnostic system.
"""
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Path helpers — work both from repo root and inside Docker
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------
overview_page    = st.Page("pages/1_overview.py",       title="Overview",              icon="🧠")
cohort_page      = st.Page("pages/2_cohort_explorer.py", title="Cohort Explorer",       icon="👥")
brain_page       = st.Page("pages/3_brain_metrics.py",   title="Brain Metrics",         icon="📊")
stats_page       = st.Page("pages/4_statistical_findings.py", title="Statistical Findings", icon="📈")
model_page       = st.Page("pages/5_model_comparison.py", title="Model Comparison",     icon="🔬")
interpret_page   = st.Page("pages/6_interpretability.py", title="Interpretability",     icon="🔍")
limits_page      = st.Page("pages/7_limitations.py",      title="Limitations & Provenance", icon="📋")

pg = st.navigation([
    overview_page,
    cohort_page,
    brain_page,
    stats_page,
    model_page,
    interpret_page,
    limits_page,
])

from pages.helpers import apply_theme

# ---------------------------------------------------------------------------
# Global sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.markdown("## NeuroScope")
    st.caption(
        "Brain aging & clinical dementia status explorer  \n"
        "Built on [OASIS-1](https://www.oasis-brains.org/) data"
    )
    st.divider()

    st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.get("dark_mode", False),
        key="dark_mode",
        help="Toggle between light and dark theme",
    )

    st.divider()
    st.markdown(
        "> ⚠️ **Not a diagnostic tool.**  \n"
        "> This dashboard is for educational and exploratory research only."
    )

apply_theme()
pg.run()
