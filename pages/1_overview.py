"""Page 1 — Overview."""
import streamlit as st
from pages.helpers import (
    load_cohort, load_analysis_summary, load_data_audit,
    missing_data_warning, apply_theme,
)

apply_theme()

st.header("🧠 NeuroScope — Overview")

st.markdown("""
**NeuroScope** investigates the question:

> *Do structural brain measurements provide additional information about
> recorded dementia status beyond chronological age in the OASIS-1 dataset?*

This dashboard presents an **educational, exploratory** analysis — not a
clinical diagnostic system.
""")

st.divider()

# ── Key metrics ──────────────────────────────────────────────────────────
summary = load_analysis_summary()
audit = load_data_audit()
cohort = load_cohort()

if summary is None or audit is None or cohort is None:
    missing_data_warning("Analysis artifacts")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Total OASIS-1 rows",
        f"{audit['oasis1']['shape'][0]:,}",
        help="Rows in the original OASIS-1 cross-sectional spreadsheet.",
    )
with col2:
    st.metric(
        "Labelled cohort",
        f"{summary['rows_analyzed']:,}",
        help="Rows with a valid CDR label and complete model features.",
    )
with col3:
    st.metric(
        "CDR = 0",
        f"{summary['target_counts']['0']:,}",
        help="Participants with no recorded dementia (CDR = 0).",
    )
with col4:
    st.metric(
        "CDR > 0",
        f"{summary['target_counts']['1']:,}",
        help="Participants with recorded dementia (CDR > 0).",
    )

st.divider()

# ── Research context ─────────────────────────────────────────────────────
st.subheader("Research Context")

st.markdown("""
| Aspect | Detail |
|--------|--------|
| **Dataset** | OASIS-1 cross-sectional MRI / clinical data |
| **Target** | Recorded CDR status (0 vs > 0) at time of scan |
| **Features explored** | Age, nWBV (normalized whole-brain volume), eTIV (estimated total intracranial volume) |
| **Analysis** | Logistic regression with repeated stratified 5-fold CV (10 repeats) |
| **Key finding** | Adding nWBV to age improves cross-validated ROC-AUC from ≈ 0.66 to ≈ 0.78 |
""")

st.info(
    "**Important:** The target is *recorded CDR status* at the time of the MRI scan, "
    "not a future dementia diagnosis.  These results should not be interpreted as "
    "predictive of future cognitive decline.",
    icon="ℹ️",
)

st.divider()

# ── How to use this dashboard ────────────────────────────────────────────
st.subheader("How to Navigate")

st.markdown("""
Use the **sidebar** to explore the pages:

| Page | What you'll find |
|------|------------------|
| **Cohort Explorer** | Sample demographics, missingness, and inclusion criteria |
| **Brain Metrics** | Age and nWBV distributions by CDR group |
| **Statistical Findings** | Mann–Whitney U tests, Spearman correlation |
| **Model Comparison** | Cross-validated ROC-AUC and balanced accuracy |
| **Interpretability** | Logistic regression coefficients |
| **Limitations & Provenance** | Data provenance, caveats, and methodology notes |
""")
