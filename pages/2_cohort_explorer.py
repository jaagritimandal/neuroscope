"""Page 2 — Cohort Explorer."""
import streamlit as st
import pandas as pd
import plotly.express as px
from pages.helpers import (
    load_cohort, load_original, load_data_audit,
    missing_data_warning, section_header,
    apply_theme, get_theme_colors, style_figure,
)

apply_theme()
theme = get_theme_colors()

st.header("👥 Cohort Explorer")

# ── Full dataset overview ────────────────────────────────────────────────
section_header(
    "OASIS-1 Full Dataset",
    "Overview of the complete OASIS-1 cross-sectional spreadsheet before filtering.",
)

audit = load_data_audit()
original = load_original()
cohort = load_cohort()

if audit is None or original is None:
    missing_data_warning("Data audit / original export")
    st.stop()

oasis1 = audit.get("oasis1", {})

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", f"{oasis1['shape'][0]:,}")
with col2:
    st.metric("Columns", f"{oasis1['shape'][1]:,}")
with col3:
    st.metric("Duplicate rows", oasis1.get("exact_duplicate_rows", "N/A"))

st.divider()

# ── Missingness ──────────────────────────────────────────────────────────
section_header("Missing Values", "Number of missing entries per column in the original dataset.")

missing = oasis1.get("missing", {})
if missing:
    miss_df = pd.DataFrame(
        {"Column": list(missing.keys()), "Missing": list(missing.values())}
    )
    miss_df["Present"] = oasis1["shape"][0] - miss_df["Missing"]
    miss_df = miss_df.sort_values("Missing", ascending=False)

    fig = px.bar(
        miss_df,
        x="Column",
        y="Missing",
        color_discrete_sequence=[theme["accent"]],
        title="Missing Values by Column",
        text="Missing",
    )
    fig.update_layout(
        yaxis_title="Count",
        xaxis_title="",
    )
    fig.update_traces(textposition="outside")
    style_figure(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Missingness table"):
        st.dataframe(miss_df[["Column", "Missing", "Present"]], hide_index=True)

st.divider()

# ── Labelled cohort ──────────────────────────────────────────────────────
section_header(
    "Labelled Cohort",
    "Participants with a valid CDR label and complete model features (Age, nWBV, eTIV).",
)

if cohort is None:
    missing_data_warning("Labelled cohort")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Cohort size", f"{len(cohort):,}")
with col2:
    st.metric("CDR = 0", f"{(cohort['target'] == 0).sum():,}")
with col3:
    st.metric("CDR > 0", f"{(cohort['target'] == 1).sum():,}")

# CDR group pie chart
cdr_counts = cohort["CDR Group"].value_counts().reset_index()
cdr_counts.columns = ["Group", "Count"]
fig = px.pie(
    cdr_counts,
    names="Group",
    values="Count",
    color="Group",
    color_discrete_map={"CDR = 0": theme["cdr0"], "CDR > 0": theme["cdr_pos"]},
    title="Cohort Composition",
    hole=0.4,
)
style_figure(fig, height=350)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Sex distribution ─────────────────────────────────────────────────────
section_header("Demographics", "Sex and age distribution in the original dataset.")

if "M/F" in original.columns:
    sex_counts = original["M/F"].value_counts().reset_index()
    sex_counts.columns = ["Sex", "Count"]
    fig = px.bar(
        sex_counts,
        x="Sex",
        y="Count",
        color="Sex",
        color_discrete_map={"M": theme["primary"], "F": theme["secondary"]},
        title="Sex Distribution (Full Dataset)",
        text="Count",
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition="outside")
    style_figure(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)

# ── Data table ───────────────────────────────────────────────────────────
with st.expander("Browse labelled cohort data"):
    st.dataframe(
        cohort[["ID", "Age", "nWBV", "eTIV", "CDR", "CDR Group"]],
        hide_index=True,
        use_container_width=True,
    )
