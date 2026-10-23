"""Page 3 — Brain Metrics."""
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pages.helpers import (
    load_cohort, missing_data_warning, section_header,
    get_theme_colors, style_figure, apply_theme,
)

apply_theme()
theme = get_theme_colors()

st.header("📊 Brain Metrics")

cohort = load_cohort()
if cohort is None:
    missing_data_warning("Labelled cohort")
    st.stop()

# ── Age distributions ────────────────────────────────────────────────────
section_header(
    "Age Distribution by CDR Group",
    "Box plots and overlaid histograms of participant age, split by recorded CDR status.",
)

col1, col2 = st.columns(2)

with col1:
    fig = px.box(
        cohort,
        x="CDR Group",
        y="Age",
        color="CDR Group",
        color_discrete_map={"CDR = 0": theme["cdr0"], "CDR > 0": theme["cdr_pos"]},
        title="Age by CDR Group",
        points="outliers",
    )
    fig.update_layout(showlegend=False, yaxis_title="Age (years)")
    style_figure(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        cohort,
        x="Age",
        color="CDR Group",
        color_discrete_map={"CDR = 0": theme["cdr0"], "CDR > 0": theme["cdr_pos"]},
        barmode="overlay",
        opacity=0.65,
        nbins=25,
        title="Age Histogram by CDR Group",
    )
    fig.update_layout(xaxis_title="Age (years)", yaxis_title="Count")
    style_figure(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)

# Summary stats
st.markdown("**Descriptive statistics:**")
age_desc = cohort.groupby("CDR Group")["Age"].describe().round(1)
st.dataframe(age_desc, use_container_width=True)

st.divider()

# ── nWBV distributions ───────────────────────────────────────────────────
section_header(
    "nWBV Distribution by CDR Group",
    "Normalized whole-brain volume — a structural MRI measure that tends to "
    "decrease with brain atrophy.",
)

col1, col2 = st.columns(2)

with col1:
    fig = px.box(
        cohort,
        x="CDR Group",
        y="nWBV",
        color="CDR Group",
        color_discrete_map={"CDR = 0": theme["cdr0"], "CDR > 0": theme["cdr_pos"]},
        title="nWBV by CDR Group",
        points="outliers",
    )
    fig.update_layout(showlegend=False, yaxis_title="Normalized whole-brain volume")
    style_figure(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        cohort,
        x="nWBV",
        color="CDR Group",
        color_discrete_map={"CDR = 0": theme["cdr0"], "CDR > 0": theme["cdr_pos"]},
        barmode="overlay",
        opacity=0.65,
        nbins=25,
        title="nWBV Histogram by CDR Group",
    )
    fig.update_layout(xaxis_title="nWBV", yaxis_title="Count")
    style_figure(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)

nwbv_desc = cohort.groupby("CDR Group")["nWBV"].describe().round(4)
st.dataframe(nwbv_desc, use_container_width=True)

st.divider()

# ── Age vs nWBV scatter ──────────────────────────────────────────────────
section_header(
    "Age vs. nWBV Correlation",
    "Scatter plot showing the relationship between age and brain volume, "
    "coloured by CDR group.",
)

fig = px.scatter(
    cohort,
    x="Age",
    y="nWBV",
    color="CDR Group",
    color_discrete_map={"CDR = 0": theme["cdr0"], "CDR > 0": theme["cdr_pos"]},
    opacity=0.75,
    title="Age vs. Normalized Whole-Brain Volume",
)

# Add linear trendlines computed via numpy (no statsmodels dependency)
for grp, col_val in [("CDR = 0", theme["cdr0"]), ("CDR > 0", theme["cdr_pos"])]:
    sub = cohort[cohort["CDR Group"] == grp].dropna(subset=["Age", "nWBV"])
    if len(sub) > 1:
        slope, intercept = np.polyfit(sub["Age"], sub["nWBV"], 1)
        x_vals = np.linspace(sub["Age"].min(), sub["Age"].max(), 50)
        y_vals = slope * x_vals + intercept
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=f"{grp} trendline",
            line=dict(color=col_val, dash="dash", width=2),
        ))

fig.update_layout(
    xaxis_title="Age (years)",
    yaxis_title="Normalized whole-brain volume",
)
style_figure(fig, height=480)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Brain volume generally decreases with age. The CDR > 0 group tends to "
    "show lower nWBV at comparable ages, suggesting additional volume loss "
    "beyond what age alone would explain."
)
