"""Page 8 — Educational Simulator (What-If Explorer).

An interactive simulation tool demonstrating how logistic regression models
combine age and structural brain volume (nWBV) to produce classification scores.

STRICT DISCLAIMER: FOR EDUCATIONAL SIMULATION ONLY.
NOT A CLINICAL DIAGNOSTIC TOOL.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pages.helpers import (
    load_cohort, missing_data_warning, section_header,
    apply_theme, get_theme_colors, style_figure,
)

apply_theme()
theme = get_theme_colors()

st.header("🎛️ Educational Simulator (What-If Explorer)")

# ── Strict Educational Disclaimer ─────────────────────────────────────────
st.error(
    "### ⚠️ STRICT DISCLAIMER: NOT FOR CLINICAL USE\n"
    "This simulator is provided **strictly for educational demonstration** of logistic "
    "regression mathematics on research data. It is **NOT** a clinical diagnostic tool, "
    "risk assessment calculator, or prognostic instrument. Do not use this tool for "
    "medical evaluation, diagnosis, or treatment decisions.",
    icon="🚨",
)

st.markdown("""
This interactive module demonstrates how structural brain measures (nWBV) modulate 
statistical predictions beyond chronological age alone. Adjust the inputs below to 
observe how each model responds in real time.
""")

st.divider()

cohort = load_cohort()
if cohort is None:
    missing_data_warning("Labelled cohort")
    st.stop()


# ── Train simulation models (cached) ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_simulation_models(data: pd.DataFrame):
    """Fit standard models on the labelled cohort for simulation."""
    models = {}
    scalers = {}
    feature_sets = {
        "Age only": ["Age"],
        "Age + nWBV": ["Age", "nWBV"],
        "Age + nWBV + eTIV": ["Age", "nWBV", "eTIV"],
    }
    y = data["target"].values
    for name, cols in feature_sets.items():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[cols])
        clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        clf.fit(X_scaled, y)
        models[name] = clf
        scalers[name] = (scaler, cols)
    return models, scalers


models, scalers = get_simulation_models(cohort)

# ── Input Sliders ────────────────────────────────────────────────────────
section_header("1. Profile Parameters", "Select demographic and structural brain measurements:")

col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    input_age = st.slider(
        "Age (years)",
        min_value=50,
        max_value=95,
        value=74,
        step=1,
        help="Chronological age at the time of assessment (OASIS-1 cohort range: 50–96).",
    )

with col_in2:
    input_nwbv = st.slider(
        "Normalized Whole-Brain Volume (nWBV)",
        min_value=0.640,
        max_value=0.850,
        value=0.730,
        step=0.005,
        format="%.3f",
        help="Atlas-normalized whole-brain volume ratio (typically decreases with atrophy).",
    )

with col_in3:
    input_etiv = st.slider(
        "Estimated Total Intracranial Volume (eTIV, mm³)",
        min_value=1100,
        max_value=1850,
        value=1450,
        step=10,
        help="Estimated intracranial cavity size in cubic millimetres.",
    )

# ── Model Inferences ──────────────────────────────────────────────────────
st.divider()
section_header("2. Simulated Model Responses", "Comparison of predicted probabilities across feature sets:")

user_df_full = pd.DataFrame([[input_age, input_nwbv, input_etiv]], columns=["Age", "nWBV", "eTIV"])

probs = {}
for name, (scaler, cols) in scalers.items():
    X_sub = user_df_full[cols]
    prob_val = models[name].predict_proba(scaler.transform(X_sub))[0, 1]
    probs[name] = prob_val

col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    p_age = probs["Age only"]
    st.metric(
        label="Model 1: Age Only",
        value=f"{p_age * 100:.1f}%",
        delta="Baseline model",
        delta_color="off",
        help="Predicted probability of CDR > 0 using age as the sole feature.",
    )

with col_res2:
    p_nwbv = probs["Age + nWBV"]
    diff_nwbv = (p_nwbv - p_age) * 100
    st.metric(
        label="Model 2: Age + nWBV",
        value=f"{p_nwbv * 100:.1f}%",
        delta=f"{diff_nwbv:+.1f}% vs. Age only",
        delta_color="inverse" if diff_nwbv > 0 else "normal",
        help="Predicted probability adding normalized whole-brain volume.",
    )

with col_res3:
    p_full = probs["Age + nWBV + eTIV"]
    diff_full = (p_full - p_nwbv) * 100
    st.metric(
        label="Model 3: Age + nWBV + eTIV",
        value=f"{p_full * 100:.1f}%",
        delta=f"{diff_full:+.1f}% vs. Model 2",
        delta_color="off",
        help="Predicted probability including intracranial head size.",
    )

# Quantitative insight
st.caption(
    f"💡 **Observation**: At age **{input_age}**, adjusting nWBV from 0.800 to **{input_nwbv:.3f}** "
    f"shifts the probability estimate by **{diff_nwbv:+.1f}%**, demonstrating the substantive role "
    "of structural volumetric measurements over age alone."
)

st.divider()

# ── Interactive Graph 1: Cohort Placement Map ─────────────────────────────
section_header(
    "3. Cohort Scatter Placement",
    "Where your simulated profile sits within the empirical OASIS-1 distribution:",
)

fig_scatter = px.scatter(
    cohort,
    x="Age",
    y="nWBV",
    color="CDR Group",
    color_discrete_map={"CDR = 0": theme["cdr0"], "CDR > 0": theme["cdr_pos"]},
    opacity=0.6,
    title="OASIS-1 Cohort with Simulated Profile Locus",
)

# Overlay the simulated user profile point
fig_scatter.add_trace(go.Scatter(
    x=[input_age],
    y=[input_nwbv],
    mode="markers+text",
    name="Simulated Profile",
    marker=dict(
        symbol="star",
        size=20,
        color=theme["accent"],
        line=dict(color="#FFFFFF" if theme["plotly_template"] == "plotly_dark" else "#000000", width=2),
    ),
    text=["★ You"],
    textposition="top center",
    textfont=dict(size=14, color=theme["accent"]),
))

# Empirical percentile calculation
nwbv_percentile = (cohort["nWBV"] < input_nwbv).mean() * 100
fig_scatter.add_annotation(
    x=input_age,
    y=input_nwbv,
    text=f"Simulated Profile<br>Age: {input_age}y, nWBV: {input_nwbv:.3f}<br>(Percentile: {nwbv_percentile:.1f}%)",
    showarrow=True,
    arrowhead=2,
    arrowcolor=theme["accent"],
    arrowsize=1.2,
    ax=60,
    ay=-50,
    bgcolor=theme["card_bg"],
    bordercolor=theme["border"],
    borderwidth=1,
)

fig_scatter.update_layout(
    xaxis_title="Age (years)",
    yaxis_title="Normalized Whole-Brain Volume (nWBV)",
)
style_figure(fig_scatter, height=480)
st.plotly_chart(fig_scatter, use_container_width=True)

st.caption(
    f"📊 **Normative Context:** An nWBV of **{input_nwbv:.3f}** is greater than "
    f"**{nwbv_percentile:.1f}%** of all observed OASIS-1 cohort participants."
)

st.divider()

# ── Interactive Graph 2: Dynamic Probability Response Curve ───────────────
section_header(
    "4. Dynamic Volumetric Response Curve",
    f"How predicted probability of CDR > 0 changes across brain volume for an individual aged {input_age}:",
)

nwbv_range = np.linspace(0.640, 0.850, 100)
curve_df = pd.DataFrame({
    "Age": [input_age] * 100,
    "nWBV": nwbv_range,
    "eTIV": [input_etiv] * 100,
})

scaler_2, cols_2 = scalers["Age + nWBV"]
probs_curve_2 = models["Age + nWBV"].predict_proba(scaler_2.transform(curve_df[cols_2]))[:, 1]

scaler_3, cols_3 = scalers["Age + nWBV + eTIV"]
probs_curve_3 = models["Age + nWBV + eTIV"].predict_proba(scaler_3.transform(curve_df[cols_3]))[:, 1]

fig_curve = go.Figure()

# Age only flat baseline
fig_curve.add_trace(go.Scatter(
    x=nwbv_range,
    y=[p_age] * len(nwbv_range),
    mode="lines",
    name="Age Only Baseline",
    line=dict(color=theme["muted"], dash="dot", width=2),
))

# Age + nWBV curve
fig_curve.add_trace(go.Scatter(
    x=nwbv_range,
    y=probs_curve_2,
    mode="lines",
    name="Age + nWBV Model",
    line=dict(color=theme["primary"], width=3),
))

# Age + nWBV + eTIV curve
fig_curve.add_trace(go.Scatter(
    x=nwbv_range,
    y=probs_curve_3,
    mode="lines",
    name="Age + nWBV + eTIV Model",
    line=dict(color=theme["secondary"], dash="dash", width=2.5),
))

# Active selected point marker
fig_curve.add_trace(go.Scatter(
    x=[input_nwbv],
    y=[p_nwbv],
    mode="markers",
    name="Current Selected nWBV",
    marker=dict(size=14, color=theme["accent"], symbol="diamond"),
))

# Vertical line indicator
fig_curve.add_vline(
    x=input_nwbv,
    line_dash="dash",
    line_color=theme["accent"],
    annotation_text=f"Selected: {input_nwbv:.3f}",
    annotation_position="top left",
)

fig_curve.update_layout(
    title=f"Probability Response Curve at Age = {input_age}",
    xaxis_title="Normalized Whole-Brain Volume (nWBV)",
    yaxis_title="Predicted Probability (P(CDR > 0))",
    yaxis_range=[-0.05, 1.05],
)
style_figure(fig_curve, height=450)
st.plotly_chart(fig_curve, use_container_width=True)

st.info(
    "**How to interpret this graph:**  \n"
    "- The horizontal dotted grey line represents what an **Age-only** model outputs (constant because it ignores brain volume).  \n"
    "- The solid blue line shows the **steep sigmoidal gradient** created when nWBV is introduced: lower brain volumes sharply increase the probability score, while preserved brain volumes keep it low even in older individuals.",
    icon="📘",
)
