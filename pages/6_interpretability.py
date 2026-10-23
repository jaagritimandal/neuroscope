"""Page 6 — Interpretability.

Fits a logistic regression on the full labelled cohort to extract
standardized coefficients.  This is for interpretability only — the
cross-validated results on other pages are the proper evaluation.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pages.helpers import (
    load_cohort, missing_data_warning, section_header,
    apply_theme, get_theme_colors, style_figure,
)

apply_theme()
theme = get_theme_colors()

st.header("🔍 Interpretability")

cohort = load_cohort()
if cohort is None:
    missing_data_warning("Labelled cohort")
    st.stop()

st.info(
    "The coefficients below come from fitting a logistic regression to the "
    "**entire labelled cohort** after standardizing features. They illustrate "
    "relative feature importance for interpretability — the *cross-validated* "
    "results on the Model Comparison page are the proper evaluation.",
    icon="ℹ️",
)

# ── Fit interpretability model ───────────────────────────────────────────
features = ["Age", "nWBV", "eTIV"]
X = cohort[features].values
y = cohort["target"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_scaled, y)

coefs = pd.DataFrame({
    "Feature": features,
    "Coefficient (standardized)": model.coef_[0],
    "Abs. Coefficient": np.abs(model.coef_[0]),
}).sort_values("Abs. Coefficient", ascending=True)

st.divider()

# ── Coefficient chart ────────────────────────────────────────────────────
section_header(
    "Standardized Logistic Regression Coefficients",
    "Coefficients after standard-scaling all features. "
    "Positive values push toward CDR > 0; negative values push toward CDR = 0.",
)

bar_colors = [
    theme["cdr_pos"] if c > 0 else theme["cdr0"]
    for c in coefs["Coefficient (standardized)"]
]

fig = go.Figure()
fig.add_trace(go.Bar(
    y=coefs["Feature"],
    x=coefs["Coefficient (standardized)"],
    orientation="h",
    marker_color=bar_colors,
    text=[f"{v:+.3f}" for v in coefs["Coefficient (standardized)"]],
    textposition="outside",
))
fig.update_layout(
    title="Feature Coefficients (Standardized)",
    xaxis_title="Coefficient value",
    yaxis_title="",
)
style_figure(fig, height=350)
fig.update_xaxes(zeroline=True, zerolinecolor=theme["muted"])
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Coefficient table ────────────────────────────────────────────────────
section_header("Coefficient Values")
st.dataframe(
    coefs[["Feature", "Coefficient (standardized)"]].style.format({
        "Coefficient (standardized)": "{:+.4f}",
    }),
    hide_index=True,
    use_container_width=True,
)

intercept_val = model.intercept_[0]
st.caption(f"Intercept: {intercept_val:+.4f}")

st.divider()

# ── Interpretation ───────────────────────────────────────────────────────
section_header("How to Read These Coefficients")

st.markdown(f"""
| Feature | Coefficient | Direction | Interpretation |
|---------|-------------|-----------|----------------|
| **Age** | {coefs.loc[coefs['Feature'] == 'Age', 'Coefficient (standardized)'].values[0]:+.4f} | {"→ CDR > 0" if coefs.loc[coefs['Feature'] == 'Age', 'Coefficient (standardized)'].values[0] > 0 else "→ CDR = 0"} | Older participants are more likely to have CDR > 0 |
| **nWBV** | {coefs.loc[coefs['Feature'] == 'nWBV', 'Coefficient (standardized)'].values[0]:+.4f} | {"→ CDR > 0" if coefs.loc[coefs['Feature'] == 'nWBV', 'Coefficient (standardized)'].values[0] > 0 else "→ CDR = 0"} | Lower brain volume is associated with CDR > 0 |
| **eTIV** | {coefs.loc[coefs['Feature'] == 'eTIV', 'Coefficient (standardized)'].values[0]:+.4f} | {"→ CDR > 0" if coefs.loc[coefs['Feature'] == 'eTIV', 'Coefficient (standardized)'].values[0] > 0 else "→ CDR = 0"} | Total intracranial volume has a minor contribution |
""")

st.warning(
    "**Caution:** These coefficients are from a single model fit on all data "
    "and should not be used to draw causal conclusions. They are shown to "
    "aid understanding of the model's behavior, not to suggest diagnostic value.",
    icon="⚠️",
)
