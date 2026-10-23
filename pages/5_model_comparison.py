"""Page 5 — Model Comparison."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pages.helpers import (
    load_model_comparison, load_analysis_summary,
    missing_data_warning, section_header,
    apply_theme, get_theme_colors, style_figure,
)

apply_theme()
theme = get_theme_colors()

st.header("🔬 Model Comparison")

comp = load_model_comparison()
summary = load_analysis_summary()

if comp is None or summary is None:
    missing_data_warning("Model comparison results")
    st.stop()

# ── Methodology ──────────────────────────────────────────────────────────
section_header(
    "Cross-Validation Setup",
    "All models use the same repeated stratified 5-fold cross-validation "
    "(10 repeats, 50 total folds) with identical splits.",
)

cv_info = summary.get("cross_validation", {})
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Folds per repeat", cv_info.get("splits", "N/A"))
with col2:
    st.metric("Repeats", cv_info.get("repeats", "N/A"))
with col3:
    st.metric("Total evaluations", comp.iloc[0]["folds"] if len(comp) > 0 else "N/A")

st.markdown("""
Each model is a **logistic regression** pipeline:
1. Median imputation (fitted inside each fold)
2. Standard scaling (fitted inside each fold)
3. Logistic regression with balanced class weights

All preprocessing is fit *inside* each cross-validation fold — no data leakage.
""")

st.divider()

# ── ROC-AUC comparison ──────────────────────────────────────────────────
section_header(
    "ROC-AUC Comparison",
    "Mean ROC-AUC across 50 folds. Error bars show ±1 SD of fold scores "
    "(not confidence intervals).",
)

bar_colors = [theme["primary"], theme["secondary"], theme["accent"]]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=comp["model"],
    y=comp["roc_auc_mean"],
    error_y=dict(type="data", array=comp["roc_auc_sd"], visible=True),
    marker_color=bar_colors[:len(comp)],
    text=[f"{v:.4f}" for v in comp["roc_auc_mean"]],
    textposition="outside",
))
fig.update_layout(
    title="Mean ROC-AUC by Model",
    yaxis_title="ROC-AUC",
    yaxis_range=[0, 1],
    xaxis_title="",
)
style_figure(fig, height=450)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Balanced accuracy comparison ─────────────────────────────────────────
section_header(
    "Balanced Accuracy Comparison",
    "Mean balanced accuracy across 50 folds. Balanced accuracy adjusts for "
    "class imbalance by averaging recall per class.",
)

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=comp["model"],
    y=comp["balanced_accuracy_mean"],
    error_y=dict(type="data", array=comp["balanced_accuracy_sd"], visible=True),
    marker_color=bar_colors[:len(comp)],
    text=[f"{v:.4f}" for v in comp["balanced_accuracy_mean"]],
    textposition="outside",
))
fig2.update_layout(
    title="Mean Balanced Accuracy by Model",
    yaxis_title="Balanced Accuracy",
    yaxis_range=[0, 1],
    xaxis_title="",
)
style_figure(fig2, height=450)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Results table ────────────────────────────────────────────────────────
section_header("Detailed Results Table")

display_cols = {
    "model": "Model",
    "features": "Features",
    "roc_auc_mean": "ROC-AUC (mean)",
    "roc_auc_sd": "ROC-AUC (SD)",
    "balanced_accuracy_mean": "Bal. Accuracy (mean)",
    "balanced_accuracy_sd": "Bal. Accuracy (SD)",
    "folds": "Total Folds",
}
display = comp.rename(columns=display_cols)
st.dataframe(
    display[list(display_cols.values())].style.format({
        "ROC-AUC (mean)": "{:.4f}",
        "ROC-AUC (SD)": "{:.4f}",
        "Bal. Accuracy (mean)": "{:.4f}",
        "Bal. Accuracy (SD)": "{:.4f}",
    }),
    hide_index=True,
    use_container_width=True,
)

st.divider()

# ── Interpretation ───────────────────────────────────────────────────────
section_header("Interpretation")

st.markdown("""
| Finding | Implication |
|---------|-------------|
| **Age alone** gives ROC-AUC ≈ 0.66 | Age is moderately associated with CDR status |
| **Adding nWBV** improves to ≈ 0.78 | Brain volume adds substantial information beyond age |
| **Adding eTIV** does not improve further | Total intracranial volume contributes little additional signal |
""")

st.warning(
    "**Caution:** These are internal cross-validation estimates, not "
    "externally validated performance figures. The SD values reflect "
    "fold-to-fold variability, not formal confidence intervals.",
    icon="⚠️",
)

st.markdown("""
**What is ROC-AUC?**

The Receiver Operating Characteristic Area Under the Curve measures the
model's ability to distinguish between the two groups across all possible
decision thresholds. A value of 0.5 indicates chance-level discrimination;
1.0 indicates perfect discrimination.

**What is Balanced Accuracy?**

The average of recall (sensitivity) for each class. It accounts for class
imbalance by giving equal weight to each class.
""")
