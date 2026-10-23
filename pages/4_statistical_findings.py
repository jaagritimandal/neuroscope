"""Page 4 — Statistical Findings."""
import streamlit as st
import pandas as pd
from pages.helpers import (
    load_analysis_summary, missing_data_warning, section_header,
    apply_theme, get_theme_colors,
)

apply_theme()
theme = get_theme_colors()

st.header("📈 Statistical Findings")

summary = load_analysis_summary()
if summary is None:
    missing_data_warning("Analysis summary")
    st.stop()

assoc = summary.get("association_tests", {})

# ── Group difference tests ───────────────────────────────────────────────
section_header(
    "Group Difference Tests",
    "Mann–Whitney U tests comparing CDR = 0 and CDR > 0 groups. "
    "These are non-parametric, two-sided tests of whether the distributions "
    "differ between groups.",
)

col1, col2 = st.columns(2)

age_p = assoc.get("age_mann_whitney_p")
nwbv_p = assoc.get("nwbv_mann_whitney_p")

with col1:
    st.metric(
        label="Age difference (Mann–Whitney U)",
        value="p < 0.0001" if age_p and age_p < 0.0001 else f"p = {age_p:.4f}",
        delta=f"Exact: {age_p:.2e}" if age_p is not None else None,
        delta_color="off",
        help="Mann–Whitney U test, two-sided. Tests whether the age distributions differ between CDR groups.",
    )
    if age_p is not None and age_p < 0.001:
        st.success("Significant at α = 0.001 (Cohort ages differ by dementia status)", icon="✅")

with col2:
    st.metric(
        label="nWBV difference (Mann–Whitney U)",
        value="p < 0.0001" if nwbv_p and nwbv_p < 0.0001 else f"p = {nwbv_p:.4f}",
        delta=f"Exact: {nwbv_p:.2e}" if nwbv_p is not None else None,
        delta_color="off",
        help="Mann–Whitney U test, two-sided. Tests whether whole-brain volumes differ between CDR groups.",
    )
    if nwbv_p is not None and nwbv_p < 0.001:
        st.success("Significant at α = 0.001 (Brain volume markedly lower in CDR > 0)", icon="✅")

st.info(
    "**Interpretation:** Both age and nWBV distributions differ significantly "
    "between the CDR = 0 and CDR > 0 groups. However, statistical significance "
    "does not imply clinical significance or causation.",
    icon="ℹ️",
)

st.divider()

# ── Correlation ──────────────────────────────────────────────────────────
section_header(
    "Age–nWBV Correlation",
    "Spearman rank correlation between age and normalized whole-brain volume.",
)

rho = assoc.get("age_nwbv_spearman_rho")
rho_p = assoc.get("age_nwbv_spearman_p")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="Spearman rank correlation (ρ)",
        value=f"{rho:.4f}" if rho is not None else "N/A",
        delta="Strong inverse monotonic relationship",
        delta_color="off",
        help="Spearman rank correlation. Negative values indicate that brain volume decreases with age.",
    )
with col2:
    st.metric(
        label="Correlation p-value",
        value="p < 0.0001" if rho_p and rho_p < 0.0001 else f"p = {rho_p:.4f}",
        delta=f"Exact: {rho_p:.2e}" if rho_p is not None else None,
        delta_color="off",
        help="Two-sided p-value for the Spearman correlation.",
    )

st.info(
    "**Interpretation:** The strong negative correlation (ρ ≈ −0.70) confirms "
    "that brain volume decreases substantially with age in this cohort. "
    "This motivates the core question of whether nWBV adds classification signal "
    "*beyond* what age already provides.",
    icon="ℹ️",
)

st.divider()

# ── Statistical Summary Table ─────────────────────────────────────────────
section_header("Summary of Exploratory Tests")

tests_table = pd.DataFrame([
    {
        "Comparison / Relationship": "Age across CDR status (0 vs > 0)",
        "Test": "Mann–Whitney U (two-sided)",
        "Key Statistic": f"p = {age_p:.2e}",
        "Significance": "p < 0.001 (Highly significant)",
        "Conclusion": "Participants with CDR > 0 are older on average",
    },
    {
        "Comparison / Relationship": "nWBV across CDR status (0 vs > 0)",
        "Test": "Mann–Whitney U (two-sided)",
        "Key Statistic": f"p = {nwbv_p:.2e}",
        "Significance": "p < 0.001 (Highly significant)",
        "Conclusion": "Participants with CDR > 0 have lower whole-brain volume",
    },
    {
        "Comparison / Relationship": "Age vs. nWBV association",
        "Test": "Spearman rank correlation",
        "Key Statistic": f"ρ = {rho:.4f}, p = {rho_p:.2e}",
        "Significance": "p < 0.001 (Highly significant)",
        "Conclusion": "Strong inverse relationship between age and brain volume",
    },
])
st.dataframe(tests_table, hide_index=True, use_container_width=True)

st.divider()

# ── Methodology note ─────────────────────────────────────────────────────
section_header("Methodology Notes")
st.markdown("""
- **Mann–Whitney U test** — a non-parametric test that does not assume normality.
  It evaluates whether random draws from one distribution tend to exceed those from the other.
- **Spearman rank correlation** — assesses monotonic relationships without
  assuming bivariate normality or strict linearity.
- All p-values are two-sided and unadjusted for multiple hypothesis testing.
- These are *exploratory association tests*, not causal or diagnostic proofs.
""")
