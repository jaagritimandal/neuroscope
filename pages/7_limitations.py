"""Page 7 — Limitations & Data Provenance."""
import streamlit as st
from pages.helpers import load_analysis_summary, section_header, apply_theme

apply_theme()

st.header("📋 Limitations & Data Provenance")

# ── Limitations ──────────────────────────────────────────────────────────
section_header(
    "Limitations",
    "Important caveats for interpreting the results in this dashboard.",
)

summary = load_analysis_summary()
limitations = (
    summary.get("limitations", [])
    if summary
    else [
        "Analysis artifacts not yet generated — run the analysis pipeline first.",
    ]
)

for lim in limitations:
    st.markdown(f"- {lim}")

st.error(
    "**⚠️ This is NOT a medical diagnostic system.**  \n"
    "The results are exploratory and educational.  They must not be used "
    "for clinical decision-making.",
    icon="🚨",
)

st.divider()

# ── Data Provenance ──────────────────────────────────────────────────────
section_header(
    "Data Provenance",
    "Origin and handling of the datasets used in this analysis.",
)

st.markdown("""
### OASIS-1 Dataset

| Property | Value |
|----------|-------|
| **Full name** | Open Access Series of Imaging Studies (OASIS-1) |
| **Type** | Cross-sectional MRI and clinical data |
| **Participants** | 416 subjects aged 18–96 (including 100 with mild to moderate Alzheimer's disease) |
| **Source** | [oasis-brains.org](https://www.oasis-brains.org/) |
| **Citation** | Marcus et al. (2007). *Journal of Cognitive Neuroscience*, 19(9), 1498–1507 |

### Data Processing

1. **Raw data** — Original OASIS-1 spreadsheets downloaded from the OASIS website.
   These are never modified by this pipeline.
2. **Export** — `src/inspect_data.py` reads the Excel workbook and exports
   `data/oasis1_original_export.csv` (a faithful CSV copy for reproducibility).
3. **Labelled cohort** — `src/analyze_data.py` filters to rows with valid CDR
   labels and complete model features, producing
   `data/oasis1_labeled_cohort.csv`.
4. **Analysis** — Statistical tests and cross-validated models run on the
   labelled cohort.  Results are saved in `reports/` and `figures/`.

### Key Variables

| Variable | Description |
|----------|-------------|
| **CDR** | Clinical Dementia Rating (0 = no dementia, 0.5 = very mild, 1 = mild, 2 = moderate) |
| **nWBV** | Normalized whole-brain volume (Atlas scaling factor × brain volume) |
| **eTIV** | Estimated total intracranial volume (mm³) |
| **Age** | Participant age at time of scan (years) |
""")

st.divider()

# ── Methodology ──────────────────────────────────────────────────────────
section_header(
    "Methodology Notes",
    "Details of the analysis pipeline.",
)

st.markdown("""
### Target Variable

The binary target is derived from the **recorded CDR** at the time of the MRI
scan: CDR = 0 → target 0; CDR > 0 → target 1.  This is *not* a prospective
dementia prediction — it represents dementia status as documented during the
clinical assessment.

### Cross-Validation

- **Method:** Repeated Stratified 5-Fold Cross-Validation (10 repeats)
- **Pipeline:** Median imputation → Standard scaling → Logistic regression
  (balanced class weights)
- **All preprocessing is fit inside each fold** — no data leakage
- **CDR and MMSE are excluded from features** — they would circularly predict
  the CDR-derived target
- **OASIS-1 and OASIS-2 analyses are kept strictly separate**

### What This Analysis Does NOT Do

- It does not predict future dementia diagnosis
- It does not establish causal relationships
- It does not provide externally validated performance
- It does not replace clinical assessment
- It does not use deep learning or MRI images directly (only derived metrics)
""")

st.divider()

# ── Reproducibility ──────────────────────────────────────────────────────
section_header(
    "Reproducibility",
    "How to regenerate the analysis results.",
)

st.code(
    "# 1. Place OASIS Excel workbooks in ~/Downloads/\n"
    "# 2. Run the data inspection script:\n"
    "python src/inspect_data.py\n\n"
    "# 3. Run the analysis pipeline:\n"
    "python src/analyze_data.py\n\n"
    "# 4. Launch the dashboard:\n"
    "streamlit run app.py",
    language="bash",
)

st.caption(
    "All generated data files and reports are excluded from Git via `.gitignore`.  "
    "The analysis uses `random_state=42` for reproducible cross-validation splits."
)
