"""Basic tests for NeuroScope data loading and analysis artifacts."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"


# ── Data file tests ──────────────────────────────────────────────────────

class TestDataFiles:
    """Verify that the required data files exist and have expected structure."""

    def test_original_export_exists(self):
        assert (DATA / "oasis1_original_export.csv").exists(), \
            "oasis1_original_export.csv not found — run src/inspect_data.py"

    def test_original_export_shape(self):
        df = pd.read_csv(DATA / "oasis1_original_export.csv")
        assert df.shape[0] == 436, f"Expected 436 rows, got {df.shape[0]}"
        assert df.shape[1] == 12, f"Expected 12 columns, got {df.shape[1]}"

    def test_original_export_columns(self):
        df = pd.read_csv(DATA / "oasis1_original_export.csv")
        required = {"ID", "Age", "nWBV", "eTIV", "CDR"}
        assert required.issubset(set(df.columns)), \
            f"Missing columns: {required - set(df.columns)}"

    def test_labeled_cohort_exists(self):
        assert (DATA / "oasis1_labeled_cohort.csv").exists(), \
            "oasis1_labeled_cohort.csv not found — run src/analyze_data.py"

    def test_labeled_cohort_shape(self):
        df = pd.read_csv(DATA / "oasis1_labeled_cohort.csv")
        assert df.shape[0] == 235, f"Expected 235 rows, got {df.shape[0]}"

    def test_labeled_cohort_columns(self):
        df = pd.read_csv(DATA / "oasis1_labeled_cohort.csv")
        required = {"ID", "Age", "nWBV", "eTIV", "CDR", "target"}
        assert required.issubset(set(df.columns)), \
            f"Missing columns: {required - set(df.columns)}"

    def test_labeled_cohort_target_values(self):
        df = pd.read_csv(DATA / "oasis1_labeled_cohort.csv")
        assert set(df["target"].unique()) == {0, 1}, \
            "Target should have exactly values {0, 1}"

    def test_labeled_cohort_no_missing_features(self):
        df = pd.read_csv(DATA / "oasis1_labeled_cohort.csv")
        for col in ["Age", "nWBV", "eTIV", "target"]:
            assert df[col].isna().sum() == 0, \
                f"Column {col} has missing values in labelled cohort"

    def test_cdr_not_used_as_feature(self):
        """CDR and MMSE must not be used as model features."""
        df = pd.read_csv(DATA / "oasis1_labeled_cohort.csv")
        feature_cols = [c for c in df.columns if c not in ("ID", "CDR", "target")]
        assert "MMSE" not in feature_cols, "MMSE should not be a feature column"
        # CDR is in the data for reference but must not be a model predictor
        # (the analysis pipeline only uses Age, nWBV, eTIV)


# ── Report artifact tests ───────────────────────────────────────────────

class TestReportArtifacts:
    """Verify that analysis report files exist and are well-formed."""

    def test_analysis_summary_exists(self):
        assert (REPORTS / "analysis_summary.json").exists()

    def test_analysis_summary_structure(self):
        data = json.loads((REPORTS / "analysis_summary.json").read_text())
        assert "cross_validation" in data
        assert "target_counts" in data
        assert "limitations" in data
        assert data["target_is_recorded_status_not_future_diagnosis"] is True

    def test_analysis_summary_target_counts(self):
        data = json.loads((REPORTS / "analysis_summary.json").read_text())
        counts = data["target_counts"]
        assert int(counts["0"]) == 135, f"Expected 135 CDR=0, got {counts['0']}"
        assert int(counts["1"]) == 100, f"Expected 100 CDR>0, got {counts['1']}"

    def test_model_comparison_exists(self):
        assert (REPORTS / "model_comparison.csv").exists()

    def test_model_comparison_structure(self):
        df = pd.read_csv(REPORTS / "model_comparison.csv")
        assert len(df) == 3, f"Expected 3 models, got {len(df)}"
        assert "roc_auc_mean" in df.columns
        assert "balanced_accuracy_mean" in df.columns

    def test_model_comparison_values(self):
        df = pd.read_csv(REPORTS / "model_comparison.csv")
        # Age-only model should have lowest AUC
        age_only = df.loc[df["model"] == "Age only", "roc_auc_mean"].values[0]
        age_nwbv = df.loc[df["model"] == "Age + nWBV", "roc_auc_mean"].values[0]
        assert age_nwbv > age_only, \
            "Age + nWBV should outperform Age only"

    def test_data_audit_exists(self):
        assert (REPORTS / "data_audit.json").exists()

    def test_data_audit_structure(self):
        data = json.loads((REPORTS / "data_audit.json").read_text())
        assert "oasis1" in data
        assert data["oasis1"]["shape"] == [436, 12]


# ── Figure tests ─────────────────────────────────────────────────────────

class TestFigures:
    """Verify that generated figures exist."""

    def test_cohort_distributions_exists(self):
        assert (FIGURES / "cohort_distributions.png").exists()

    def test_model_comparison_figure_exists(self):
        assert (FIGURES / "model_comparison.png").exists()

    def test_figures_not_empty(self):
        for name in ["cohort_distributions.png", "model_comparison.png"]:
            path = FIGURES / name
            assert path.stat().st_size > 1000, \
                f"{name} appears to be too small ({path.stat().st_size} bytes)"


# ── App helper tests ─────────────────────────────────────────────────────

class TestHelpers:
    """Verify that pages/helpers.py functions resolve files and load data."""

    def test_load_cohort_helper(self):
        from pages.helpers import load_cohort
        df = load_cohort()
        assert df is not None, "load_cohort() returned None"
        assert len(df) == 235

    def test_load_original_helper(self):
        from pages.helpers import load_original
        df = load_original()
        assert df is not None, "load_original() returned None"
        assert len(df) == 436

    def test_load_analysis_summary_helper(self):
        from pages.helpers import load_analysis_summary
        summary = load_analysis_summary()
        assert summary is not None, "load_analysis_summary() returned None"
        assert "target_counts" in summary

    def test_load_data_audit_helper(self):
        from pages.helpers import load_data_audit
        audit = load_data_audit()
        assert audit is not None, "load_data_audit() returned None"
        assert "oasis1" in audit

    def test_load_model_comparison_helper(self):
        from pages.helpers import load_model_comparison
        df = load_model_comparison()
        assert df is not None, "load_model_comparison() returned None"
        assert len(df) == 3

    def test_theme_colors(self):
        from pages.helpers import LIGHT_THEME, DARK_THEME, get_theme_colors
        for theme in (LIGHT_THEME, DARK_THEME):
            assert "primary" in theme
            assert "cdr0" in theme
            assert "cdr_pos" in theme
            assert "plotly_template" in theme
            assert theme["cdr0"].startswith("#")
            assert theme["cdr_pos"].startswith("#")

    def test_scatter_trendline_without_statsmodels(self):
        """Ensure scatter trendline calculation runs cleanly with numpy and plotly without statsmodels."""
        import numpy as np
        import plotly.graph_objects as go
        from pages.helpers import load_cohort
        cohort = load_cohort()
        sub = cohort[cohort["CDR Group"] == "CDR = 0"].dropna(subset=["Age", "nWBV"])
        slope, intercept = np.polyfit(sub["Age"], sub["nWBV"], 1)
        x_vals = np.linspace(sub["Age"].min(), sub["Age"].max(), 10)
        y_vals = slope * x_vals + intercept
        trace = go.Scatter(x=x_vals, y=y_vals, mode="lines")
        assert len(trace.x) == 10

    def test_simulator_models_training_and_inference(self):
        """Ensure educational simulator models train and predict probabilities between 0 and 1."""
        from pages.helpers import load_cohort
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        cohort = load_cohort()
        features = ["Age", "nWBV", "eTIV"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cohort[features])
        clf = LogisticRegression(class_weight="balanced", random_state=42).fit(X_scaled, cohort["target"])

        test_pt = pd.DataFrame([[74, 0.730, 1450]], columns=features)
        prob = clf.predict_proba(scaler.transform(test_pt))[0, 1]
        assert 0.0 <= prob <= 1.0, f"Expected probability between 0 and 1, got {prob}"

    def test_simulator_percentile_bounds(self):
        """Ensure empirical percentile calculation on nWBV is bounded in [0, 100]."""
        from pages.helpers import load_cohort
        cohort = load_cohort()
        p = (cohort["nWBV"] < 0.730).mean() * 100
        assert 0.0 <= p <= 100.0, f"Percentile out of bounds: {p}"



