"""Shared data loading, styling, and theme helpers for NeuroScope pages."""
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Colour palettes — scientific, accessible, colour-blind-friendly
# ---------------------------------------------------------------------------
LIGHT_THEME = {
    "primary":    "#2E86AB",   # Steel blue
    "secondary":  "#A23B72",   # Muted magenta
    "accent":     "#F18F01",   # Warm amber
    "success":    "#2D936C",   # Sage green
    "danger":     "#C73E1D",   # Muted red
    "text":       "#1A1A2E",   # Near-black
    "muted":      "#6C757D",   # Grey
    "bg":         "#FFFFFF",
    "card_bg":    "#F8F9FA",
    "border":     "#E2E8F0",
    "cdr0":       "#2E86AB",   # Blue — CDR = 0
    "cdr_pos":    "#A23B72",   # Magenta — CDR > 0
    "plot_bg":    "#FFFFFF",
    "paper_bg":   "#FFFFFF",
    "grid":       "#E5E7EB",
    "plotly_template": "plotly_white",
}

DARK_THEME = {
    "primary":    "#38BDF8",   # Sky blue
    "secondary":  "#F472B6",   # Rose pink
    "accent":     "#FBBF24",   # Amber
    "success":    "#34D399",   # Emerald green
    "danger":     "#F87171",   # Red
    "text":       "#F3F4F6",   # Near white
    "muted":      "#9CA3AF",   # Light grey
    "bg":         "#0F172A",   # Slate dark
    "card_bg":    "#1E293B",   # Slate card
    "border":     "#334155",   # Slate border
    "cdr0":       "#38BDF8",   # High-contrast sky blue
    "cdr_pos":    "#F472B6",   # High-contrast rose
    "plot_bg":    "#1E293B",
    "paper_bg":   "#1E293B",
    "grid":       "#334155",
    "plotly_template": "plotly_dark",
}

# Backward compatibility
COLORS = LIGHT_THEME


def is_dark_mode() -> bool:
    """Return whether dark mode is currently active."""
    return st.session_state.get("dark_mode", False)


def get_theme_colors() -> dict:
    """Return current palette depending on dark mode setting."""
    return DARK_THEME if is_dark_mode() else LIGHT_THEME


def apply_theme():
    """Inject CSS to style the app according to current theme."""
    dark = is_dark_mode()
    if dark:
        css = """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #0F172A !important;
            color: #F3F4F6 !important;
        }
        [data-testid="stHeader"] {
            background-color: #0F172A !important;
        }
        [data-testid="stSidebar"] {
            background-color: #1E293B !important;
            border-right: 1px solid #334155 !important;
        }
        [data-testid="stSidebar"] * {
            color: #F3F4F6 !important;
        }
        .stMarkdown, p, span, label, h1, h2, h3, h4, h5, h6 {
            color: #F3F4F6 !important;
        }
        div[data-testid="stMetric"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #9CA3AF !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #38BDF8 !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #9CA3AF !important;
        }
        div[data-testid="stExpander"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }
        hr {
            border-color: #334155 !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        div[data-testid="stMetric"] {
            background-color: #F8F9FA !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #64748B !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #1E293B !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #64748B !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def style_figure(fig, height=None):
    """Apply current theme styling to a Plotly figure."""
    theme = get_theme_colors()
    fig.update_layout(
        template=theme["plotly_template"],
        plot_bgcolor=theme["plot_bg"],
        paper_bgcolor=theme["paper_bg"],
        font=dict(color=theme["text"]),
    )
    fig.update_xaxes(gridcolor=theme["grid"], zerolinecolor=theme["grid"])
    fig.update_yaxes(gridcolor=theme["grid"], zerolinecolor=theme["grid"])
    if height:
        fig.update_layout(height=height)
    return fig


@st.cache_data(show_spinner=False)
def load_cohort() -> pd.DataFrame | None:
    """Load the labelled cohort CSV.  Returns None if the file is missing."""
    path = DATA / "oasis1_labeled_cohort.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Ensure correct types
    for col in ("Age", "nWBV", "eTIV", "CDR"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "target" in df.columns:
        df["target"] = df["target"].astype(int)
    df["CDR Group"] = df["target"].map({0: "CDR = 0", 1: "CDR > 0"})
    return df


@st.cache_data(show_spinner=False)
def load_original() -> pd.DataFrame | None:
    """Load the full original OASIS-1 export."""
    path = DATA / "oasis1_original_export.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_analysis_summary() -> dict | None:
    """Load analysis_summary.json."""
    path = REPORTS / "analysis_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_data_audit() -> dict | None:
    """Load data_audit.json."""
    path = REPORTS / "data_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    """Load model_comparison.csv."""
    path = REPORTS / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def missing_data_warning(artifact_name: str):
    """Show a consistent warning when a required artifact is missing."""
    st.warning(
        f"**{artifact_name}** not found.  \n"
        "Run the analysis pipeline first:  \n"
        "```bash\n"
        "python src/inspect_data.py\n"
        "python src/analyze_data.py\n"
        "```",
        icon="⚠️",
    )


def metric_card(label: str, value: str, help_text: str = ""):
    """Render a styled metric card."""
    st.metric(label=label, value=value, help=help_text or None)


def section_header(title: str, description: str = ""):
    """Render a consistent section header."""
    st.markdown(f"### {title}")
    if description:
        st.caption(description)
