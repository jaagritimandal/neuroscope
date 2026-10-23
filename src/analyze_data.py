from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
REPORTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

source = DATA / "oasis1_original_export.csv"
if not source.exists():
    raise FileNotFoundError(
        f"{source} not found. Run python src/inspect_data.py first."
    )

df = pd.read_csv(source)

# Keep only rows with an observed CDR label.
df["CDR"] = pd.to_numeric(df["CDR"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["nWBV"] = pd.to_numeric(df["nWBV"], errors="coerce")
df["eTIV"] = pd.to_numeric(df["eTIV"], errors="coerce")

cohort = df.loc[
    df["CDR"].notna() & df["CDR"].isin([0, 0.5, 1, 2]),
    ["ID", "Age", "nWBV", "eTIV", "CDR"]
].copy()

cohort["target"] = (cohort["CDR"] > 0).astype(int)

# The initial three model features are complete in the source audit.
# Explicitly remove any incomplete feature rows rather than silently
# imputing values without a need.
features = ["Age", "nWBV", "eTIV"]
before = len(cohort)
cohort = cohort.dropna(subset=features + ["target"]).copy()
cohort.to_csv(DATA / "oasis1_labeled_cohort.csv", index=False)

print("=" * 65)
print("NEUROSCOPE — OASIS-1 INITIAL ANALYSIS")
print("=" * 65)
print(f"Rows with valid CDR and complete model features: {len(cohort)}")
print(f"Rows excluded after feature completeness check: {before - len(cohort)}")
print("\nTarget counts (0 = CDR 0, 1 = CDR > 0):")
print(cohort["target"].value_counts().sort_index().to_string())
print("\nAge summary by target:")
print(cohort.groupby("target")["Age"].describe().round(2).to_string())
print("\nnWBV summary by target:")
print(cohort.groupby("target")["nWBV"].describe().round(4).to_string())

# Descriptive association tests; these are not causal tests.
group0 = cohort.loc[cohort["target"] == 0]
group1 = cohort.loc[cohort["target"] == 1]

age_test = mannwhitneyu(group0["Age"], group1["Age"], alternative="two-sided")
nwbv_test = mannwhitneyu(group0["nWBV"], group1["nWBV"], alternative="two-sided")
rho, rho_p = spearmanr(cohort["Age"], cohort["nWBV"])

print("\nExploratory statistics:")
print(f"Age group difference, Mann-Whitney U p-value: {age_test.pvalue:.6g}")
print(f"nWBV group difference, Mann-Whitney U p-value: {nwbv_test.pvalue:.6g}")
print(f"Age vs nWBV Spearman rho: {rho:.4f}, p-value: {rho_p:.6g}")

# All models use identical cross-validation splits.
X_models = {
    "Age only": ["Age"],
    "Age + nWBV": ["Age", "nWBV"],
    "Age + nWBV + eTIV": ["Age", "nWBV", "eTIV"],
}
y = cohort["target"].to_numpy()

cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=10, random_state=42
)

results = []
fold_scores = {}

for model_name, columns in X_models.items():
    X = cohort[columns]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )),
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "balanced_accuracy": "balanced_accuracy",
        },
        n_jobs=1,
        error_score="raise",
    )

    fold_scores[model_name] = scores
    results.append({
        "model": model_name,
        "features": ", ".join(columns),
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_sd": float(np.std(scores["test_roc_auc"], ddof=1)),
        "balanced_accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
        "balanced_accuracy_sd": float(
            np.std(scores["test_balanced_accuracy"], ddof=1)
        ),
        "folds": len(scores["test_roc_auc"]),
    })

summary = pd.DataFrame(results)
summary.to_csv(REPORTS / "model_comparison.csv", index=False)

print("\nRepeated stratified 5-fold CV (10 repeats):")
print(summary.round(4).to_string(index=False))

# Figure 1: age and nWBV distributions by recorded CDR-derived group.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
labels = ["CDR = 0", "CDR > 0"]

for ax, column, title, ylabel in [
    (axes[0], "Age", "Age by recorded CDR group", "Age (years)"),
    (axes[1], "nWBV", "nWBV by recorded CDR group", "Normalized whole-brain volume"),
]:
    ax.boxplot(
        [
            cohort.loc[cohort["target"] == 0, column],
            cohort.loc[cohort["target"] == 1, column],
        ],
        tick_labels=labels,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(FIGURES / "cohort_distributions.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Figure 2: mean cross-validation scores. Error bars show fold SD,
# not confidence intervals.
fig, ax = plt.subplots(figsize=(8, 4.8))
x = np.arange(len(summary))
means = summary["roc_auc_mean"].to_numpy()
errors = summary["roc_auc_sd"].to_numpy()

ax.bar(x, means, yerr=errors, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(summary["model"], rotation=12, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("ROC-AUC")
ax.set_title("Model comparison — repeated cross-validation")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURES / "model_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

report = {
    "dataset": "OASIS-1 cross-sectional clinical/morphometry spreadsheet",
    "target_definition": "1 if CDR > 0; 0 if CDR == 0",
    "target_is_recorded_status_not_future_diagnosis": True,
    "rows_with_valid_cdr_before_feature_check": int(before),
    "rows_analyzed": int(len(cohort)),
    "excluded_for_missing_model_features": int(before - len(cohort)),
    "target_counts": {
        str(k): int(v) for k, v in cohort["target"].value_counts().sort_index().items()
    },
    "association_tests": {
        "age_mann_whitney_p": float(age_test.pvalue),
        "nwbv_mann_whitney_p": float(nwbv_test.pvalue),
        "age_nwbv_spearman_rho": float(rho),
        "age_nwbv_spearman_p": float(rho_p),
    },
    "cross_validation": {
        "method": "RepeatedStratifiedKFold",
        "splits": 5,
        "repeats": 10,
        "random_state": 42,
        "results": results,
    },
    "limitations": [
        "Observational analysis; associations do not establish causation.",
        "Recorded CDR status is not a prospective dementia prediction target.",
        "Cross-validation fold variability is not a confidence interval.",
        "Small cohort and selected labelled subset may limit generalizability.",
        "No external validation has been performed.",
    ],
}

(REPORTS / "analysis_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print("\nSaved:")
print("  data/oasis1_labeled_cohort.csv")
print("  reports/model_comparison.csv")
print("  reports/analysis_summary.json")
print("  figures/cohort_distributions.png")
print("  figures/model_comparison.png")
print("\nAnalysis complete.")
