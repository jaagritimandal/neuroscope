from pathlib import Path
import json
import pandas as pd
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
REPORTS = ROOT / "reports"
DATA = ROOT / "data"

REPORTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

FILES = {
    "oasis1": DOWNLOADS / "oasis_cross-sectional-5708aa0a98d82080.xlsx",
    "oasis1_reliability": DOWNLOADS / "oasis_cross-sectional-reliability-063c8642b909ee76.xlsx",
    "oasis2": DOWNLOADS / "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx",
}

audit = {}

def inspect_frame(name, df):
    print(f"\n{'=' * 75}")
    print(f"{name.upper()}: {df.shape[0]} rows x {df.shape[1]} columns")
    print("=" * 75)
    print("\nColumns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nData types:")
    print(df.dtypes.astype(str).to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print("\nExact duplicate rows:", int(df.duplicated().sum()))

    result = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing": {str(k): int(v) for k, v in df.isna().sum().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }
    return result

for key, path in FILES.items():
    if not path.exists():
        print(f"\nWARNING: File not found: {path}")
        continue

    df = pd.read_excel(path)
    audit[key] = inspect_frame(key, df)

    # Export only for analysis; raw source workbooks remain untouched.
    if key == "oasis1":
        df.to_csv(DATA / "oasis1_original_export.csv", index=False)
        print("\nOASIS-1 ID duplicates:",
              int(df["ID"].duplicated().sum()) if "ID" in df else "ID column missing")

        if "CDR" in df:
            print("\nCDR counts including missing:")
            print(df["CDR"].value_counts(dropna=False).sort_index().to_string())

            cdr = pd.to_numeric(df["CDR"], errors="coerce")
            eligible = cdr.notna()
            print("\nBinary target counts (CDR 0 vs CDR > 0; non-missing CDR):")
            print(pd.Series(
                cdr.loc[eligible].eq(0).map({True: "CDR=0", False: "CDR>0"})
            ).value_counts().to_string())

        for col in ["Age", "nWBV", "eTIV", "ASF", "MMSE", "Educ", "SES"]:
            if col in df:
                values = pd.to_numeric(df[col], errors="coerce")
                print(f"\n{col} summary:")
                print(values.describe().to_string())

    elif key == "oasis1_reliability":
        df.to_csv(DATA / "oasis1_reliability_export.csv", index=False)

    elif key == "oasis2":
        df.to_csv(DATA / "oasis2_separate_export.csv", index=False)
        for col in ["Subject ID", "MRI ID", "Group", "CDR"]:
            if col in df:
                print(f"\n{col} counts:")
                print(df[col].value_counts(dropna=False).head(20).to_string())

        if "Subject ID" in df:
            print("\nOASIS-2 unique participants:",
                  int(df["Subject ID"].nunique()))
            print("Rows beyond one per participant:",
                  int(len(df) - df["Subject ID"].nunique()))

audit["package_availability"] = {
    name: importlib.util.find_spec(name) is not None
    for name in ["pandas", "numpy", "scipy", "sklearn",
                 "streamlit", "matplotlib", "openpyxl"]
}

print("\nPACKAGE AVAILABILITY")
for package, available in audit["package_availability"].items():
    print(f"{package}: {'available' if available else 'NOT FOUND'}")

report_path = REPORTS / "data_audit.json"
report_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

print(f"\nAudit report saved to: {report_path}")
print("Audit complete. Original Excel workbooks were not modified.")
