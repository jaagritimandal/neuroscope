# 🧠 NeuroScope

**Brain Aging & Clinical Dementia Status Explorer**

> ⚠️ **This is an educational and exploratory research tool, NOT a medical diagnostic system.**

NeuroScope investigates the question:

> *Do structural brain measurements provide additional information about recorded dementia status beyond chronological age in the OASIS-1 dataset?*

[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                   │
│  ┌──────┐ ┌────────┐ ┌───────┐ ┌───────┐ ┌───────────┐ │
│  │Over- │ │Cohort  │ │Brain  │ │Model  │ │Interpret- │ │
│  │view  │ │Explorer│ │Metrics│ │Compare│ │ability    │ │
│  └──┬───┘ └───┬────┘ └───┬───┘ └───┬───┘ └─────┬─────┘ │
│     └─────────┴─────────┴─────────┴───────────┘         │
│                        │                                 │
│              ┌─────────▼─────────┐                       │
│              │  pages/helpers.py │  ← shared loaders     │
│              └─────────┬─────────┘                       │
└────────────────────────┼────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │  data/     │ │  reports/  │ │  figures/  │
   │  CSV files │ │  JSON/CSV  │ │  PNG plots │
   └─────┬──────┘ └─────┬──────┘ └────────────┘
         │               │
         └───────┬───────┘
                 ▼
   ┌─────────────────────────┐
   │  src/inspect_data.py    │  ← data audit
   │  src/analyze_data.py    │  ← analysis pipeline
   └─────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- OASIS-1 Excel workbooks in `~/Downloads/` (for regenerating data)
- OR: pre-generated CSV/JSON artifacts in `data/`, `reports/`, `figures/`

### Installation

```bash
# Clone the repository
git clone https://github.com/jaagritimandal/neuroscope.git
cd neuroscope

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Generate Analysis Results

If you have the OASIS Excel workbooks in `~/Downloads/`:

```bash
python src/inspect_data.py      # Audit and export data
python src/analyze_data.py      # Run analysis pipeline
```

### Launch the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Run Tests

```bash
pytest tests/ -v
```

---

## Data Provenance

| Property | Value |
|----------|-------|
| **Dataset** | OASIS-1 (Open Access Series of Imaging Studies) |
| **Type** | Cross-sectional MRI & clinical data |
| **Participants** | 416 subjects aged 18–96 |
| **Source** | [oasis-brains.org](https://www.oasis-brains.org/) |
| **Citation** | Marcus et al. (2007). *J. Cognitive Neuroscience*, 19(9), 1498–1507 |

Generated data files, reports, and figures are **excluded from Git** via `.gitignore`. The original OASIS Excel workbooks are never modified by this pipeline.

---

## Methodology

### Target Variable

Binary: CDR = 0 (no dementia) vs CDR > 0 (mild to moderate dementia), as **recorded at the time of MRI scan**.

### Models

Three logistic regression models compared via **Repeated Stratified 5-Fold Cross-Validation** (10 repeats, 50 total folds):

| Model | Features | ROC-AUC | Balanced Accuracy |
|-------|----------|---------|-------------------|
| Age only | Age | 0.6572 | 0.6137 |
| Age + nWBV | Age, nWBV | 0.7819 | 0.7102 |
| Age + nWBV + eTIV | Age, nWBV, eTIV | 0.7778 | 0.7077 |

### Key Design Decisions

- **No data leakage**: All preprocessing (imputation, scaling) is fitted inside each CV fold.
- **CDR and MMSE excluded as features**: These would circularly predict the CDR-derived target.
- **OASIS-1 and OASIS-2 kept separate**: Different study designs and populations.
- **Balanced class weights**: To handle the 135:100 class imbalance.

---

## Evaluation & Limitations

- These are **internal cross-validation** estimates, not externally validated performance.
- Fold-to-fold standard deviations are **not confidence intervals**.
- This is an **observational** analysis — associations do not establish causation.
- The target is **recorded CDR status**, not a prospective dementia prediction.
- The small cohort and selected subset may limit generalizability.

---

## Docker

### Build

```bash
docker build -t neuroscope .
```

### Run

```bash
docker run -p 8501:8501 neuroscope
```

Open [http://localhost:8501](http://localhost:8501).

### Health Check

The container includes a health check that polls Streamlit's `/_stcore/health` endpoint every 30 seconds.

---

## AWS EC2 Deployment

### 1. Launch an EC2 Instance

- **AMI**: Amazon Linux 2023 or Ubuntu 22.04
- **Instance type**: `t3.small` or larger
- **Security group**: Allow inbound TCP on port **8501** and SSH (22)

### 2. Install Docker on EC2

```bash
# Amazon Linux 2023
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Ubuntu
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

### 3. Transfer and Run

```bash
# From your local machine, copy project files:
scp -r neuroscope/ ec2-user@<EC2-IP>:~/neuroscope

# On the EC2 instance:
cd ~/neuroscope
docker build -t neuroscope .
docker run -d -p 8501:8501 --restart unless-stopped neuroscope
```

### 4. Access

Navigate to `http://<EC2-Public-IP>:8501` in your browser.

> **Note**: No AWS credentials are required to run the application. The dashboard serves pre-generated analysis artifacts and does not access any AWS services at runtime.

---

## Project Structure

```
neuroscope/
├── app.py                          # Streamlit entry point
├── pages/
│   ├── helpers.py                  # Shared data loaders & utilities
│   ├── 1_overview.py               # Overview page
│   ├── 2_cohort_explorer.py        # Cohort demographics & missingness
│   ├── 3_brain_metrics.py          # Age & nWBV distributions
│   ├── 4_statistical_findings.py   # Mann-Whitney U, Spearman correlation
│   ├── 5_model_comparison.py       # Cross-validated model results
│   ├── 6_interpretability.py       # Logistic regression coefficients
│   ├── 7_limitations.py            # Caveats & data provenance
│   └── 8_simulator.py              # Educational simulator (What-If explorer)
├── src/
│   ├── inspect_data.py             # Data audit script
│   └── analyze_data.py             # Analysis pipeline
├── data/                           # Generated CSVs (git-ignored)
├── reports/                        # JSON/CSV results (git-ignored)
├── figures/                        # PNG plots (git-ignored)
├── tests/
│   └── test_artifacts.py           # Data & artifact tests
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .streamlit/config.toml
├── .gitignore
├── LICENSE
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).
