# PhishGuard --- Explainable Machine Learning Based Phishing URL Detection System

## Overview
PhishGuard is an explainable machine-learning project and Streamlit web application designed to detect phishing URLs. Given a URL string, the system extracts deterministic features (`FS-URL`) and classifies the URL as **Phishing** (`is_phishing = 1`) or **Legitimate** (`is_phishing = 0`).

## Workspace & Dataset Summary
- **Dataset File:** `PhiUSIIL_Phishing_30000.csv` (UCI ML Repository ID: 967, Prasad & Chandra 2023, CC BY 4.0)
- **Total Records:** 30,000 observations
- **Total Attributes:** 56 columns (0 missing values across all columns)
- **Target Mapping:** `is_phishing = (label == 0).astype("int8")`
  - Legitimate (`is_phishing = 0`): 17,157 records (57.19%)
  - Phishing (`is_phishing = 1`): 12,843 records (42.81%)
- **Raw URL Column:** Present (`URL`), enabling training of deterministic URL-only models matching the Streamlit app.

## Project Structure
```text
PhishGuard/
├── .venv/                      # Isolated Python Virtual Environment
├── app/
│   └── streamlit_app.py        # Streamlit Web Application (Step 13)
├── data/
│   ├── raw/                    # Raw dataset storage
│   └── processed/              # Cleaned & processed datasets
├── models/                     # Saved model pipelines & schema artifacts
├── notebooks/
│   ├── 01_problem_data_audit.ipynb  # Audit of Steps 1-6 (Pre-rendered & executed)
│   ├── 02_eda.ipynb            # Exploratory Data Analysis (Step 7)
│   └── 03_feature_engineering.ipynb# Feature Dictionary & Extraction (Step 8)
├── results/
│   ├── figures/                # EDA & Evaluation plots
│   ├── metrics/                # CV & held-out test evaluation JSONs
│   └── tables/                 # Summary tables
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration & paths
│   ├── data_validation.py      # Schema & quality audit validation
│   └── preprocessing.py        # Leak-free ColumnTransformer & Pipeline wrappers
├── tests/
│   ├── test_data_validation.py # Unit tests for data validation
│   └── test_preprocessing.py   # Unit tests for preprocessing pipelines
├── .gitignore
├── requirements.txt
├── README.md
└── WORKFLOW.md                 # Mandatory project specification
```

## Setup & Environment
1. Activate virtual environment or use project `.venv`:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Run unit tests:
   ```powershell
   python -m pytest
   ```
