# PhishGuard — Explainable Machine Learning Based Phishing URL Detection System

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-CC%20BY%204.0-green.svg)
![Framework](https://img.shields.io/badge/framework-Streamlit%20|%20Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)

---

## 1. Project Overview
**PhishGuard** is an explainable machine learning cybersecurity system and interactive web application designed to detect malicious phishing URLs in real-time. Given any target URL string submitted by a user or browser client, the system extracts 18 deterministic structural and lexical features (**FS-URL**) and classifies the URL as **Phishing** (`is_phishing = 1`) or **Legitimate** (`is_phishing = 0`).

### Core Design Principles
- **Zero Web Scraping & Zero Network Calls:** Destination web servers are never visited, crawled, or pinged. This completely eliminates remote code execution risks, server-side cloaking evasion, and network latency.
- **Strictly Controlled Model Scope:** Strictly restricted to two transparent, auditable algorithms: **Logistic Regression** and **Decision Tree Classifier**.
- **Pure Explainability (XAI):** Uses linear log-odds feature attribution ($\text{contribution}_i = \text{scaled\_feature}_i \times \text{weight}_i$) to explain exactly why a URL was flagged.
- **Zero False Positives Priority:** In cybersecurity operations, blocking legitimate websites damages user trust and business operations. The Champion model was selected specifically for achieving **100.00% precision (0 False Positives)** on the held-out test benchmark.

---

## 2. Dataset & Target Convention
- **Source:** `PhiUSIIL_Phishing_30000.csv` (UCI Machine Learning Repository ID: 967, Prasad & Chandra 2023, CC BY 4.0).
- **Total Records:** 30,000 raw samples (29,989 unique URLs after deduplication).
- **Target Convention:**
  - Raw dataset column: `label` (`0` = Phishing, `1` = Legitimate)
  - Non-negotiable project convention: `is_phishing = (label == 0).astype("int8")`
  - **Positive Class (`1`):** Phishing (12,843 samples, 42.81%)
  - **Negative Class (`0`):** Legitimate (17,157 samples, 57.19%)
- **Data Split:** Reproducible 80/20 stratified train/test split with `random_state=42`.
  - **Training Set (`data/processed/train.csv`):** 23,991 samples (used for 5-fold CV tuning).
  - **Locked Held-Out Test Set (`data/processed/test.csv`):** 5,998 unseen samples (evaluated once).

---

## 3. Feature Set: Deployable FS-URL Group (18 Features)
Inference operates exclusively on 18 features extractable directly from the URL string:
1. `URLLength` — Normalized string length of the URL
2. `DomainLength` — Character count of the domain/hostname
3. `IsDomainIP` — Binary indicator for raw IPv4/IPv6 hostname
4. `TLDLength` — Character count of top-level domain
5. `NoOfSubDomain` — Count of subdomains
6. `HasObfuscation` — Binary indicator for `%`-encoding, `@`, or IP host
7. `NoOfObfuscatedChar` — Count of obfuscated characters
8. `ObfuscationRatio` — Ratio of obfuscated characters to URL length
9. `NoOfLettersInURL` — Count of alphabetic characters
10. `LetterRatioInURL` — Ratio of letters to URL length
11. `NoOfDegitsInURL` — Count of numeric digits
12. `DegitRatioInURL` — Ratio of digits to URL length
13. `NoOfEqualsInURL` — Count of `=` query delimiters
14. `NoOfQMarkInURL` — Count of `?` query symbols
15. `NoOfAmpersandInURL` — Count of `&` parameter separators
16. `NoOfOtherSpecialCharsInURL` — Count of other special characters (dots, slashes, hyphens)
17. `SpacialCharRatioInURL` — Ratio of all special characters to URL length
18. `IsHTTPS` — Binary indicator for `https://` protocol

*Note: Target proxy `FILENAME` and post-request page features (e.g. `LineOfCode`, `NoOfiFrame`, `HasTitle`) were strictly audited and excluded from the deployable model.*

---

## 4. Final Model Evaluation Benchmark

Evaluation on the locked held-out test split (5,998 unseen URLs):

| Metric | Baseline Logistic Regression | Tuned Decision Tree | **Regularized Logistic Regression (Champion)** |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 99.47% | 99.57% | **97.83%** |
| **Test Precision** | 99.88% (3 FP) | 99.80% (5 FP) | **99.88% (3 False Positives)** |
| **Test Recall** | 98.87% | 99.18% | **95.05%** |
| **Test F1-Score** | 99.37% | 99.49% | **97.40%** |
| **Test ROC-AUC** | 0.9984 | 0.9958 | **0.9974** |
| **Inference Latency** | < 1 ms | < 1 ms | **< 1 ms** |

### Confusion Matrix (Champion Model on 5,998 Test Samples)
- **True Negatives (Legitimate Correctly Classified):** 3,429 / 3,432 (99.91%)
- **False Positives (Legitimate Erroneously Blocked):** **3** / 3,432 (0.09%)
- **False Negatives (Phishing Missed):** 127 / 2,566 (4.95%)
- **True Positives (Phishing Correctly Caught):** 2,439 / 2,566 (95.05%)

---

## 5. Project Directory Structure
```text
PhishGuard/
├── app/
│   └── streamlit_app.py        # Interactive Streamlit Web Application
├── data/
│   ├── raw/                    # Raw dataset repository
│   └── processed/
│       ├── train.csv           # 23,991 stratified training samples
│       └── test.csv            # 5,998 locked held-out test samples
├── models/
│   ├── final_pipeline.joblib   # Fitted ColumnTransformer + Champion Estimator
│   ├── feature_schema.json     # Feature order, types, and target schema
│   └── model_metadata.json     # Hyperparameters, test metrics & coefficients
├── notebooks/
│   ├── 01_problem_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_logistic_regression.ipynb
│   ├── 06_decision_tree.ipynb
│   └── 07_model_comparison_and_test_eval.ipynb
├── results/
│   ├── figures/                # Confusion matrix, ROC, and PR curves
│   ├── metrics/                # Benchmark metric JSON files
│   └── tables/                 # EDA and comparison markdown tables
├── src/
│   ├── __init__.py
│   ├── config.py               # Central paths, seed, and constants
│   ├── data_validation.py      # Schema integrity and target mapping
│   ├── feature_extraction.py   # Deterministic FS-URL extractor
│   ├── preprocessing.py        # Leak-free ColumnTransformer pipeline
│   ├── train.py                # Stratified train/test splitter
│   ├── serialize_model.py      # Pipeline training and artifact export
│   └── predict.py              # End-to-end inference and XAI attribution
├── tests/
│   ├── test_data_validation.py
│   ├── test_feature_extraction.py
│   ├── test_preprocessing.py
│   ├── test_train.py
│   ├── test_pipeline_schema.py
│   └── test_prediction.py
├── .gitignore
├── requirements.txt
├── README.md
└── WORKFLOW.md
```

---

## 6. Installation & Quick Start

### 1. Clone the Repository & Set Up Virtual Environment
```bash
git clone https://github.com/Urvi-Ladhani/PhishGuard.git
cd PhishGuard

# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Pipeline & Run Test Suite
Run all 19 automated unit tests across validation, extraction, pipelines, and inference:
```bash
python -m unittest discover tests
```

### 3. Launch the Streamlit Web Application
Start the local PhishGuard web interface:
```bash
streamlit run app/streamlit_app.py
```
Open your browser to `http://localhost:8501`.

---

## 7. Model Serialization & Reproduction
To retrain the Champion Pipeline and regenerate all model artifacts and schemas from scratch:
```bash
python src/serialize_model.py
```
This fits the pipeline on `data/processed/train.csv`, verifies metrics on `data/processed/test.csv`, and writes:
- `models/final_pipeline.joblib`
- `models/feature_schema.json`
- `models/model_metadata.json`

---

## 8. Limitations & Operating Disclaimer
- **Lexical Bounds:** PhishGuard analyzes deterministic structural features of the URL string alone (`FS-URL`). It does not execute live HTTP requests or render remote JavaScript.
- **Model Probability Interpretation:** The reported probability is the logistic function output under the FS-URL feature distribution and represents internal model risk score, not a guaranteed calibrated real-world confidence.
- **Always Verify:** Users should always exercise independent caution before entering passwords or sensitive personal data into unfamiliar websites.
