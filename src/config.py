"""
PhishGuard Configuration Module
Centralized configuration for dataset paths, random seed, model parameters, and target conventions.
"""

import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_PATH = BASE_DIR / "PhiUSIIL_Phishing_30000.csv"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model & Artifact Paths
MODELS_DIR = BASE_DIR / "models"
FINAL_PIPELINE_PATH = MODELS_DIR / "final_pipeline.joblib"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

# Results Directory
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"

# Notebooks Directory
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure essential output directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR, TABLES_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Project Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Target Label Convention (Non-negotiable)
# Original dataset: label = 0 (Phishing), label = 1 (Legitimate)
# Derived target: is_phishing = 1 (Phishing), is_phishing = 0 (Legitimate)
RAW_TARGET_COL = "label"
DERIVED_TARGET_COL = "is_phishing"

TARGET_MAPPING = {
    1: "Phishing",
    0: "Legitimate"
}
