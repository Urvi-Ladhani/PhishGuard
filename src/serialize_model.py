"""
PhishGuard Model Serialization Module
Fits the Champion Model (Regularized Logistic Regression with C=0.01)
on the training dataset, evaluates consistency on the locked test set
with calibrated threshold (0.65), and serializes:
1. models/final_pipeline.joblib
2. models/feature_schema.json
3. models/model_metadata.json
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FINAL_PIPELINE_PATH,
    FEATURE_SCHEMA_PATH, RANDOM_STATE
)
from src.feature_extraction import FS_URL_FEATURES, FS_URL_NUMERIC, FS_URL_BINARY
from src.preprocessing import get_feature_preprocessor

# Calibrated decision threshold to account for dataset homepage bias
DECISION_THRESHOLD = 0.65
REGULARIZATION_C = 0.01


def serialize_champion_model():
    print("=" * 60)
    print("SERIALIZING CALIBRATED CHAMPION MODEL & SCHEMA ARTIFACTS")
    print("=" * 60)
    
    # 1. Load Train and Test Data
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {train_path} or {test_path}. Run src/train.py first."
        )
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df[FS_URL_FEATURES]
    y_train = train_df['is_phishing']
    
    X_test = test_df[FS_URL_FEATURES]
    y_test = test_df['is_phishing']
    
    print(f"Loaded training samples: {len(X_train)}")
    print(f"Loaded locked test samples: {len(X_test)}")
    
    # 2. Build Pipeline with Regularization (C=0.01) to prevent exploding coefficients
    preprocessor = get_feature_preprocessor(FS_URL_NUMERIC, FS_URL_BINARY)
    champion_model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        C=REGULARIZATION_C,
        class_weight=None,
        solver='lbfgs'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', champion_model)
    ])
    
    # 3. Fit Pipeline
    print(f"Fitting champion pipeline on training data (C={REGULARIZATION_C})...")
    pipeline.fit(X_train, y_train)
    
    # 4. Evaluate on Test Data using Calibrated Threshold (0.65)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Test Evaluation Verified (Threshold={DECISION_THRESHOLD}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f} (False Positives: {fp})")
    print(f"  Recall:    {rec:.4f} (False Negatives: {fn})")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # 5. Extract Feature Names and Learned Coefficients
    transformed_feature_names = FS_URL_NUMERIC + FS_URL_BINARY
    fitted_model = pipeline.named_steps['model']
    coefficients = fitted_model.coef_[0]
    intercept = float(fitted_model.intercept_[0])
    
    feat_coef_map = {
        name: float(round(coef, 5))
        for name, coef in zip(transformed_feature_names, coefficients)
    }
    
    # 6. Save Pipeline Joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, FINAL_PIPELINE_PATH)
    print(f"[OK] Pipeline saved to: {FINAL_PIPELINE_PATH}")
    
    # 7. Save Feature Schema
    schema = {
        "version": "2.0",
        "description": "Deterministic URL-only feature schema (FS-URL) with calibrated threshold.",
        "feature_count": len(FS_URL_FEATURES),
        "feature_order": FS_URL_FEATURES,
        "numeric_features": FS_URL_NUMERIC,
        "binary_features": FS_URL_BINARY,
        "target_column": "is_phishing",
        "target_mapping": {
            "0": "Legitimate",
            "1": "Phishing"
        },
        "decision_threshold": DECISION_THRESHOLD,
        "regularization_c": REGULARIZATION_C,
        "positive_class": 1,
        "positive_label": "Phishing",
        "negative_class": 0,
        "negative_label": "Legitimate",
        "extraction_rules": {
            "web_scraping": False,
            "network_requests": False,
            "live_dns": False,
            "deterministic_parsing_only": True
        }
    }
    with open(FEATURE_SCHEMA_PATH, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=4)
    print(f"[OK] Feature schema saved to: {FEATURE_SCHEMA_PATH}")
    
    # 8. Save Model Metadata
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata = {
        "model_name": "PhishGuard Regularized Logistic Regression",
        "model_role": "Champion Model",
        "algorithm": "sklearn.linear_model.LogisticRegression",
        "hyperparameters": {
            "C": REGULARIZATION_C,
            "class_weight": None,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": RANDOM_STATE
        },
        "decision_threshold": DECISION_THRESHOLD,
        "scaling": "sklearn.preprocessing.StandardScaler on numeric, passthrough on binary",
        "feature_order": FS_URL_FEATURES,
        "feature_coefficients": feat_coef_map,
        "model_intercept": round(intercept, 5),
        "test_performance": {
            "test_samples": int(len(X_test)),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "reproducibility": {
            "random_state": RANDOM_STATE,
            "train_test_split": "80/20 stratified",
            "train_samples": int(len(X_train))
        },
        "created_at": datetime.now().isoformat()
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"[OK] Model metadata saved to: {metadata_path}")
    print("=" * 60)
    print("Serialization completed successfully!")


if __name__ == '__main__':
    serialize_champion_model()
