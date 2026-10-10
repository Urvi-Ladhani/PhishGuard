"""
PhishGuard Evaluation Module
Provides evaluation functions for cross-validation, baseline metrics,
confusion matrix generation, and JSON metrics persistence.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.config import METRICS_DIR, RANDOM_STATE, CV_FOLDS


def evaluate_cv_pipeline(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "baseline_logistic_regression"
) -> Dict[str, Any]:
    """
    Evaluates an sklearn pipeline using 5-fold Stratified Cross-Validation on training data.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    
    metrics_summary = {
        "model_name": model_name,
        "cv_folds": CV_FOLDS,
        "train_samples": len(X_train),
        "metrics": {
            "accuracy": {
                "mean": float(cv_results['test_accuracy'].mean()),
                "std": float(cv_results['test_accuracy'].std())
            },
            "precision": {
                "mean": float(cv_results['test_precision'].mean()),
                "std": float(cv_results['test_precision'].std())
            },
            "recall": {
                "mean": float(cv_results['test_recall'].mean()),
                "std": float(cv_results['test_recall'].std())
            },
            "f1": {
                "mean": float(cv_results['test_f1'].mean()),
                "std": float(cv_results['test_f1'].std())
            },
            "roc_auc": {
                "mean": float(cv_results['test_roc_auc'].mean()),
                "std": float(cv_results['test_roc_auc'].std())
            }
        }
    }
    
    # Save metrics JSON artifact
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = METRICS_DIR / f"{model_name}_cv_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved cross-validation metrics: {json_path}")
    
    return metrics_summary
