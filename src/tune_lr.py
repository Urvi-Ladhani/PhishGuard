import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, CV_FOLDS, METRICS_DIR
from src.feature_extraction import FS_URL_NUMERIC, FS_URL_BINARY
from src.preprocessing import get_feature_preprocessor

def main():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
    X_train = train_df.drop(columns=['is_phishing'])
    y_train = train_df['is_phishing']

    print(f"Loaded training dataset for Step 10: {X_train.shape[0]} rows, {X_train.shape[1]} features.", flush=True)

    preprocessor = get_feature_preprocessor(FS_URL_NUMERIC, FS_URL_BINARY)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver='lbfgs', penalty='l2'))
    ])

    param_grid = {
        'model__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'model__class_weight': [None, 'balanced']
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        'f1': make_scorer(f1_score),
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'roc_auc': make_scorer(roc_auc_score)
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit='f1',
        n_jobs=1,
        return_train_score=False
    )

    print("Running Logistic Regression GridSearchCV (lbfgs solver)...", flush=True)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_idx = grid_search.best_index_
    res = grid_search.cv_results_

    print("\n=== LOGISTIC REGRESSION GRID SEARCH RESULTS (5-FOLD CV ON TRAIN SET) ===", flush=True)
    print("Best Parameters:", best_params, flush=True)
    print("Best CV F1-Score: ", round(res['mean_test_f1'][best_idx], 4), "+/-", round(res['std_test_f1'][best_idx], 4), flush=True)
    print("Best CV Accuracy: ", round(res['mean_test_accuracy'][best_idx], 4), "+/-", round(res['std_test_accuracy'][best_idx], 4), flush=True)
    print("Best CV Precision:", round(res['mean_test_precision'][best_idx], 4), "+/-", round(res['std_test_precision'][best_idx], 4), flush=True)
    print("Best CV Recall:   ", round(res['mean_test_recall'][best_idx], 4), "+/-", round(res['std_test_recall'][best_idx], 4), flush=True)
    print("Best CV ROC-AUC:  ", round(res['mean_test_roc_auc'][best_idx], 4), "+/-", round(res['std_test_roc_auc'][best_idx], 4), flush=True)

    metrics_payload = {
        'model_name': 'tuned_logistic_regression',
        'best_params': best_params,
        'cv_folds': CV_FOLDS,
        'metrics': {
            'f1': {'mean': float(res['mean_test_f1'][best_idx]), 'std': float(res['std_test_f1'][best_idx])},
            'accuracy': {'mean': float(res['mean_test_accuracy'][best_idx]), 'std': float(res['std_test_accuracy'][best_idx])},
            'precision': {'mean': float(res['mean_test_precision'][best_idx]), 'std': float(res['std_test_precision'][best_idx])},
            'recall': {'mean': float(res['mean_test_recall'][best_idx]), 'std': float(res['std_test_recall'][best_idx])},
            'roc_auc': {'mean': float(res['mean_test_roc_auc'][best_idx]), 'std': float(res['std_test_roc_auc'][best_idx])}
        }
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = METRICS_DIR / 'tuned_logistic_regression_cv_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\nSaved tuned CV metrics JSON: {json_path}", flush=True)

if __name__ == '__main__':
    main()
