import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
)

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, METRICS_DIR, FIGURES_DIR
from src.feature_extraction import FS_URL_NUMERIC, FS_URL_BINARY
from src.preprocessing import get_feature_preprocessor

def main():
    print("=== UNLOCKING HELD-OUT TEST SET FOR FINAL EVALUATION ===")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
    test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test.csv')

    X_train = train_df.drop(columns=['is_phishing'])
    y_train = train_df['is_phishing']

    X_test = test_df.drop(columns=['is_phishing'])
    y_test = test_df['is_phishing']

    print(f"Loaded Train Set: {X_train.shape[0]} rows")
    print(f"Loaded Held-out Test Set: {X_test.shape[0]} rows (LOCKED TEST EVALUATION)")

    preprocessor = get_feature_preprocessor(FS_URL_NUMERIC, FS_URL_BINARY)

    DECISION_THRESHOLD = 0.65

    # 1. Champion Model: Regularized Logistic Regression (C=0.01, threshold=0.65)
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, C=0.01, class_weight=None, solver='lbfgs'))
    ])
    lr_pipeline.fit(X_train, y_train)

    # 2. Challenger Model: Tuned Decision Tree
    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', DecisionTreeClassifier(random_state=RANDOM_STATE, criterion='entropy', max_depth=15, min_samples_split=2, min_samples_leaf=1, class_weight='balanced'))
    ])
    dt_pipeline.fit(X_train, y_train)

    # Predict on test set
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_prob_lr >= DECISION_THRESHOLD).astype(int)

    y_prob_dt = dt_pipeline.predict_proba(X_test)[:, 1]
    y_pred_dt = dt_pipeline.predict(X_test)

    # Metrics for LR
    lr_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_lr)),
        'precision': float(precision_score(y_test, y_pred_lr)),
        'recall': float(recall_score(y_test, y_pred_lr)),
        'f1': float(f1_score(y_test, y_pred_lr)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_lr)),
        'threshold': DECISION_THRESHOLD,
        'regularization_c': 0.01
    }

    # Metrics for DT
    dt_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_dt)),
        'precision': float(precision_score(y_test, y_pred_dt)),
        'recall': float(recall_score(y_test, y_pred_dt)),
        'f1': float(f1_score(y_test, y_pred_dt)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_dt))
    }

    print("\n=== FINAL HELD-OUT TEST SET PERFORMANCE ===")
    print("1. Champion Model: Regularized Logistic Regression (C=0.01, Threshold=0.65)")
    print(f"   Accuracy:  {lr_metrics['accuracy']:.4f}")
    print(f"   Precision: {lr_metrics['precision']:.4f}")
    print(f"   Recall:    {lr_metrics['recall']:.4f}")
    print(f"   F1-Score:  {lr_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {lr_metrics['roc_auc']:.4f}")

    print("\n2. Challenger Model: Tuned Decision Tree")
    print(f"   Accuracy:  {dt_metrics['accuracy']:.4f}")
    print(f"   Precision: {dt_metrics['precision']:.4f}")
    print(f"   Recall:    {dt_metrics['recall']:.4f}")
    print(f"   F1-Score:  {dt_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {dt_metrics['roc_auc']:.4f}")

    # Serialize final test metrics
    final_payload = {
        'champion_model': 'Tuned Logistic Regression',
        'test_samples': len(X_test),
        'logistic_regression_test_metrics': lr_metrics,
        'decision_tree_test_metrics': dt_metrics,
        'classification_report_lr': classification_report(y_test, y_pred_lr, output_dict=True)
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    json_path = METRICS_DIR / 'final_test_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nSaved final test metrics JSON: {json_path}")

    # --- PLOTS ---
    # 1. Confusion Matrix (LR & DT)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_dt = confusion_matrix(y_test, y_pred_dt)

    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=['Legitimate (0)', 'Phishing (1)'],
                yticklabels=['Legitimate (0)', 'Phishing (1)'])
    axes[0].set_title('Tuned Logistic Regression (Champion)\nTest Confusion Matrix', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
                xticklabels=['Legitimate (0)', 'Phishing (1)'],
                yticklabels=['Legitimate (0)', 'Phishing (1)'])
    axes[1].set_title('Tuned Decision Tree\nTest Confusion Matrix', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrix.png', dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot: {FIGURES_DIR / 'confusion_matrix.png'}")

    # 2. ROC Curve
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_metrics["roc_auc"]:.4f})', color='#2980b9', lw=2)
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {dt_metrics["roc_auc"]:.4f})', color='#27ae60', lw=2, linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Held-Out Test Set', fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'roc_curve.png', dpi=300)
    plt.close()
    print(f"Saved ROC curve plot: {FIGURES_DIR / 'roc_curve.png'}")

    # 3. Precision-Recall Curve
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
    prec_dt, rec_dt, _ = precision_recall_curve(y_test, y_prob_dt)

    plt.figure(figsize=(8, 6))
    plt.plot(rec_lr, prec_lr, label=f'Logistic Regression (F1 = {lr_metrics["f1"]:.4f})', color='#2980b9', lw=2)
    plt.plot(rec_dt, prec_dt, label=f'Decision Tree (F1 = {dt_metrics["f1"]:.4f})', color='#27ae60', lw=2, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Held-Out Test Set', fontweight='bold')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'precision_recall_curve.png', dpi=300)
    plt.close()
    print(f"Saved PR curve plot: {FIGURES_DIR / 'precision_recall_curve.png'}")

if __name__ == '__main__':
    main()
