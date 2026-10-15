"""
PhishGuard Training & Data Splitting Module
Handles reproducible data splitting (80/20 stratified split), dataset deduplication,
and train/test set persistence.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DATA_PATH, RAW_DATA_ENCODING, RAW_TARGET_COL, DERIVED_TARGET_COL,
    PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE
)
from src.feature_extraction import FS_URL_FEATURES
from src.data_validation import add_derived_target


def prepare_and_split_data(
    save_splits: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw CSV, deduplicates on URL string, adds derived target `is_phishing`,
    and executes a reproducible 80/20 stratified train/test split.
    
    Locks the test set into data/processed/test.csv.
    """
    df_raw = pd.read_csv(RAW_DATA_PATH, encoding=RAW_DATA_ENCODING)
    df_target = add_derived_target(df_raw)
    
    # Deduplicate on URL to prevent split leakage
    df_clean = df_target.drop_duplicates(subset=['URL'], keep='first').copy()
    
    X = df_clean[FS_URL_FEATURES]
    y = df_clean[DERIVED_TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    if save_splits:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
        print(f"Saved processed train split: {PROCESSED_DATA_DIR / 'train.csv'} ({len(train_df)} rows)")
        print(f"Saved processed test split: {PROCESSED_DATA_DIR / 'test.csv'} ({len(test_df)} rows)")
        
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    prepare_and_split_data()
