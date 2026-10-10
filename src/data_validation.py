"""
PhishGuard Data Validation Module
Provides functions for raw dataset schema validation, missingness checks,
data quality auditing, and target transformation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.config import RAW_TARGET_COL, DERIVED_TARGET_COL


EXPECTED_COLUMNS = [
    'FILENAME', 'URL', 'URLLength', 'Domain', 'DomainLength', 'IsDomainIP', 'TLD',
    'URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb',
    'TLDLength', 'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL', 'IsHTTPS', 'LineOfCode', 'LargestLineLength', 'HasTitle', 'Title',
    'DomainTitleMatchScore', 'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive',
    'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 'NoOfiFrame',
    'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields',
    'HasPasswordField', 'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS',
    'NoOfJS', 'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef', 'label'
]

BINARY_COLUMNS = [
    'IsHTTPS', 'IsDomainIP', 'HasObfuscation', 'HasTitle', 'HasFavicon',
    'Robots', 'IsResponsive', 'HasDescription', 'HasExternalFormSubmit',
    'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField',
    'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo'
]


def validate_raw_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validates that the input dataframe contains all required columns.
    """
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        return False, f"Missing expected columns: {sorted(list(missing_cols))}"
    return True, "Schema validation passed."


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs a comprehensive data quality audit on the raw dataframe.
    Returns audit statistics.
    """
    schema_ok, msg = validate_raw_schema(df)
    if not schema_ok:
        raise ValueError(msg)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_val_count = int(df.isnull().sum().sum())
    dup_rows = int(df.duplicated().sum())
    dup_filenames = int(df['FILENAME'].duplicated().sum())
    dup_urls = int(df['URL'].duplicated().sum())
    
    # Label validity check
    unique_labels = sorted(df[RAW_TARGET_COL].unique().tolist())
    valid_labels = unique_labels == [0, 1]
    
    # Binary columns integrity
    invalid_binary = {}
    for col in BINARY_COLUMNS:
        invalid = set(df[col].unique()) - {0, 1}
        if invalid:
            invalid_binary[col] = list(invalid)
            
    # Check duplicate URLs for conflicting labels
    dup_url_mask = df.duplicated(subset=['URL'], keep=False)
    dup_urls_df = df[dup_url_mask]
    conflicting_urls_count = 0
    if len(dup_urls_df) > 0:
        grouped = dup_urls_df.groupby('URL')[RAW_TARGET_COL].nunique()
        conflicting_urls_count = int((grouped > 1).sum())

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "missing_values_count": missing_val_count,
        "complete_duplicate_rows": dup_rows,
        "duplicate_filenames": dup_filenames,
        "duplicate_urls": dup_urls,
        "conflicting_url_labels_count": conflicting_urls_count,
        "unique_raw_labels": unique_labels,
        "labels_valid": valid_labels,
        "invalid_binary_columns": invalid_binary
    }


def add_derived_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived target column `is_phishing` according to non-negotiable rule:
    label = 0 -> is_phishing = 1 (Phishing)
    label = 1 -> is_phishing = 0 (Legitimate)
    """
    if RAW_TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{RAW_TARGET_COL}' not found in dataframe.")
    
    df_copy = df.copy()
    df_copy[DERIVED_TARGET_COL] = (df_copy[RAW_TARGET_COL] == 0).astype("int8")
    return df_copy
