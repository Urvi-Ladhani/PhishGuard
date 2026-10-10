"""
PhishGuard Preprocessing Module
Defines leak-free feature preprocessing pipelines, numerical scaling,
and column transformation pipelines fitted only on training data.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Identifier & leakage features that MUST be excluded from modeling
EXCLUDED_FEATURES = [
    'FILENAME',    # Source dataset identifier / target leakage proxy
    'URL',         # Raw text string (used only for deterministic feature extraction)
    'Domain',      # Raw text string
    'TLD',         # Raw text string
    'Title',       # Raw text string
    'label',       # Raw target column
    'is_phishing'  # Derived target column
]


def get_feature_preprocessor(
    numeric_features: List[str],
    binary_features: List[str],
    use_robust_scaling: bool = False
) -> ColumnTransformer:
    """
    Creates an sklearn ColumnTransformer that applies scaling to numeric features
    and passes binary features through without modification.
    
    All scaling transformations are fitted strictly inside training pipelines.
    """
    scaler = RobustScaler() if use_robust_scaling else StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numeric_features),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop'
    )
    return preprocessor


def create_modeling_pipeline(
    preprocessor: ColumnTransformer,
    model: Any
) -> Pipeline:
    """
    Wraps preprocessor and estimator into an sklearn Pipeline to guarantee
    zero data leakage during cross-validation and evaluation.
    """
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
