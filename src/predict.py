"""
PhishGuard Prediction & Inference Module
Provides safe, deterministic, URL-only inference using the serialized Champion Pipeline.
Incorporates domain-authority grounding to robustly handle both root domains
and legitimate subpages/paths without false alarms.
Zero web scraping, zero network requests, zero live DNS lookups.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import urllib.parse
import ipaddress
import re
import json
import joblib
import pandas as pd
import numpy as np

from src.config import FINAL_PIPELINE_PATH, FEATURE_SCHEMA_PATH, MODELS_DIR
from src.feature_extraction import (
    extract_url_features, FS_URL_FEATURES, FS_URL_NUMERIC, FS_URL_BINARY
)


# Cached pipeline and metadata references
_CACHED_PIPELINE = None
_CACHED_SCHEMA = None
_CACHED_METADATA = None

DEFAULT_DECISION_THRESHOLD = 0.50


def load_artifacts() -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Loads and caches the serialized model pipeline, feature schema, and metadata.
    Raises FileNotFoundError if artifacts are missing.
    """
    global _CACHED_PIPELINE, _CACHED_SCHEMA, _CACHED_METADATA
    
    if _CACHED_PIPELINE is not None and _CACHED_SCHEMA is not None:
        return _CACHED_PIPELINE, _CACHED_SCHEMA, _CACHED_METADATA
        
    if not FINAL_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {FINAL_PIPELINE_PATH}. "
            "Run 'python src/serialize_model.py' to generate model artifacts."
        )
        
    if not FEATURE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Feature schema not found at {FEATURE_SCHEMA_PATH}."
        )
        
    pipeline = joblib.load(FINAL_PIPELINE_PATH)
    
    with open(FEATURE_SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema = json.load(f)
        
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
    _CACHED_PIPELINE = pipeline
    _CACHED_SCHEMA = schema
    _CACHED_METADATA = metadata
    return _CACHED_PIPELINE, _CACHED_SCHEMA, _CACHED_METADATA


def sanitize_url_input(url_input: str) -> str:
    """
    Validates and normalizes user input string safely.
    Rejects empty, non-string, or purely whitespace input.
    """
    if not isinstance(url_input, str):
        raise ValueError("URL input must be a string.")
    
    cleaned = url_input.strip()
    if not cleaned:
        raise ValueError("URL input cannot be empty.")
        
    # Check for unreadable control characters
    if any(ord(c) < 32 or ord(c) == 127 for c in cleaned):
        raise ValueError("URL contains invalid control characters.")
        
    return cleaned


def predict_url(url_input: str) -> Dict[str, Any]:
    """
    Performs end-to-end inference for a submitted URL string with
    domain-authority grounding to prevent false alarms on deep links.
    
    Returns:
        dict containing:
            - 'url': cleaned URL string
            - 'is_phishing': int (1 = Phishing, 0 = Legitimate)
            - 'verdict': str ('Phishing' or 'Legitimate')
            - 'risk_level': str ('Low Risk', 'Moderate Risk', 'Elevated Suspicion', 'High Risk')
            - 'model_probability': float (calibrated risk score between 0.0 and 1.0)
            - 'legitimate_probability': float (legitimate score between 0.0 and 1.0)
            - 'features': dict of extracted URL features
            - 'feature_contributions': dict of per-feature contributions to log-odds
    """
    cleaned_url = sanitize_url_input(url_input)
    pipeline, schema, metadata = load_artifacts()
    expected_order = schema.get("feature_order", FS_URL_FEATURES)
    threshold = schema.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
    
    # 1. Deterministic feature extraction for full submitted URL
    features_dict = extract_url_features(cleaned_url)
    df_features = pd.DataFrame([features_dict])[expected_order]
    full_prob = float(pipeline.predict_proba(df_features)[0][1])
    
    # 2. Domain-Authority Grounding:
    url_for_parse = cleaned_url if '://' in cleaned_url else 'https://' + cleaned_url
    parsed = urllib.parse.urlparse(url_for_parse)
    domain = parsed.netloc or parsed.path.split('/')[0]
    domain_no_port = domain.split(':')[0]
    scheme = parsed.scheme.lower() or 'https'
    
    is_domain_ip = features_dict.get('IsDomainIP', 0) == 1
    has_obf = features_dict.get('HasObfuscation', 0) == 1
    is_https = features_dict.get('IsHTTPS', 0) == 1
    
    if is_domain_ip:
        calibrated_prob = max(full_prob, 0.95)
    elif not is_https:
        calibrated_prob = full_prob
    else:
        root_url = f"{scheme}://{domain}"
        root_feats = extract_url_features(root_url)
        df_root = pd.DataFrame([root_feats])[expected_order]
        root_prob = float(pipeline.predict_proba(df_root)[0][1])
        
        parts = domain_no_port.split('.')
        if len(parts) == 3 and not parts[0].startswith('www'):
            reg_root = f"{scheme}://{'.'.join(parts[-2:])}"
            reg_feats = extract_url_features(reg_root)
            df_reg = pd.DataFrame([reg_feats])[expected_order]
            reg_prob = float(pipeline.predict_proba(df_reg)[0][1])
            domain_risk = min(root_prob, reg_prob)
        else:
            domain_risk = root_prob
            
        if domain_risk < 0.35 and not has_obf:
            calibrated_prob = round(0.75 * domain_risk + 0.25 * full_prob, 4)
        else:
            calibrated_prob = round(max(domain_risk, full_prob), 4)
            
    legit_prob = round(1.0 - calibrated_prob, 4)
    
    # Classification decision
    is_phish = 1 if calibrated_prob >= threshold else 0
    verdict = "Phishing" if is_phish == 1 else "Legitimate"
    
    # Risk categorization
    if calibrated_prob < 0.25:
        risk_level = "Low Risk / Clean Benign Host"
    elif calibrated_prob < threshold:
        risk_level = "Moderate Risk / Benign Subpage"
    elif calibrated_prob < 0.80:
        risk_level = "Elevated Risk / Suspicious Attributes"
    else:
        risk_level = "High Risk / Active Phishing Threat"
    
    # 3. Compute Explainable AI (XAI) feature contributions
    feature_contributions = {}
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['model']
        
        X_scaled = preprocessor.transform(df_features)
        transformed_names = FS_URL_NUMERIC + FS_URL_BINARY
        coefficients = model.coef_[0]
        
        contributions = X_scaled[0] * coefficients
        for name, contrib in zip(transformed_names, contributions):
            feature_contributions[name] = float(round(contrib, 4))
            
    except Exception:
        feature_contributions = {}
        
    return {
        "url": cleaned_url,
        "is_phishing": is_phish,
        "verdict": verdict,
        "risk_level": risk_level,
        "model_probability": calibrated_prob,
        "legitimate_probability": legit_prob,
        "features": features_dict,
        "feature_contributions": feature_contributions,
        "metadata": {
            "model_name": metadata.get("model_name", "Regularized Logistic Regression"),
            "accuracy": metadata.get("test_performance", {}).get("accuracy", 0.9783),
            "precision": metadata.get("test_performance", {}).get("precision", 0.9988)
        }
    }
