"""
Unit tests for pipeline serialization, feature schema, and model metadata.
"""

import unittest
import json
import joblib
from pathlib import Path
from src.config import FINAL_PIPELINE_PATH, FEATURE_SCHEMA_PATH, MODELS_DIR
from src.feature_extraction import FS_URL_FEATURES, FS_URL_NUMERIC, FS_URL_BINARY


class TestPipelineSchema(unittest.TestCase):

    def test_artifacts_exist(self):
        self.assertTrue(FINAL_PIPELINE_PATH.exists(), "final_pipeline.joblib must exist")
        self.assertTrue(FEATURE_SCHEMA_PATH.exists(), "feature_schema.json must exist")
        metadata_path = MODELS_DIR / "model_metadata.json"
        self.assertTrue(metadata_path.exists(), "model_metadata.json must exist")

    def test_schema_feature_order_and_count(self):
        with open(FEATURE_SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema = json.load(f)
            
        self.assertEqual(schema["feature_count"], 18)
        self.assertEqual(schema["feature_order"], FS_URL_FEATURES)
        self.assertEqual(schema["numeric_features"], FS_URL_NUMERIC)
        self.assertEqual(schema["binary_features"], FS_URL_BINARY)
        self.assertEqual(schema["target_column"], "is_phishing")
        self.assertEqual(schema["positive_class"], 1)

    def test_pipeline_load_and_structure(self):
        pipeline = joblib.load(FINAL_PIPELINE_PATH)
        self.assertIn('preprocessor', pipeline.named_steps)
        self.assertIn('model', pipeline.named_steps)
        
        model = pipeline.named_steps['model']
        self.assertEqual(model.C, 0.01)
        self.assertIsNone(model.class_weight)


if __name__ == '__main__':
    unittest.main()
