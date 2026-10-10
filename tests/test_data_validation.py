"""
Unit tests for src/data_validation.py using standard unittest library.
"""

import unittest
import pandas as pd
import numpy as np
from src.data_validation import validate_raw_schema, audit_data_quality, add_derived_target
from src.config import RAW_DATA_PATH, RAW_TARGET_COL, DERIVED_TARGET_COL


class TestDataValidation(unittest.TestCase):

    def setUp(self):
        self.raw_df = pd.read_csv(RAW_DATA_PATH)

    def test_validate_raw_schema(self):
        valid, msg = validate_raw_schema(self.raw_df)
        self.assertTrue(valid)
        self.assertEqual(msg, "Schema validation passed.")

    def test_audit_data_quality(self):
        audit = audit_data_quality(self.raw_df)
        self.assertEqual(audit["total_rows"], 30000)
        self.assertEqual(audit["total_columns"], 56)
        self.assertEqual(audit["missing_values_count"], 0)
        self.assertEqual(audit["complete_duplicate_rows"], 0)
        self.assertEqual(audit["duplicate_filenames"], 0)
        self.assertTrue(audit["labels_valid"])
        self.assertEqual(audit["conflicting_url_labels_count"], 0)
        self.assertEqual(len(audit["invalid_binary_columns"]), 0)

    def test_add_derived_target(self):
        sample_df = pd.DataFrame({
            "FILENAME": ["a.txt", "b.txt"],
            "label": [0, 1]
        })
        res = add_derived_target(sample_df)
        self.assertIn(DERIVED_TARGET_COL, res.columns)
        self.assertEqual(res.loc[res["label"] == 0, DERIVED_TARGET_COL].iloc[0], 1)  # 0 -> Phishing (1)
        self.assertEqual(res.loc[res["label"] == 1, DERIVED_TARGET_COL].iloc[0], 0)  # 1 -> Legitimate (0)


if __name__ == '__main__':
    unittest.main()
