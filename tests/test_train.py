"""
Unit tests for src/train.py using standard unittest library.
"""

import unittest
import pandas as pd
from src.train import prepare_and_split_data
from src.config import PROCESSED_DATA_DIR, DERIVED_TARGET_COL


class TestTrainDataSplitting(unittest.TestCase):

    def test_prepare_and_split_data(self):
        X_train, X_test, y_train, y_test = prepare_and_split_data(save_splits=True)
        
        total_samples = len(X_train) + len(X_test)
        self.assertEqual(total_samples, 29989)  # Deduplicated on URL
        
        # Verify 80/20 split ratio (within 1 sample due to integer rounding)
        expected_test_len = int(round(29989 * 0.20))
        self.assertAlmostEqual(len(X_test), expected_test_len, delta=2)
        
        # Verify target stratification (phishing ratio in train vs test)
        train_phish_pct = y_train.mean()
        test_phish_pct = y_test.mean()
        self.assertAlmostEqual(train_phish_pct, test_phish_pct, places=3)
        
        # Verify processed files saved
        self.assertTrue((PROCESSED_DATA_DIR / "train.csv").exists())
        self.assertTrue((PROCESSED_DATA_DIR / "test.csv").exists())


if __name__ == '__main__':
    unittest.main()
