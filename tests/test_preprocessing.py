"""
Unit tests for src/preprocessing.py using standard unittest library.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.preprocessing import EXCLUDED_FEATURES, get_feature_preprocessor, create_modeling_pipeline


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            'FILENAME': ['fn1.txt', 'fn2.txt', 'fn3.txt', 'fn4.txt'],
            'URL': ['http://a.com', 'http://b.com', 'https://c.com', 'http://d.com'],
            'URLLength': [12, 45, 120, 30],
            'DomainLength': [5, 10, 15, 8],
            'IsHTTPS': [0, 0, 1, 0],
            'IsDomainIP': [0, 1, 0, 0],
            'label': [0, 1, 1, 0],
            'is_phishing': [1, 0, 0, 1]
        })

    def test_excluded_features_list(self):
        self.assertIn('FILENAME', EXCLUDED_FEATURES)
        self.assertIn('label', EXCLUDED_FEATURES)
        self.assertIn('is_phishing', EXCLUDED_FEATURES)

    def test_preprocessor_transformation(self):
        num_cols = ['URLLength', 'DomainLength']
        bin_cols = ['IsHTTPS', 'IsDomainIP']
        preprocessor = get_feature_preprocessor(num_cols, bin_cols)
        
        # Fit on sample data
        X_trans = preprocessor.fit_transform(self.sample_df)
        self.assertEqual(X_trans.shape, (4, 4))
        
        # Mean of scaled numeric columns should be approximately 0
        np.testing.assert_almost_equal(X_trans[:, 0].mean(), 0.0, decimal=5)
        np.testing.assert_almost_equal(X_trans[:, 1].mean(), 0.0, decimal=5)
        
        # Binary columns should remain 0 and 1
        np.testing.assert_array_equal(X_trans[:, 2], self.sample_df['IsHTTPS'].values)
        np.testing.assert_array_equal(X_trans[:, 3], self.sample_df['IsDomainIP'].values)

    def test_modeling_pipeline(self):
        num_cols = ['URLLength', 'DomainLength']
        bin_cols = ['IsHTTPS', 'IsDomainIP']
        preprocessor = get_feature_preprocessor(num_cols, bin_cols)
        model = LogisticRegression(random_state=42)
        pipeline = create_modeling_pipeline(preprocessor, model)
        
        X = self.sample_df[num_cols + bin_cols]
        y = self.sample_df['is_phishing']
        
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        self.assertEqual(len(preds), 4)


if __name__ == '__main__':
    unittest.main()
