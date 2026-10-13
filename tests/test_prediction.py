"""
Unit tests for end-to-end URL inference, validation, and explainability.
"""

import unittest
from src.predict import predict_url, sanitize_url_input
from src.feature_extraction import FS_URL_FEATURES


class TestPrediction(unittest.TestCase):

    def test_predict_url_returns_expected_structure(self):
        url = "https://www.google.com"
        res = predict_url(url)
        
        self.assertIn("url", res)
        self.assertIn("is_phishing", res)
        self.assertIn("verdict", res)
        self.assertIn("model_probability", res)
        self.assertIn("legitimate_probability", res)
        self.assertIn("features", res)
        self.assertIn("feature_contributions", res)
        
        self.assertIn(res["verdict"], ["Phishing", "Legitimate"])
        self.assertIn(res["is_phishing"], [0, 1])
        self.assertAlmostEqual(res["model_probability"] + res["legitimate_probability"], 1.0, places=3)
        self.assertEqual(len(res["features"]), 18)

    def test_predict_url_determinism(self):
        url = "http://paypal-verification-alert.com/login.php?user=alert&id=928420"
        res1 = predict_url(url)
        res2 = predict_url(url)
        
        self.assertEqual(res1["is_phishing"], res2["is_phishing"])
        self.assertEqual(res1["model_probability"], res2["model_probability"])
        self.assertEqual(res1["features"], res2["features"])
        self.assertEqual(res1["feature_contributions"], res2["feature_contributions"])

    def test_predict_empty_url_raises_value_error(self):
        with self.assertRaises(ValueError):
            predict_url("")

        with self.assertRaises(ValueError):
            predict_url("   ")

    def test_sanitize_url_control_characters(self):
        with self.assertRaises(ValueError):
            sanitize_url_input("https://google.com/\x00evil")


if __name__ == '__main__':
    unittest.main()
