"""
Unit tests for src/feature_extraction.py using standard unittest library.
"""

import unittest
from src.feature_extraction import extract_url_features, FS_URL_FEATURES, FS_URL_NUMERIC, FS_URL_BINARY


class TestFeatureExtraction(unittest.TestCase):

    def test_fs_url_feature_list_length(self):
        self.assertEqual(len(FS_URL_FEATURES), 18)
        self.assertEqual(len(FS_URL_NUMERIC) + len(FS_URL_BINARY), 18)

    def test_extract_valid_https_url(self):
        url = "https://www.google.com/search?q=test"
        feats = extract_url_features(url)
        
        self.assertEqual(feats["IsHTTPS"], 1)
        self.assertEqual(feats["IsDomainIP"], 0)
        self.assertEqual(feats["URLLength"], len(url))
        self.assertEqual(feats["DomainLength"], len("www.google.com"))
        self.assertEqual(feats["NoOfSubDomain"], 1)  # www
        self.assertEqual(feats["NoOfEqualsInURL"], 1)
        self.assertEqual(feats["NoOfQMarkInURL"], 1)

    def test_extract_http_ip_url(self):
        url = "http://192.168.1.1/login.php"
        feats = extract_url_features(url)
        
        self.assertEqual(feats["IsHTTPS"], 0)
        self.assertEqual(feats["IsDomainIP"], 1)
        self.assertEqual(feats["HasObfuscation"], 1)
        self.assertEqual(feats["NoOfSubDomain"], 0)
        self.assertEqual(feats["NoOfDegitsInURL"], 8)  # '19216811' has 8 digits

    def test_extract_obfuscated_url(self):
        url = "http://user@example.com/%20admin"
        feats = extract_url_features(url)
        
        self.assertEqual(feats["HasObfuscation"], 1)
        self.assertEqual(feats["NoOfObfuscatedChar"], 2)  # @ and %

    def test_empty_url_raises_value_error(self):
        with self.assertRaises(ValueError):
            extract_url_features("")


if __name__ == '__main__':
    unittest.main()
