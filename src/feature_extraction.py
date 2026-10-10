"""
PhishGuard Feature Extraction Module
Implements deterministic, URL-only feature extraction (FS-URL) from raw URL strings.
Zero web scraping or live network fetching is performed.
"""

import urllib.parse
import re
import ipaddress
from typing import Dict, Any, List


# Deterministic URL-only feature list (FS-URL deployable group)
FS_URL_FEATURES = [
    'URLLength',
    'DomainLength',
    'IsDomainIP',
    'TLDLength',
    'NoOfSubDomain',
    'HasObfuscation',
    'NoOfObfuscatedChar',
    'ObfuscationRatio',
    'NoOfLettersInURL',
    'LetterRatioInURL',
    'NoOfDegitsInURL',
    'DegitRatioInURL',
    'NoOfEqualsInURL',
    'NoOfQMarkInURL',
    'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL',
    'IsHTTPS'
]

# Continuous numeric features requiring scaling in FS-URL
FS_URL_NUMERIC = [
    'URLLength',
    'DomainLength',
    'TLDLength',
    'NoOfSubDomain',
    'NoOfObfuscatedChar',
    'ObfuscationRatio',
    'NoOfLettersInURL',
    'LetterRatioInURL',
    'NoOfDegitsInURL',
    'DegitRatioInURL',
    'NoOfEqualsInURL',
    'NoOfQMarkInURL',
    'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL'
]

# Binary features in FS-URL (no scaling required)
FS_URL_BINARY = [
    'IsDomainIP',
    'HasObfuscation',
    'IsHTTPS'
]


def extract_url_features(url_input: str) -> Dict[str, Any]:
    """
    Extracts deterministic structural and lexical features from a raw URL string.
    
    Args:
        url_input: The target URL string submitted by the user or from dataset.
        
    Returns:
        Dict mapping feature names to their extracted numerical/binary values.
    """
    url_str = str(url_input).strip()
    if not url_str:
        raise ValueError("URL input string cannot be empty.")
        
    parsed = urllib.parse.urlparse(url_str)
    if not parsed.scheme:
        # If no scheme is provided, parse as http
        parsed = urllib.parse.urlparse('http://' + url_str)
        scheme = ''
    else:
        scheme = parsed.scheme.lower()

    domain = parsed.netloc or parsed.path.split('/')[0]
    domain_no_port = domain.split(':')[0]
    
    url_length = len(url_str)
    domain_length = len(domain)
    
    # Check IP hostname
    try:
        ipaddress.ip_address(domain_no_port)
        is_ip = 1
    except ValueError:
        is_ip = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain_no_port) else 0

    # TLD extraction
    domain_parts = domain_no_port.split('.')
    if len(domain_parts) > 1 and not is_ip:
        tld = domain_parts[-1]
        no_of_subdomains = max(0, len(domain_parts) - 2)
    else:
        tld = ''
        no_of_subdomains = 0
    tld_length = len(tld)
    
    is_https = 1 if scheme == 'https' else 0
    
    num_letters = sum(c.isalpha() for c in url_str)
    num_digits = sum(c.isdigit() for c in url_str)
    num_equals = url_str.count('=')
    num_qmark = url_str.count('?')
    num_ampersand = url_str.count('&')
    
    # Obfuscation checks (% hex encoding or @ symbol)
    num_obfuscated = url_str.count('%') + url_str.count('@')
    has_obfuscation = 1 if num_obfuscated > 0 or is_ip == 1 else 0
    obfuscation_ratio = round(num_obfuscated / url_length, 4) if url_length > 0 else 0.0
    
    # Special characters check (excluding alphanumeric, =, ?, &)
    special_chars = [c for c in url_str if not c.isalnum() and c not in ['=', '?', '&']]
    num_other_special = len(special_chars)
    special_ratio = round((num_equals + num_qmark + num_ampersand + num_other_special) / url_length, 4) if url_length > 0 else 0.0
    
    letter_ratio = round(num_letters / url_length, 4) if url_length > 0 else 0.0
    digit_ratio = round(num_digits / url_length, 4) if url_length > 0 else 0.0

    return {
        'URLLength': url_length,
        'DomainLength': domain_length,
        'IsDomainIP': is_ip,
        'TLDLength': tld_length,
        'NoOfSubDomain': no_of_subdomains,
        'HasObfuscation': has_obfuscation,
        'NoOfObfuscatedChar': num_obfuscated,
        'ObfuscationRatio': obfuscation_ratio,
        'NoOfLettersInURL': num_letters,
        'LetterRatioInURL': letter_ratio,
        'NoOfDegitsInURL': num_digits,
        'DegitRatioInURL': digit_ratio,
        'NoOfEqualsInURL': num_equals,
        'NoOfQMarkInURL': num_qmark,
        'NoOfAmpersandInURL': num_ampersand,
        'NoOfOtherSpecialCharsInURL': num_other_special,
        'SpacialCharRatioInURL': special_ratio,
        'IsHTTPS': is_https
    }
