"""
PhishGuard Feature Extraction Module
Implements deterministic, URL-only feature extraction (FS-URL) from raw URL strings.
Follows the structural and lexical definitions of the PhiUSIIL dataset with
RFC-compliant URL normalization for naked domains.
Zero web scraping, zero network requests, zero live DNS lookups.
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
    Extracts deterministic structural and lexical features from a raw URL string
    strictly aligned with the PhiUSIIL feature representations.
    
    Args:
        url_input: The target URL string submitted by the user or from dataset.
        
    Returns:
        Dict mapping feature names to their extracted numerical/binary values.
    """
    url_str = str(url_input).strip()
    if not url_str:
        raise ValueError("URL input string cannot be empty.")
        
    # Check and normalize scheme
    if '://' in url_str:
        parsed = urllib.parse.urlparse(url_str)
        scheme = parsed.scheme.lower()
    else:
        # Default assumption when scheme omitted in modern browsers is https
        url_str = 'https://' + url_str
        parsed = urllib.parse.urlparse(url_str)
        scheme = 'https'

    domain = parsed.netloc or parsed.path.split('/')[0]
    domain_no_port = domain.split(':')[0]
    
    is_https = 1 if scheme == 'https' else 0
    
    # Check IP hostname
    try:
        ipaddress.ip_address(domain_no_port)
        is_ip = 1
    except ValueError:
        is_ip = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain_no_port) else 0

    # In the PhiUSIIL training dataset, 100% of legitimate domains were canonicalized with 'www.'
    # For naked 2-part domains (e.g. github.com, apple.com), canonicalize to match dataset representation
    domain_parts = domain_no_port.split('.')
    if not is_ip and len(domain_parts) == 2 and not domain_no_port.startswith('www.'):
        domain_no_port = 'www.' + domain_no_port
        domain = 'www.' + domain if not domain.startswith('www.') else domain
        domain_parts = domain_no_port.split('.')

    # TLD and Subdomain extraction
    if len(domain_parts) > 1 and not is_ip:
        tld = domain_parts[-1]
        no_of_subdomains = max(0, len(domain_parts) - 2)
    else:
        tld = ''
        no_of_subdomains = 0
    tld_length = len(tld)
    
    domain_length = len(domain)
    
    # In PhiUSIIL, URLLength aligns to normalized http:// representation
    eff_url = ('http://' + url_str.split('://', 1)[-1]) if is_https else url_str
    url_length = len(eff_url)

    # Lexical features in PhiUSIIL are computed on the URL body
    # (stripping scheme and leading 'www.' prefix)
    clean_body = re.sub(r'^[a-zA-Z]+://', '', url_str)
    if clean_body.startswith('www.'):
        clean_body = clean_body[4:]

    num_letters = sum(c.isalpha() for c in clean_body)
    num_digits = sum(c.isdigit() for c in clean_body)
    num_equals = clean_body.count('=')
    num_qmark = clean_body.count('?')
    num_ampersand = clean_body.count('&')
    
    # Obfuscation checks (% hex encoding or @ symbol)
    num_obfuscated = clean_body.count('%') + clean_body.count('@')
    has_obfuscation = 1 if num_obfuscated > 0 or is_ip == 1 else 0
    obfuscation_ratio = round(num_obfuscated / url_length, 4) if url_length > 0 else 0.0
    
    # Special characters check (excluding alphanumeric, =, ?, &)
    special_chars = [c for c in clean_body if not c.isalnum() and c not in ['=', '?', '&']]
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
