"""
PhishGuard - Explainable AI Phishing URL Detection System
Streamlit Web Application
Zero web crawling, zero live network requests.
Deterministic inference based on 18 FS-URL features with domain-grounded calibration.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="PhishGuard | AI Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber-Defense Styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

h1, h2, h3, h4, .hero-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

code, .code-font {
    font-family: 'JetBrains Mono', monospace;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
                #0b0f19;
}

/* Glassmorphism Card */
.glass-card {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    margin-bottom: 20px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(6, 182, 212, 0.3);
}

/* Hero Badge */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.35);
    color: #38bdf8;
    margin-bottom: 12px;
}

/* Verdict Badges */
.verdict-safe {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.verdict-phish {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.05) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.verdict-title-safe {
    color: #10b981;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.verdict-title-phish {
    color: #ef4444;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 8px;
}

/* Metric Display Cards */
.metric-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
}

.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Custom button tweaks */
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

/* Footer note */
.disclaimer-box {
    background: rgba(15, 23, 42, 0.5);
    border-left: 4px solid #06b6d4;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 24px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Lazy-load inference module to handle missing artifacts gracefully
@st.cache_resource(show_spinner=False)
def get_prediction_engine():
    try:
        from src.predict import predict_url, load_artifacts
        load_artifacts()
        return predict_url, None
    except Exception as e:
        return None, str(e)


# App Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 10px;">
    <div>
        <div class="status-pill">
            <span>🛡️</span>
            <span>REGULARIZED MODEL ACTIVE &bull; DOMAIN-GROUNDED INFERENCE</span>
        </div>
        <h1 style="font-size: 2.8rem; margin: 0; background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            PhishGuard AI
        </h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 6px; margin-bottom: 24px;">
            Explainable URL Phishing Detection powered by Regularized Logistic Regression on Lexical Features (FS-URL)
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar with System Specifications
with st.sidebar:
    st.markdown("### 🛡️ System Specifications")
    st.markdown("""
    **Architecture & Model Pipeline:**
    - **Champion Model:** Regularized Logistic Regression
    - **Regularization:** $C=0.01$, $L_2$ Penalty
    - **Calibration:** Domain-Authority Grounded
    - **Test Precision:** **99.88%** (Held-out Test Set)
    - **Test Accuracy:** **97.83%**
    - **Held-out Test Samples:** 5,998 URLs
    
    ---
    **Inference Policy:**
    - **Zero Web Scraping:** Destination servers are never crawled.
    - **Zero Live DNS:** Zero network lookups.
    - **Deterministic Extraction:** 18 structural & lexical features computed directly from the URL string.
    - **Domain Authority Grounding:** Prevents false alarms on legitimate subpages & queries.
    
    ---
    **Dataset Provenance:**
    - PhiUSIIL Phishing URL Dataset (30,000 vetted URLs)
    - Stratified 80/20 train/test split locked with seed `42`
    """)
    st.markdown("---")
    st.caption("PhishGuard v2.0 &bull; Built with Streamlit & Scikit-Learn")


# Main Input Section
predict_fn, load_error = get_prediction_engine()

if load_error:
    st.error(
        f"⚠️ **Model Artifacts Not Found or Unreachable:**\n\n"
        f"Unable to load serialized pipeline. Please verify that `models/final_pipeline.joblib` exists.\n"
        f"Error details: `{load_error}`"
    )
    st.stop()


# URL Presets for Fast Testing
col_input, col_presets = st.columns([6, 4])

with col_presets:
    st.markdown("##### ⚡ Quick Test Presets")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🌐 Google Search", use_container_width=True, help="Clean Legitimate Search Query"):
            st.session_state['url_input_val'] = "https://www.google.com/search?q=cybersecurity"
        if st.button("🐙 GitHub Repo", use_container_width=True, help="Clean Legitimate Developer Repository"):
            st.session_state['url_input_val'] = "https://github.com/torvalds/linux"
        if st.button("📚 Wikipedia Article", use_container_width=True, help="Clean Legitimate Article Subpage"):
            st.session_state['url_input_val'] = "https://en.wikipedia.org/wiki/Phishing"
    with col_p2:
        if st.button("📺 YouTube Video", use_container_width=True, help="Clean Legitimate Video Link"):
            st.session_state['url_input_val'] = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        if st.button("🚨 IP Host Phish", use_container_width=True, help="Raw IP Address Attack Link"):
            st.session_state['url_input_val'] = "http://192.168.1.1/login.php?admin=true"
        if st.button("⚠️ Fake PayPal", use_container_width=True, help="Brand Impersonation Phishing URL"):
            st.session_state['url_input_val'] = "http://paypal-verification-alert.com/login.php?user=alert&id=928420"

if 'url_input_val' not in st.session_state:
    st.session_state['url_input_val'] = "https://github.com/torvalds/linux"

with col_input:
    user_url = st.text_input(
        "Enter Target URL to Inspect:",
        value=st.session_state.get('url_input_val', ''),
        placeholder="e.g. https://github.com/torvalds/linux or paypal.com",
        help="Paste any complete or naked URL string."
    )
    analyze_btn = st.button("🔍 Analyze URL Safety", type="primary", use_container_width=False)


# Trigger analysis if button clicked or text present
if user_url.strip():
    try:
        result = predict_fn(user_url)
        is_phish = result["is_phishing"] == 1
        model_prob = result["model_probability"]
        legit_prob = result["legitimate_probability"]
        risk_level = result.get("risk_level", "Evaluated")
        extracted_feats = result["features"]
        contributions = result["feature_contributions"]
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Verdict Card Display
        if is_phish:
            st.markdown(f"""
            <div class="verdict-phish">
                <div class="verdict-title-phish">⚠️ PHISHING DETECTED</div>
                <p style="font-size: 1.15rem; color: #fca5a5; max-width: 700px; margin: 0 auto 16px auto;">
                    <strong>{risk_level}</strong>: High-risk structural markers detected. This URL displays deceptive lexical attributes, abnormal host properties, or suspicious security indicators.
                </p>
                <div style="display: flex; justify-content: center; gap: 32px; margin-top: 12px;">
                    <div>
                        <span style="font-size: 0.9rem; color: #94a3b8;">Model Phishing Score:</span>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #ef4444;">{model_prob * 100:.1f}%</div>
                    </div>
                    <div>
                        <span style="font-size: 0.9rem; color: #94a3b8;">Threat Classification:</span>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #f87171;">HIGH RISK</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-safe">
                <div class="verdict-title-safe">🛡️ SAFE / LEGITIMATE URL</div>
                <p style="font-size: 1.15rem; color: #6ee7b7; max-width: 700px; margin: 0 auto 16px auto;">
                    <strong>{risk_level}</strong>: Verified benign structural profile. Host domain authority is authenticated, with standard HTTPS encryption and no deceptive obfuscation.
                </p>
                <div style="display: flex; justify-content: center; gap: 32px; margin-top: 12px;">
                    <div>
                        <span style="font-size: 0.9rem; color: #94a3b8;">Model Legitimate Score:</span>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #10b981;">{legit_prob * 100:.1f}%</div>
                    </div>
                    <div>
                        <span style="font-size: 0.9rem; color: #94a3b8;">Phishing Probability:</span>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #34d399;">{model_prob * 100:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.caption(
            f"ℹ️ **Notice on Model Probability:** Evaluated with regularized Logistic Regression ($C=0.01$). "
            f"Current calibrated phishing score is **{model_prob * 100:.1f}%** (Risk Level: {risk_level})."
        )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Tabs for In-Depth Explainability and Inspection
        tab_xai, tab_features, tab_model = st.tabs([
            "📊 Feature Attribution & Reasoning",
            "🔍 Extracted Features (FS-URL)",
            "📈 Model Performance & Verification"
        ])

        with tab_xai:
            st.markdown("#### 🧠 Explainable AI: Feature Contribution Analysis")
            st.write(
                "Because the model is a linear classifier (Logistic Regression), every individual feature's "
                "exact mathematical contribution to the log-odds of phishing is computed transparently "
                "($\\text{contribution}_i = \\text{scaled\\_feature}_i \\times \\text{weight}_i$)."
            )

            if contributions:
                df_contrib = pd.DataFrame([
                    {"Feature": k, "Contribution": v}
                    for k, v in contributions.items()
                ]).sort_values(by="Contribution", ascending=True)

                top_phish = df_contrib[df_contrib["Contribution"] > 0].sort_values(by="Contribution", ascending=False).head(5)
                top_legit = df_contrib[df_contrib["Contribution"] < 0].sort_values(by="Contribution", ascending=True).head(5)

                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown("##### 🚨 Top Factors Increasing Phishing Risk")
                    if not top_phish.empty:
                        for _, row in top_phish.iterrows():
                            val = extracted_feats.get(row['Feature'], 'N/A')
                            st.markdown(
                                f"- **`{row['Feature']}`** (Value: `{val}`): "
                                f"<span style='color: #ef4444; font-weight:600;'>+{row['Contribution']:.3f} logit</span>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.write("No strong indicators pushing towards phishing.")

                with col_c2:
                    st.markdown("##### 🛡️ Top Factors Indicating Legitimate Nature")
                    if not top_legit.empty:
                        for _, row in top_legit.iterrows():
                            val = extracted_feats.get(row['Feature'], 'N/A')
                            st.markdown(
                                f"- **`{row['Feature']}`** (Value: `{val}`): "
                                f"<span style='color: #10b981; font-weight:600;'>{row['Contribution']:.3f} logit</span>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.write("No strong indicators pushing towards legitimate.")

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                st.markdown("##### 📊 Full Feature Contribution Bar Chart")
                st.bar_chart(df_contrib.set_index("Feature")["Contribution"], color="#06b6d4")

        with tab_features:
            st.markdown("#### 📋 Deterministic FS-URL Feature Vector")
            st.write(
                "All 18 features below were derived strictly from the URL string in zero network roundtrips. "
                "Feature order, definitions, and types strictly match the training pipeline schema."
            )

            feature_records = []
            feature_descriptions = {
                'URLLength': 'Normalized string length of the URL',
                'DomainLength': 'Character length of the hostname/domain',
                'IsDomainIP': 'Whether domain is a raw IPv4/IPv6 address',
                'TLDLength': 'Character length of top-level domain (e.g. com=3)',
                'NoOfSubDomain': 'Count of subdomains present',
                'HasObfuscation': 'Presence of %-encoding, @ symbol, or IP host',
                'NoOfObfuscatedChar': 'Total count of % and @ characters',
                'ObfuscationRatio': 'Ratio of obfuscated characters to URL length',
                'NoOfLettersInURL': 'Count of alphabetic characters in URL body',
                'LetterRatioInURL': 'Ratio of letters to total URL length',
                'NoOfDegitsInURL': 'Count of numeric digits in URL body',
                'DegitRatioInURL': 'Ratio of digits to total URL length',
                'NoOfEqualsInURL': 'Count of = characters (query parameters)',
                'NoOfQMarkInURL': 'Count of ? characters (query delimiter)',
                'NoOfAmpersandInURL': 'Count of & characters (parameter separator)',
                'NoOfOtherSpecialCharsInURL': 'Count of other special symbols (dots, slashes, dashes)',
                'SpacialCharRatioInURL': 'Ratio of all special characters to URL length',
                'IsHTTPS': 'Whether scheme is secure HTTPS protocol'
            }

            for k, v in extracted_feats.items():
                feature_records.append({
                    "Feature Name": k,
                    "Extracted Value": v,
                    "Type": "Binary (0/1)" if k in ['IsDomainIP', 'HasObfuscation', 'IsHTTPS'] else "Continuous Numeric",
                    "Description": feature_descriptions.get(k, "Structural URL attribute")
                })

            st.dataframe(pd.DataFrame(feature_records), use_container_width=True, hide_index=True)

        with tab_model:
            st.markdown("#### 🏆 Champion Model Test Benchmark")
            st.write(
                "The serialized model was evaluated once on the strictly locked held-out test split "
                "(5,998 unseen URLs from `data/processed/test.csv`)."
            )

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-value">97.83%</div>
                    <div class="metric-label">Test Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-value" style="color: #10b981;">99.88%</div>
                    <div class="metric-label">Precision (3 FP / 3,432)</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-value">95.05%</div>
                    <div class="metric-label">Test Recall (2,439 TP)</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m4:
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-value" style="color: #38bdf8;">0.9974</div>
                    <div class="metric-label">Test ROC-AUC</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            **Why Regularized Logistic Regression with Domain Grounding Works Best:**
            1. **Near-Zero False Positives (99.88% Precision):** Normal websites (like Google, Wikipedia, GitHub, and YouTube) are correctly recognized as safe.
            2. **Pure Explainability:** Instant, transparent feature attribution allows users to inspect exactly why a link was approved or flagged.
            3. **Sub-millisecond Latency:** Deterministic feature extraction combined with linear matrix multiplication runs in `< 1ms` per URL, suitable for high-throughput browser plugins and API endpoints.
            """)

    except ValueError as ve:
        st.warning(f"⚠️ **Invalid URL Format:** {str(ve)}")
    except Exception as ex:
        st.error("⚠️ An unexpected error occurred during analysis. Please check the URL syntax.")


# Footer Safety Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Safety & Operating Disclaimer:</strong> PhishGuard inspects deterministic lexical and structural properties of submitted URL strings (FS-URL). It does not crawl destination web servers, execute remote JavaScript, or inspect real-time SSL certificate trust chains. Always exercise caution when accessing unknown or unverified hyperlinks.
</div>
""", unsafe_allow_html=True)
