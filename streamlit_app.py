import asyncio
import io
import json
import os
import warnings
from typing import Any, List

import streamlit as st
from pandas.errors import PerformanceWarning

# Suppress pandas fragmentation warnings emitted from model_logic when adding many columns
warnings.simplefilter("ignore", PerformanceWarning)

from model_logic import make_prediction, predict_price_with_gemini

st.set_page_config(
    page_title="Southern Spain Real Estate Price Calculator",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Minimal CSS for brighter look, center content, primary button color, auto-grow textarea, and hide header link icons
st.markdown("""
<style>
/* Hide Streamlit chrome */
div[data-testid="stToolbar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }

/* Respect theme background */
.stApp { background-color: var(--background-color) !important; }

/* Center content */
.block-container { max-width: 900px; margin: 0 auto; }

/* Remove anchor icons next to markdown headers */
div[data-testid="stMarkdownContainer"] a { display: none !important; }

/* Row for buttons: align baseline and same height */
.btn-row { 
  display: flex; 
  gap: 12px; 
  align-items: center;  /* vertically center within row */
  margin-top: 6px; 
  margin-bottom: 6px;
}

/* Normalize both buttons' height/padding/margins */
.btn-row button {
  min-height: 44px !important;      /* consistent button height */
  padding: 8px 16px !important;
  margin: 0 !important;
}

/* Primary CatBoost button styling (first button in row) */
.btn-row .primary-btn button {
  background-color: #14b8a6 !important;  /* teal base */
  color: #0f172a !important;             /* dark slate text */
  border: none !important;
  box-shadow: 0 2px 8px rgba(20, 184, 166, 0.35) !important;
}
.btn-row .primary-btn button:hover { background-color: #0d9488 !important; color: #0f172a !important; }
.btn-row .primary-btn button:active, 
.btn-row .primary-btn button:focus { background-color: #0f766e !important; color: #0f172a !important; outline: none !important; }

/* Disabled Gemini button gets a subtle border to look consistent when dimmed */
.btn-row button[disabled] {
  opacity: 0.65 !important;
  border: 1px solid rgba(229, 231, 235, 0.25) !important;
  background-color: var(--secondary-background-color) !important;
  color: var(--text-color) !important;
}
</style>
""", unsafe_allow_html=True)

# Header and helper text rendered as plain HTML to avoid anchor icons
st.markdown("<h1 style='margin-bottom:0.25rem;'>Southern Spain Real Estate Price Calculator</h1>", unsafe_allow_html=True)
st.write(
    "Enter property details to estimate real estate market price.\n\n"
    "- All fields are required except images.\n"
    "- Upload up to 3 images or paste up to 3 image URLs.\n"
)

# Warn if artifacts missing
if not os.path.exists("real_estate_model.cbm"):
    st.warning("Model file real_estate_model.cbm not found. Predictions will fail until it is present.", icon="⚠️")
if not os.path.exists("location_stats.json"):
    st.info("location_stats.json not found. The app will fall back to a default encoding for location.", icon="ℹ️")

def _prepare_image_inputs_from_files(uploaded_files: List[Any] | None) -> List[io.BytesIO]:
    inputs: List[io.BytesIO] = []
    if uploaded_files:
        for f in uploaded_files[:3]:
            try:
                buf = io.BytesIO(f.read())
                buf.seek(0)
                inputs.append(buf)
            except Exception:
                continue
    return inputs

def _prepare_image_inputs(uploaded_files: List[Any] | None, image_urls_text: str) -> List[Any]:
    inputs: List[Any] = []
    inputs.extend(_prepare_image_inputs_from_files(uploaded_files))
    text = (image_urls_text or "").strip()
    if text:
        urls: List[str] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                urls = [str(u) for u in parsed][:3]
            else:
                urls = [u.strip() for u in text.split(",") if u.strip()][:3]
        except json.JSONDecodeError:
            urls = [u.strip() for u in text.split(",") if u.strip()][:3]
        inputs.extend(urls)
    return inputs

def _validate(
    location: str, title: str, bedrooms: float, bathrooms: float,
    indoor_area: float, outdoor_area: float, features: str, description: str
) -> list[str]:
    errors = []
    if not location or not location.strip(): errors.append("Location is required.")
    if not title or not title.strip(): errors.append("Title is required.")
    if bedrooms is None: errors.append("Bedrooms is required.")
    if bathrooms is None: errors.append("Bathrooms is required.")
    if indoor_area is None: errors.append("Indoor area is required.")
    if outdoor_area is None: errors.append("Outdoor area is required.")
    if features is None or len(features.strip()) == 0: errors.append("Features is required.")
    if description is None or len(description.strip()) == 0: errors.append("Description is required.")
    if bedrooms not in (None,) and bedrooms < 0: errors.append("Bedrooms cannot be negative.")
    if bathrooms not in (None,) and bathrooms < 0: errors.append("Bathrooms cannot be negative.")
    if indoor_area not in (None,) and indoor_area < 0: errors.append("Indoor area cannot be negative.")
    if outdoor_area not in (None,) and outdoor_area < 0: errors.append("Outdoor area cannot be negative.")
    return errors

# clear_on_submit=True prevents values from sticking after submit
with st.form("price_form", clear_on_submit=False):
    st.markdown("<h3 style='margin-top:1rem;'>Property details</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("Location", value="", placeholder="e.g., Nueva Andalucía, Costa del Sol")
    with col2:
        title = st.text_input("Title", value="", placeholder="e.g., 3 Bedroom Apartment")

    col3, col4 = st.columns(2)
    with col3:
        bedrooms = st.number_input("Bedrooms", min_value=0, step=1, value=0, format="%d")
    with col4:
        bathrooms = st.number_input("Bathrooms", min_value=0, step=1, value=0, format="%d")

    col5, col6 = st.columns(2)
    with col5:
        indoor_area = st.number_input("Indoor Area (sqm)", min_value=0.0, step=1.0, value=0.0)
    with col6:
        outdoor_area = st.number_input("Outdoor Area (sqm)", min_value=0.0, step=1.0, value=0.0)

    features = st.text_area(
        "Features (Should be separated by pipes '|' or commas ',')",
        value="",
        placeholder="Close to Shops|Fitted Kitchen|Private Terrace",
        height=80,
    )
    description = st.text_area(
        "Description",
        value="",
        placeholder="Short description of the listing...",
        height=100,
    )

    with st.expander("Images (optional)"):
        uploaded_files = st.file_uploader(
            "Upload up to 3 images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        image_urls_text = st.text_area(
            "Image URLs (JSON array or comma-separated, up to 3)",
            value="",
            placeholder='["https://example.com/img1.jpg", "https://example.com/img2.jpg"]',
            height=70,
        )

    # Buttons
    st.markdown("<div class='btn-row'>", unsafe_allow_html=True)

    # CatBoost: styled as primary
    st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
    do_catboost = st.form_submit_button("Predict (CatBoost Model)", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    # Gemini: disabled if no key
    do_gemini = st.form_submit_button(
        "Predict (Gemini LLM)",
        use_container_width=False,
        disabled=not bool(os.getenv("GEMINI_API_KEY")),
    )

    st.markdown("</div>", unsafe_allow_html=True)

# Handle actions
if do_catboost or do_gemini:
    errors = _validate(location, title, float(bedrooms), float(bathrooms), float(indoor_area), float(outdoor_area), features, description)
    if errors:
        st.error("Please fix these issues:\n- " + "\n- ".join(errors))
    else:
        if do_catboost:
            with st.spinner("Predicting with CatBoost..."):
                try:
                    # Normalize features: pipes or commas are fine; optionally replace commas with pipes
                    normalized_features = features.replace(",", "|")
                    image_inputs = _prepare_image_inputs(uploaded_files, image_urls_text)
                    data = {
                        "location": location.strip(),
                        "title": title.strip(),
                        "bedrooms": float(bedrooms),
                        "bathrooms": float(bathrooms),
                        "indoor_area": float(indoor_area),
                        "outdoor_area": float(outdoor_area),
                        "features": normalized_features.strip(),
                        "description": description.strip(),
                    }
                    price = make_prediction(data, image_inputs)
                    st.markdown("<h3 style='margin-top:1rem;'>Predicted price</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='color:#0ea5e9'>€{price:,.2f}</h2>", unsafe_allow_html=True)
                except Exception as exc:
                    st.error(f"Prediction error: {type(exc).__name__}: {exc}")

        if do_gemini:
            if not os.getenv("GEMINI_API_KEY"):
                st.error("GEMINI_API_KEY not set. Please set it in your environment to use Gemini.")
            else:
                with st.spinner("Predicting with Gemini..."):
                    try:
                        data = {
                            "location": location.strip(),
                            "title": title.strip(),
                            "bedrooms": float(bedrooms),
                            "bathrooms": float(bathrooms),
                            "indoor_area": float(indoor_area),
                            # outdoor_area omitted for Gemini schema
                            "features": features.strip(),
                            "description": description.strip(),
                        }
                        result = asyncio.run(predict_price_with_gemini(data))
                        st.markdown("<h3 style='margin-top:1rem;'>Gemini estimate</h3>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='color:#0ea5e9'>€{result.predicted_price_eur:,.2f}</h2>", unsafe_allow_html=True)
                        st.write(f"Confidence: {result.confidence_level:.2f}")
                        st.caption(f"Justification: {result.justification}")
                    except Exception as exc:
                        st.error(f"Gemini error: {type(exc).__name__}: {exc}")