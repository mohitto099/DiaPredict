"""Shared helpers for the DiaPredict multipage app."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
GITHUB_URL = "https://github.com/mohitto099/DiaPredict"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
PREFERRED_ENSEMBLE = "Voting (soft)"

TEAL = "#0F766E"
TEAL_LIGHT = "#14B8A6"
RED = "#DC2626"
AMBER = "#F59E0B"
GREEN = "#16A34A"
SLATE = "#64748B"


# ------------------------------------------------------------------ loaders
@st.cache_resource
def _load_models():
    path = MODEL_DIR / "models.joblib"
    return joblib.load(path) if path.exists() else None


def load_models():
    """Return the dict of trained models, or stop the page with a friendly error."""
    models = _load_models()
    if models is None:
        st.error("No trained model found. Run `python train.py` first, then restart the app.")
        st.stop()
    return models


def get_ensemble_name(models):
    return PREFERRED_ENSEMBLE if PREFERRED_ENSEMBLE in models else list(models)[-1]


@st.cache_data
def load_metrics():
    path = MODEL_DIR / "metrics.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name):
    path = MODEL_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_dataset():
    """Raw Pima dataset (zeros kept as-is). Returns None if the CSV is missing."""
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    cols = FEATURES + ["Outcome"]
    if not set(cols).issubset(df.columns):
        if df.shape[1] != 9:
            return None
        try:
            float(df.columns[0])  # header row was actually data
            df = pd.read_csv(DATA_PATH, header=None)
        except ValueError:
            pass
        df.columns = cols
    return df[cols]


def clean_dataset(df):
    """Same cleaning as train.py: impossible zeros become NaN."""
    out = df.copy()
    out[ZERO_AS_MISSING] = out[ZERO_AS_MISSING].replace(0, np.nan)
    return out


# ------------------------------------------------------------------ helpers
def risk_level(prob):
    """(label, colour, emoji) for a probability between 0 and 1."""
    if prob >= 0.5:
        return "High risk", RED, "🔴"
    if prob >= 0.3:
        return "Moderate risk", AMBER, "🟠"
    return "Low risk", GREEN, "🟢"


def page_header(title, subtitle):
    st.markdown(
        f"<div class='page-head'><h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def disclaimer():
    st.caption("⚠️ Educational student project. This is not a medical diagnosis or medical advice.")


CSS = f"""
<style>
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }}
.page-head h1 {{ color: {TEAL}; margin-bottom: 0.1rem; font-weight: 800; }}
.page-head p {{ color: {SLATE}; font-size: 1.05rem; margin-top: 0; }}

.hero {{
    background: linear-gradient(135deg, {TEAL} 0%, {TEAL_LIGHT} 100%);
    color: #ffffff; padding: 2.6rem 2.2rem; border-radius: 20px; margin-bottom: 1.4rem;
}}
.hero h1 {{ color: #ffffff; font-size: 2.8rem; margin: 0 0 0.4rem 0; font-weight: 800; }}
.hero p {{ color: #ECFEFF; font-size: 1.15rem; margin: 0; max-width: 640px; }}

.card {{
    background: #ffffff; border: 1px solid #E2E8F0; border-radius: 16px;
    padding: 1.2rem 1.3rem; height: 100%;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}}
.card h4 {{ color: {TEAL}; margin: 0 0 0.35rem 0; }}
.card p {{ color: #334155; margin: 0; font-size: 0.95rem; }}

.stat {{
    background: #F0FDFA; border: 1px solid #CCFBF1; border-radius: 16px;
    padding: 1rem 1.1rem; text-align: center;
}}
.stat .num {{ font-size: 1.9rem; font-weight: 800; color: {TEAL}; line-height: 1.1; }}
.stat .lbl {{ color: {SLATE}; font-size: 0.85rem; }}

.flow {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin: 0.5rem 0 1rem 0; }}
.flow .step {{
    background: #F0FDFA; border: 1px solid #99F6E4; color: #115E59; border-radius: 12px;
    padding: 0.55rem 0.9rem; font-weight: 600; font-size: 0.92rem;
}}
.flow .arrow {{ color: {TEAL_LIGHT}; font-weight: 800; }}

.verdict {{
    border-radius: 14px; padding: 0.9rem 1.1rem; font-weight: 600; margin-bottom: 0.6rem;
}}
section[data-testid="stSidebar"] {{ background: #F0FDFA; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
