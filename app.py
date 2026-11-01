"""DiaPredict - Streamlit multipage app (entry point).

Run:  streamlit run app.py   (after running train.py once)
"""
import streamlit as st

from utils import GITHUB_URL, inject_css

st.set_page_config(page_title="DiaPredict", page_icon="🩺", layout="wide")
inject_css()

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/predict.py", title="Predict", icon="🩺"),
    st.Page("views/insights.py", title="Model Insights", icon="📊"),
    st.Page("views/explore.py", title="Explore Data", icon="🔍"),
    st.Page("views/how_it_works.py", title="How It Works", icon="⚙️"),
]
navigation = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🩺 DiaPredict")
    st.caption("Diabetes risk prediction with an ensemble of ML models")
    st.markdown(f"[⭐ View on GitHub]({GITHUB_URL})")
    st.caption("Educational project. Not medical advice.")

navigation.run()
