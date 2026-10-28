"""DiaPredict - Streamlit web app.

Run:  streamlit run app.py   (after running train.py once)
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path("models")
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ENSEMBLE = "Voting (soft)"

st.set_page_config(page_title="DiaPredict", page_icon="🩺", layout="wide")


@st.cache_resource
def load_models():
    return joblib.load(MODEL_DIR / "models.joblib")


@st.cache_data
def load_json(name):
    return json.loads((MODEL_DIR / name).read_text())


if not (MODEL_DIR / "models.joblib").exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

models = load_models()
metrics = pd.read_csv(MODEL_DIR / "metrics.csv")

st.title("🩺 DiaPredict")
st.caption("Diabetes risk prediction using an ensemble of machine learning models (Pima Indians dataset)")

tab_predict, tab_compare, tab_about = st.tabs(["Predict", "Model Comparison", "About"])

# ---------------------------------------------------------------- Predict
with tab_predict:
    st.subheader("Enter patient details")
    st.write("If you don't know a value (e.g. insulin or skin thickness), tick **Unknown** "
             "and the model will fill it in using the training-data median.")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 40, 250, 110)
        bp = st.number_input("Blood pressure, diastolic (mm Hg)", 30, 130, 72)
        age = st.number_input("Age (years)", 18, 100, 30)
    with col2:
        bmi = st.number_input("BMI", 10.0, 70.0, 27.0, step=0.1)
        dpf = st.number_input("Diabetes pedigree function", 0.05, 2.5, 0.40, step=0.01,
                              help="A score of how strongly diabetes runs in the family.")
        skin_unknown = st.checkbox("Skin thickness: Unknown", value=True)
        skin = st.number_input("Skin thickness (mm)", 5, 100, 25, disabled=skin_unknown)
        ins_unknown = st.checkbox("Insulin: Unknown", value=True)
        insulin = st.number_input("2-hour serum insulin (mu U/ml)", 10, 900, 80, disabled=ins_unknown)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": np.nan if skin_unknown else skin,
            "Insulin": np.nan if ins_unknown else insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }])[FEATURES]

        prob = models[ENSEMBLE].predict_proba(row)[0, 1]
        st.divider()
        left, right = st.columns([1, 2])
        with left:
            st.metric("Estimated diabetes risk", f"{prob:.0%}")
            if prob >= 0.5:
                st.error("High risk. Please consult a doctor for proper testing.")
            elif prob >= 0.3:
                st.warning("Moderate risk. A check-up is a good idea.")
            else:
                st.success("Low risk based on these inputs.")
        with right:
            st.write("**What each individual model says**")
            votes = {n: m.predict_proba(row)[0, 1] for n, m in models.items() if n != ENSEMBLE}
            vote_df = pd.DataFrame({"Model": list(votes), "Risk probability": list(votes.values())})
            st.bar_chart(vote_df.set_index("Model"), y="Risk probability", height=250)

    st.caption("⚠️ Educational project only. This is not a medical diagnosis or medical advice.")

# ------------------------------------------------------- Model comparison
with tab_compare:
    st.subheader("Test-set performance")
    st.dataframe(metrics.set_index("Model").style.highlight_max(axis=0, color="#cfe8cf"),
                 use_container_width=True)
    st.caption("Recall is highlighted as important: in screening, missing a real case costs more than a false alarm.")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**ROC curves**")
        rocs = load_json("roc.json")
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, r in rocs.items():
            ax.plot(r["fpr"], r["tpr"], label=name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7)
        st.pyplot(fig)
    with c2:
        st.write("**Confusion matrix**")
        cms = load_json("confusion.json")
        choice = st.selectbox("Model", list(cms), index=list(cms).index(ENSEMBLE))
        cm = np.array(cms[choice])
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        ax2.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax2.text(j, i, v, ha="center", va="center", fontsize=14)
        ax2.set_xticks([0, 1], ["Pred: No", "Pred: Yes"])
        ax2.set_yticks([0, 1], ["Actual: No", "Actual: Yes"])
        st.pyplot(fig2)

    st.write("**Feature importance (Random Forest)**")
    imp = pd.Series(load_json("importance.json")).sort_values()
    st.bar_chart(imp, horizontal=True)

# ------------------------------------------------------------------ About
with tab_about:
    st.subheader("How DiaPredict works")
    st.markdown(
        """
- **Data:** Pima Indians Diabetes dataset (768 records, 8 features). Impossible zeros in
  glucose, blood pressure, skin thickness, insulin and BMI are treated as missing values.
- **Leakage-safe pipeline:** the train/test split happens first; imputation and scaling are
  fit only on training data inside a scikit-learn `Pipeline`.
- **Models:** Logistic Regression (L1/L2), SVM (linear/RBF), Random Forest and a boosting model,
  each tuned with cross-validated grid search.
- **Ensembles:** soft Voting (used for the prediction here) and Stacking with a
  Logistic Regression meta-learner.
- **Evaluation:** 5-fold stratified cross-validation, then Accuracy, Precision, Recall, F1 and ROC-AUC
  on a held-out test set.
        """
    )
    st.warning("DiaPredict is a student project for learning. It must not be used for real medical decisions.")
