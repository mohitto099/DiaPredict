import pandas as pd
import streamlit as st

from utils import GITHUB_URL, disclaimer, page_header

page_header("⚙️ How DiaPredict Works", "From raw data to a risk score, step by step.")

st.markdown(
    """
<div class="flow">
  <span class="step">📥 Data</span><span class="arrow">→</span>
  <span class="step">🧹 Clean &amp; split</span><span class="arrow">→</span>
  <span class="step">⚙️ Impute &amp; scale</span><span class="arrow">→</span>
  <span class="step">🤖 Train &amp; tune 4 models</span><span class="arrow">→</span>
  <span class="step">🗳️ Ensemble</span><span class="arrow">→</span>
  <span class="step">📏 Evaluate</span><span class="arrow">→</span>
  <span class="step">🩺 Predict</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("1. Data: the Pima Indians Diabetes dataset", expanded=True):
    st.markdown(
        "768 patient records with 8 measurements (pregnancies, glucose, blood pressure, skin thickness, "
        "insulin, BMI, diabetes pedigree function, age). The target is whether the patient developed diabetes. "
        "It is the standard benchmark for this problem, so results can be compared with published work."
    )

with st.expander("2. Cleaning and the train/test split"):
    st.markdown(
        "- Zeros in glucose, blood pressure, skin thickness, insulin and BMI are impossible, so they are treated as **missing**.\n"
        "- The data is split into 80% training and 20% test **before** any preprocessing, keeping the class ratio the same in both.\n"
        "- The test set is never touched until the final evaluation."
    )

with st.expander("3. Leakage-safe preprocessing"):
    st.markdown(
        "Missing values are filled with the **median** and features are **standardised**. Both steps live inside a "
        "scikit-learn `Pipeline`, so they learn only from the training part of each cross-validation fold. "
        "If the test data influenced these steps, the scores would look better than they really are. "
        "That is called *preprocessing leakage*."
    )

with st.expander("4. Four algorithms, each tuned"):
    st.markdown(
        "- **Logistic Regression** with L1 and L2 regularisation: a fast, interpretable baseline.\n"
        "- **SVM** with linear and RBF kernels: finds the widest-margin boundary.\n"
        "- **Random Forest**: many decision trees voting together, which reduces overfitting.\n"
        "- **XGBoost** (gradient boosting): trees built one after another, each fixing the last one's mistakes.\n\n"
        "Each model's settings were chosen by **5-fold stratified cross-validation with grid search**."
    )

with st.expander("5. Ensembles: models working together"):
    st.markdown(
        "- **Soft Voting** averages the predicted probabilities of all models. This is what the Predict page uses.\n"
        "- **Stacking** trains a Logistic Regression on top of the models' outputs, learning how much to trust each one.\n\n"
        "Because the models make different mistakes, combining them tends to be more stable than any single model "
        "(the bias-variance idea behind ensemble learning)."
    )

with st.expander("6. Evaluation"):
    st.markdown(
        "Scores come from the held-out test set: **Accuracy**, **Precision**, **Recall**, **F1** and **ROC-AUC**. "
        "For a screening tool, **Recall** is especially important, because missing a person who has diabetes "
        "is worse than a false alarm."
    )

# ---------------------------------------------------------- syllabus mapping
st.subheader("Where the course syllabus shows up")
mapping = pd.DataFrame(
    [
        ("Module 1.1", "Train/test split, k-fold cross-validation, overfitting, bias-variance", "Train/test split, 5-fold CV, tuning"),
        ("Module 1.2", "Confusion matrix, precision, recall, F1, ROC-AUC", "Model Insights page"),
        ("Module 1.3", "Data leakage and preprocessing leakage", "Split first, preprocessing inside a Pipeline"),
        ("Module 3.1", "Logistic Regression, L1/L2 regularisation, SVM and kernels", "Two of the four models"),
        ("Module 5.1-5.2", "Random Forest, boosting, Voting (soft), Stacking", "Two models plus both ensembles"),
        ("Module 6.2", "Version control with Git, ML pipelines", f"Code on GitHub, scikit-learn Pipeline"),
    ],
    columns=["Syllabus", "Concept", "Used in DiaPredict"],
)
st.dataframe(mapping, hide_index=True)

# ------------------------------------------------------------- limitations
st.subheader("Limitations")
st.markdown(
    "- The dataset is small (768 records) and comes from one population, so results may not generalise.\n"
    "- Insulin and skin thickness are missing for many records and are filled with medians.\n"
    "- It estimates risk from a few numbers. It cannot replace a blood test or a doctor."
)
st.warning("DiaPredict is a student project for learning. It must not be used for real medical decisions.")
st.markdown(f"Source code: [{GITHUB_URL}]({GITHUB_URL})")
disclaimer()
