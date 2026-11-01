import streamlit as st

from utils import GITHUB_URL, disclaimer, get_ensemble_name, load_dataset, load_metrics, load_models

models = load_models()
metrics = load_metrics()
ds = load_dataset()

st.markdown(
    """
<div class="hero">
  <h1>🩺 DiaPredict</h1>
  <p>Estimate diabetes risk from a few health measurements, using not one but an ensemble of
  machine learning models working together.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.page_link("views/predict.py", label="Start a prediction", icon="🩺")

# ------------------------------------------------------------- headline stats
rows = len(ds) if ds is not None else 768
n_algos = max(len(models) - 2, 1)  # everything except the two ensembles

best_auc, best_name, ens_recall = "-", "-", "-"
if metrics is not None and len(metrics):
    top = metrics.loc[metrics["ROC-AUC"].idxmax()]
    best_auc, best_name = f"{top['ROC-AUC']:.2f}", top["Model"]
    ens_row = metrics[metrics["Model"] == get_ensemble_name(models)]
    if len(ens_row):
        ens_recall = f"{ens_row.iloc[0]['Recall']:.0%}"

st.write("")
c1, c2, c3, c4 = st.columns(4)
for col, num, lbl in [
    (c1, f"{rows}", "patient records"),
    (c2, f"{n_algos} + 2", "algorithms + ensembles"),
    (c3, best_auc, f"best ROC-AUC ({best_name})"),
    (c4, ens_recall, "ensemble recall (test set)"),
]:
    col.markdown(
        f"<div class='stat'><div class='num'>{num}</div><div class='lbl'>{lbl}</div></div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------- feature cards
st.write("")
st.subheader("What you can do here")
a, b, c = st.columns(3)
a.markdown(
    "<div class='card'><h4>🩺 Predict</h4><p>Enter health details and get a risk score, "
    "see what each model thinks, and explore what-if scenarios.</p></div>",
    unsafe_allow_html=True,
)
b.markdown(
    "<div class='card'><h4>📊 Model Insights</h4><p>Compare the algorithms with metrics, "
    "ROC curves, confusion matrices and feature importance.</p></div>",
    unsafe_allow_html=True,
)
c.markdown(
    "<div class='card'><h4>🔍 Explore Data</h4><p>Look inside the Pima dataset: "
    "distributions, correlations and how missing values were handled.</p></div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------- pipeline strip
st.write("")
st.subheader("How it works, in one line")
st.markdown(
    """
<div class="flow">
  <span class="step">📥 Pima data</span><span class="arrow">→</span>
  <span class="step">🧹 Clean &amp; split</span><span class="arrow">→</span>
  <span class="step">⚙️ Impute &amp; scale</span><span class="arrow">→</span>
  <span class="step">🤖 4 tuned models</span><span class="arrow">→</span>
  <span class="step">🗳️ Ensemble</span><span class="arrow">→</span>
  <span class="step">🩺 Risk score</span>
</div>
""",
    unsafe_allow_html=True,
)
st.page_link("views/how_it_works.py", label="See the full explanation", icon="⚙️")

st.divider()
st.markdown(f"Source code: [{GITHUB_URL}]({GITHUB_URL})")
disclaimer()
