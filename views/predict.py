import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import (
    FEATURES, GREEN, RED, AMBER, SLATE, TEAL, ZERO_AS_MISSING,
    clean_dataset, disclaimer, get_ensemble_name, load_dataset, load_models,
    page_header, risk_level,
)

models = load_models()
ens_name = get_ensemble_name(models)
ensemble = models[ens_name]

page_header("🩺 Risk Prediction", "Adjust the values and the estimate updates instantly.")

left, right = st.columns([1.15, 1], gap="large")

# ------------------------------------------------------------------ inputs
with left:
    with st.container(border=True):
        st.markdown("#### 👤 Personal")
        c1, c2 = st.columns(2)
        age = c1.slider("Age (years)", 18, 90, 35)
        pregnancies = c2.slider("Pregnancies", 0, 17, 1)

    with st.container(border=True):
        st.markdown("#### ❤️ Vitals")
        c1, c2 = st.columns(2)
        bmi = c1.slider("BMI", 15.0, 60.0, 27.0, 0.1)
        bp = c2.slider("Blood pressure, diastolic (mm Hg)", 40, 120, 72)

    with st.container(border=True):
        st.markdown("#### 🧪 Lab values")
        c1, c2 = st.columns(2)
        glucose = c1.slider("Glucose (mg/dL)", 60, 220, 110)
        dpf = c2.slider(
            "Diabetes pedigree function", 0.05, 2.5, 0.40, 0.01,
            help="A score of how strongly diabetes runs in the family.",
        )
        st.caption("Don't know these two? Tick Unknown and the model uses the training-data median.")
        c1, c2 = st.columns(2)
        skin_unknown = c1.checkbox("Skin thickness: Unknown", value=True)
        skin = c1.slider("Skin thickness (mm)", 5, 100, 25, disabled=skin_unknown)
        ins_unknown = c2.checkbox("Insulin: Unknown", value=True)
        insulin = c2.slider("2-hour serum insulin (mu U/ml)", 10, 900, 80, disabled=ins_unknown)

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

prob = float(ensemble.predict_proba(row)[0, 1])
label, colour, emoji = risk_level(prob)

# ----------------------------------------------------------------- results
with right:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 46, "color": colour}},
        title={"text": "Estimated diabetes risk", "font": {"size": 16, "color": SLATE}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": colour, "thickness": 0.3},
            "steps": [
                {"range": [0, 30], "color": "#DCFCE7"},
                {"range": [30, 50], "color": "#FEF3C7"},
                {"range": [50, 100], "color": "#FEE2E2"},
            ],
        },
    ))
    gauge.update_layout(height=270, margin=dict(l=20, r=20, t=60, b=10), template="plotly_white")
    st.plotly_chart(gauge)

    bg = {"High risk": "#FEE2E2", "Moderate risk": "#FEF3C7", "Low risk": "#DCFCE7"}[label]
    msg = {
        "High risk": "Please consult a doctor for proper testing.",
        "Moderate risk": "A check-up is a good idea.",
        "Low risk": "Low risk based on these inputs.",
    }[label]
    st.markdown(
        f"<div class='verdict' style='background:{bg}; color:#1E293B;'>{emoji} {label}. {msg}</div>",
        unsafe_allow_html=True,
    )

    votes = {n: float(m.predict_proba(row)[0, 1]) for n, m in models.items() if n != ens_name}
    bar = go.Figure(go.Bar(
        x=list(votes.values()), y=list(votes.keys()), orientation="h",
        marker_color=TEAL, text=[f"{v:.0%}" for v in votes.values()], textposition="outside",
    ))
    bar.update_layout(
        title="What each model says", height=300, template="plotly_white",
        xaxis=dict(range=[0, 1.1], tickformat=".0%"), margin=dict(l=10, r=30, t=50, b=10),
    )
    st.plotly_chart(bar)

st.caption(f"Final estimate = {ens_name} ensemble (average of the individual models' probabilities).")
disclaimer()

# ---------------------------------------------------------------- extras
tab_whatif, tab_compare, tab_tips, tab_dl = st.tabs(
    ["🎚️ What-if explorer", "📏 You vs the dataset", "💡 Health tips", "⬇️ Download"]
)

with tab_whatif:
    st.write("See how the risk changes as one value moves, with everything else kept as entered above.")
    ranges = {
        "Glucose": (60, 220), "BMI": (15.0, 60.0), "Age": (18, 90), "BloodPressure": (40, 120),
    }
    feat = st.radio("Change this value", list(ranges), horizontal=True)
    lo, hi = ranges[feat]
    xs = np.linspace(lo, hi, 40)
    sweep = pd.concat([row] * len(xs), ignore_index=True)
    sweep[feat] = xs
    ys = ensemble.predict_proba(sweep)[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=TEAL, width=3), name="Risk"))
    fig.add_trace(go.Scatter(
        x=[float(row[feat].iloc[0])], y=[prob], mode="markers",
        marker=dict(size=14, color=colour, line=dict(color="white", width=2)), name="You",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color=RED, annotation_text="High risk threshold")
    fig.add_hline(y=0.3, line_dash="dot", line_color=AMBER, annotation_text="Moderate threshold")
    fig.update_layout(
        height=380, template="plotly_white", xaxis_title=feat, yaxis_title="Estimated risk",
        yaxis=dict(range=[0, 1], tickformat=".0%"), margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig)

with tab_compare:
    ds = load_dataset()
    if ds is None:
        st.info("Dataset file not found, so the comparison is unavailable.")
    else:
        clean = clean_dataset(ds)
        feats = ["Glucose", "BMI", "BloodPressure", "Age"]
        no_d = clean[clean["Outcome"] == 0][feats].mean()
        yes_d = clean[clean["Outcome"] == 1][feats].mean()
        fig = make_subplots(rows=1, cols=len(feats), subplot_titles=feats)
        for i, f in enumerate(feats, start=1):
            fig.add_trace(go.Bar(
                x=["You", "No diabetes (avg)", "Diabetes (avg)"],
                y=[float(row[f].iloc[0]), float(no_d[f]), float(yes_d[f])],
                marker_color=[TEAL, SLATE, RED],
                text=[f"{float(row[f].iloc[0]):.0f}", f"{no_d[f]:.0f}", f"{yes_d[f]:.0f}"],
                textposition="outside",
            ), row=1, col=i)
        fig.update_layout(height=360, template="plotly_white", showlegend=False,
                          margin=dict(l=10, r=10, t=50, b=10))
        fig.update_xaxes(tickfont=dict(size=9))
        st.plotly_chart(fig)
        st.caption("Averages come from the Pima dataset (impossible zeros excluded).")

with tab_tips:
    st.write("General lifestyle information, not personal medical advice.")
    if label == "High risk":
        st.markdown(
            "- **See a doctor** for proper blood tests (fasting glucose or HbA1c).\n"
            "- Cut back on sugary drinks and refined carbohydrates.\n"
            "- Aim for at least 150 minutes of moderate activity per week.\n"
            "- If your BMI is high, even a 5-7% weight loss can reduce risk."
        )
    elif label == "Moderate risk":
        st.markdown(
            "- Consider a **routine check-up** and a blood sugar test.\n"
            "- Add more fibre, vegetables and whole grains to your meals.\n"
            "- Build regular movement into your day, such as walking after meals.\n"
            "- Keep an eye on weight and blood pressure."
        )
    else:
        st.markdown(
            "- Keep up balanced eating and regular physical activity.\n"
            "- Get routine health check-ups, especially if diabetes runs in your family.\n"
            "- Stay hydrated and limit sugary drinks.\n"
            "- Risk changes with age and weight, so recheck now and then."
        )

with tab_dl:
    summary = "\n".join(
        ["DiaPredict result", "=" * 17, ""]
        + [f"{f}: {'unknown' if pd.isna(row[f].iloc[0]) else row[f].iloc[0]}" for f in FEATURES]
        + ["", f"Estimated risk: {prob:.1%} ({label})", f"Model: {ens_name}", "",
           "Educational project only. Not a medical diagnosis or medical advice."]
    )
    st.code(summary, language="text")
    st.download_button("Download summary (.txt)", summary, file_name="diapredict_result.txt")
