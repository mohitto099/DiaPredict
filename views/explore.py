import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import (
    FEATURES, RED, SLATE, TEAL, ZERO_AS_MISSING, clean_dataset, disclaimer, load_dataset, page_header,
)

page_header("🔍 Explore the Data", "What the Pima Indians Diabetes dataset looks like before any model sees it.")

ds = load_dataset()
if ds is None:
    st.error("Dataset not found. Place the Pima file at `data/diabetes.csv` (9 columns, last one `Outcome`).")
    st.stop()

clean = clean_dataset(ds)

# ----------------------------------------------------------------- overview
n_pos = int(ds["Outcome"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("Records", len(ds))
c2.metric("Features", len(FEATURES))
c3.metric(f"Diabetic ({n_pos} of {len(ds)})", f"{n_pos / len(ds):.0%}")

tab_bal, tab_miss, tab_dist, tab_corr, tab_raw = st.tabs(
    ["⚖️ Class balance", "🧹 Missing values", "📊 Distributions", "🔗 Correlations", "🗂️ Raw data"]
)

with tab_bal:
    fig = go.Figure(go.Pie(
        labels=["No diabetes", "Diabetes"], values=[len(ds) - n_pos, n_pos],
        hole=0.55, marker=dict(colors=[TEAL, RED]), textinfo="label+percent",
    ))
    fig.update_layout(height=380, template="plotly_white", showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig)
    st.caption("About two thirds are non-diabetic, so accuracy alone can look good while missing diabetics. "
               "That is why Recall, F1 and ROC-AUC are reported too.")

with tab_miss:
    st.write(
        "In these columns a value of **0 is physiologically impossible** (nobody has 0 blood pressure), "
        "so the zeros are really missing values."
    )
    zeros = (ds[ZERO_AS_MISSING] == 0).sum()
    fig = go.Figure(go.Bar(
        x=zeros.index.tolist(), y=zeros.values.tolist(), marker_color=RED,
        text=[f"{v / len(ds):.0%}" for v in zeros.values], textposition="outside",
    ))
    fig.update_layout(height=360, template="plotly_white", yaxis_title="Rows with a hidden zero",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig)

    medians = clean[ZERO_AS_MISSING].median().round(1)
    st.markdown("**Fill-in values (median), learned from training data inside the pipeline:**")
    st.dataframe(medians.rename("Median").to_frame().T)
    st.caption("Insulin and skin thickness are missing in a large share of rows, "
               "which is why the Predict page lets you mark them as Unknown.")

with tab_dist:
    feat = st.selectbox("Feature", FEATURES, index=1)
    data = clean if feat in ZERO_AS_MISSING else ds
    fig = go.Figure()
    for outcome, name, colour in [(0, "No diabetes", TEAL), (1, "Diabetes", RED)]:
        vals = data.loc[data["Outcome"] == outcome, feat].dropna()
        fig.add_trace(go.Histogram(x=vals, name=name, marker_color=colour, opacity=0.65, nbinsx=30))
    fig.update_layout(
        barmode="overlay", height=400, template="plotly_white", xaxis_title=feat,
        yaxis_title="Patients", margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig)
    m0 = data.loc[data["Outcome"] == 0, feat].mean()
    m1 = data.loc[data["Outcome"] == 1, feat].mean()
    st.caption(f"Average {feat}: {m0:.1f} without diabetes vs {m1:.1f} with diabetes.")

with tab_corr:
    corr = clean.corr(numeric_only=True)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        zmin=-1, zmax=1, colorscale="RdBu_r",
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=10),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=520, template="plotly_white", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig)
    top = corr["Outcome"].drop("Outcome").abs().sort_values(ascending=False)
    st.caption(f"Strongest link with the outcome: {top.index[0]}, then {top.index[1]} and {top.index[2]}.")

with tab_raw:
    st.dataframe(ds.head(50))
    st.caption("First 50 rows of the raw file (zeros not yet treated as missing).")

disclaimer()
