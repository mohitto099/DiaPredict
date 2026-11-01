import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import (
    SLATE, TEAL, TEAL_LIGHT, disclaimer, get_ensemble_name, load_json, load_metrics,
    load_models, page_header,
)

models = load_models()
metrics = load_metrics()
ens_name = get_ensemble_name(models)

page_header("📊 Model Insights", "How the algorithms compare on data they have never seen.")

if metrics is None:
    st.error("metrics.csv not found. Run `python train.py` first.")
    st.stop()

# ------------------------------------------------------------ headline cards
best_auc = metrics.loc[metrics["ROC-AUC"].idxmax()]
best_rec = metrics.loc[metrics["Recall"].idxmax()]
best_f1 = metrics.loc[metrics["F1"].idxmax()]
c1, c2, c3 = st.columns(3)
c1.metric(f"Best ROC-AUC ({best_auc['Model']})", f"{best_auc['ROC-AUC']:.3f}")
c2.metric(f"Best Recall ({best_rec['Model']})", f"{best_rec['Recall']:.3f}")
c3.metric(f"Best F1 ({best_f1['Model']})", f"{best_f1['F1']:.3f}")

tab_table, tab_roc, tab_cm, tab_feat = st.tabs(
    ["📋 Metrics", "📈 ROC curves", "🧮 Confusion matrix", "🌲 Feature importance"]
)

# ------------------------------------------------------------------ metrics
with tab_table:
    table = metrics.set_index("Model")
    st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, color="#CCFBF1"))
    st.caption(
        "Green = best in each column. Recall matters most in screening: "
        "missing a real case costs more than a false alarm."
    )

    cols = [c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"] if c in metrics.columns]
    palette = [TEAL, TEAL_LIGHT, "#F59E0B", "#DC2626", SLATE]
    fig = go.Figure()
    for col, colour in zip(cols, palette):
        fig.add_trace(go.Bar(name=col, x=metrics["Model"], y=metrics[col], marker_color=colour))
    fig.update_layout(
        barmode="group", height=400, template="plotly_white",
        yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig)

# ---------------------------------------------------------------------- ROC
with tab_roc:
    rocs = load_json("roc.json")
    if rocs is None:
        st.info("roc.json not found.")
    else:
        chosen = st.multiselect("Models to show", list(rocs), default=list(rocs))
        auc_lookup = metrics.set_index("Model")["ROC-AUC"].to_dict()
        fig = go.Figure()
        for name in chosen:
            r = rocs[name]
            auc = auc_lookup.get(name)
            label = f"{name} (AUC {auc:.3f})" if auc is not None else name
            fig.add_trace(go.Scatter(
                x=r["fpr"], y=r["tpr"], mode="lines", name=label,
                line=dict(width=4 if name == ens_name else 2),
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
            line=dict(color="#94A3B8", dash="dash"),
        ))
        fig.update_layout(
            height=470, template="plotly_white",
            xaxis_title="False positive rate", yaxis_title="True positive rate",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig)
        st.caption("The closer a curve hugs the top-left corner, the better the model separates the classes.")

# ---------------------------------------------------------- confusion matrix
with tab_cm:
    cms = load_json("confusion.json")
    if cms is None:
        st.info("confusion.json not found.")
    else:
        names = list(cms)
        choice = st.selectbox("Model", names, index=names.index(ens_name) if ens_name in names else 0)
        cm = np.array(cms[choice])
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
            text=cm, texttemplate="%{text}", textfont=dict(size=26),
            colorscale="Teal", showscale=False,
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=400, template="plotly_white", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig)
        tn, fp, fn, tp = cm.ravel()
        st.markdown(
            f"**{tp}** diabetic cases caught, **{fn}** missed (false negatives), "
            f"**{fp}** false alarms, **{tn}** correctly cleared."
        )

# --------------------------------------------------------- feature importance
with tab_feat:
    imp = load_json("importance.json")
    if imp is None:
        st.info("importance.json not found.")
    else:
        items = sorted(imp.items(), key=lambda kv: kv[1])
        fig = go.Figure(go.Bar(
            x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker_color=TEAL,
        ))
        fig.update_layout(height=400, template="plotly_white", margin=dict(l=10, r=10, t=20, b=10),
                          xaxis_title="Importance (Random Forest)")
        st.plotly_chart(fig)
        st.caption("Glucose, BMI and age usually dominate: the model leans on the same signals doctors do.")

params = load_json("best_params.json")
if params:
    with st.expander("Best hyperparameters found by grid search"):
        st.json(params)

disclaimer()
