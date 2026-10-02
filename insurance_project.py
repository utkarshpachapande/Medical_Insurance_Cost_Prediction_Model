from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ARTIFACT_PATH, DATA_PATH, TARGET
from src.data import load_data
from src.explain import format_feature_label, scenario_impacts
from src.model import load_bundle, train_and_save

st.set_page_config(
    page_title="Explainable Medical Insurance Cost Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#eaf0f6; --muted:#91a2b2; --panel:#101823; --line:#1f2c38; --accent:#66e3c4; --accent2:#7c9cff; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(circle at 20% 0%, #17253a 0%, #091018 36%, #070c12 100%); color:var(--ink); }
    .block-container { max-width: 1500px; padding-top: 1.6rem; padding-bottom: 3rem; }
    h1,h2,h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:-0.02em; }
    .hero { padding: 1.8rem 2rem; border:1px solid #263745; border-radius:24px; background:linear-gradient(135deg,rgba(20,35,50,.96),rgba(8,16,24,.93)); box-shadow:0 20px 60px rgba(0,0,0,.25); }
    .eyebrow { color:var(--accent); text-transform:uppercase; letter-spacing:.16em; font-weight:700; font-size:.73rem; }
    .hero h1 { font-size:3rem; margin:.35rem 0 .45rem; }
    .hero p { color:#aebdca; max-width:850px; font-size:1.02rem; line-height:1.65; }
    .chip { display:inline-block; padding:.42rem .7rem; border:1px solid #2a3c4a; border-radius:999px; margin:.2rem .25rem 0 0; background:#0d1822; color:#b7c7d5; font-size:.78rem; }
    .metric-card { padding:1rem 1.1rem; border:1px solid var(--line); border-radius:16px; background:rgba(15,24,34,.78); }
    .metric-label { color:var(--muted); font-size:.78rem; }
    .metric-value { color:#f5fafc; font-size:1.45rem; font-weight:700; margin-top:.2rem; }
    .prediction { padding:1.4rem; border-radius:22px; border:1px solid #2c4d50; background:linear-gradient(135deg,rgba(19,44,49,.9),rgba(13,25,34,.95)); }
    .prediction .value { font-family:'Space Grotesk'; font-size:3.1rem; font-weight:700; }
    .prediction .sub { color:#a8c4c3; }
    .small-note { color:#8fa1b0; font-size:.8rem; }
    .section-title { margin-top:1.2rem; margin-bottom:.3rem; }
    [data-testid="stMetric"] { background:rgba(15,24,34,.72); border:1px solid var(--line); padding:1rem; border-radius:14px; }
    [data-testid="stSidebar"] { background:#080f16; border-right:1px solid var(--line); }
    .stButton>button { border-radius:12px; border:1px solid #304453; background:#101b25; color:#edf6f7; font-weight:600; }
    .stButton>button:hover { border-color:#5e9f99; color:#ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data

def get_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


@st.cache_resource

def get_bundle():
    if ARTIFACT_PATH.exists():
        return load_bundle(ARTIFACT_PATH)
    bundle = train_and_save()
    return bundle


def money(x: float) -> str:
    return f"${x:,.0f}"


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Healthy range"
    if bmi < 30:
        return "Overweight"
    return "Obesity range"


def gauge(value: float, low: float, high: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"prefix":"$", "valueformat":",.0f"},
        gauge={
            "axis":{"range":[0, max(high * 1.25, value * 1.2, 50000)], "tickformat":"$,.0f"},
            "bar":{"thickness":0.28},
            "steps":[
                {"range":[0, low], "color":"#163238"},
                {"range":[low, high], "color":"#1a4d4c"},
                {"range":[high, max(high * 1.25, value * 1.2, 50000)], "color":"#42252d"},
            ],
        },
        title={"text":"Estimated insurance charge"},
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=55,b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#edf6f7")
    return fig


def make_input_frame(age, sex, bmi, children, smoker, region) -> pd.DataFrame:
    return pd.DataFrame([{
        "age": int(age),
        "bmi": float(bmi),
        "children": int(children),
        "sex": sex.lower(),
        "smoker": smoker.lower(),
        "region": region.lower(),
    }])


data = get_data()
bundle = get_bundle()

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">ML • Explainability • Decision Intelligence</div>
      <h1>MedPredict AI</h1>
      <p>Insurance-cost intelligence redesigned as a product: benchmark multiple regressors, estimate a personalized annual charge, show an uncertainty band, and explain the biggest drivers behind the prediction.</p>
      <span class="chip">{len(data):,} training records</span>
      <span class="chip">4 regression models</span>
      <span class="chip">5-fold cross-validation</span>
      <span class="chip">Counterfactual explanations</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Build your profile")
    age = st.slider("Age", 18, 64, 33)
    sex = st.selectbox("Gender", ["Female", "Male"])
    measurement = st.radio("BMI input", ["Direct BMI", "Height + weight"], horizontal=True)
    if measurement == "Direct BMI":
        bmi = st.slider("BMI", 16.0, 53.0, 27.5, 0.1)
        height_cm = weight_kg = None
    else:
        height_cm = st.number_input("Height (cm)", 140.0, 210.0, 175.0, 0.5)
        weight_kg = st.number_input("Weight (kg)", 40.0, 160.0, 75.0, 0.5)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        st.caption(f"Calculated BMI: **{bmi:.1f}** · {bmi_category(bmi)}")
    children = st.slider("Children / dependents", 0, 5, 1)
    smoker = st.selectbox("Smoking status", ["No", "Yes"])
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
    st.caption("Prediction updates live as you change the profile.")

row = make_input_frame(age, sex, bmi, children, smoker, region)
pred = float(bundle.best_model.predict(row)[0])
res_q10, res_q90 = np.quantile(bundle.residuals, [0.10, 0.90])
low = max(0.0, pred + res_q10)
high = max(low, pred + res_q90)

summary = data.describe(numeric_only=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Model selected</div><div class="metric-value">{bundle.metrics.iloc[0]["Model"]}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Holdout RMSE</div><div class="metric-value">{money(bundle.metrics.iloc[0]["Holdout RMSE ($)"])}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Holdout R²</div><div class="metric-value">{bundle.metrics.iloc[0]["Holdout R²"]:.3f}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Training population</div><div class="metric-value">{len(data):,}</div></div>', unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictor", "📊 Model Lab", "🔍 Explainability", "🧭 About"])

with tab1:
    st.markdown('<div class="section-title"><h2>Your estimate</h2></div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            f'<div class="prediction"><div class="metric-label">PERSONALIZED ML ESTIMATE</div><div class="value">{money(pred)}</div><div class="sub">80% empirical prediction band: {money(low)} – {money(high)}</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(gauge(pred, low, high), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("### Profile snapshot")
        p1, p2 = st.columns(2)
        p1.metric("Age", f"{age} yrs")
        p2.metric("BMI", f"{bmi:.1f}")
        p1.metric("Dependents", children)
        p2.metric("Smoker", smoker)
        st.markdown(f"**Region:** {region}  \n**BMI category:** {bmi_category(bmi)}")
        st.caption("Educational ML estimate only. It is not an insurance quote, eligibility decision, or medical recommendation.")
        report = {
            "model": bundle.metrics.iloc[0]["Model"],
            "estimate": round(pred, 2),
            "prediction_band_80pct": [round(low, 2), round(high, 2)],
            "profile": {
                "age": int(age), "sex": sex, "bmi": round(float(bmi), 2),
                "children": int(children), "smoker": smoker, "region": region,
            },
        }
        st.download_button("Download prediction report", data=json.dumps(report, indent=2), file_name="medpredict_report.json", mime="application/json")

    st.markdown("### Fast what-if scenarios")
    scenarios = []
    for label, new_smoker in [("Quit smoking", "no"), ("Current smoking status", smoker.lower())]:
        test = row.copy()
        test.loc[0, "smoker"] = new_smoker
        scenarios.append((label, float(bundle.best_model.predict(test)[0])))
    bmi_test = row.copy()
    bmi_test.loc[0, "bmi"] = max(16, bmi - 2)
    scenarios.append(("BMI −2", float(bundle.best_model.predict(bmi_test)[0])))
    bmi_test.loc[0, "bmi"] = min(53, bmi + 2)
    scenarios.append(("BMI +2", float(bundle.best_model.predict(bmi_test)[0])))

    cols = st.columns(4)
    for col, (label, value) in zip(cols, scenarios):
        col.metric(label, money(value), delta=money(value - pred))

    st.markdown("### How the estimate sits in the dataset")
    fig = px.histogram(data, x=TARGET, nbins=35, marginal="box", title="Observed insurance charges")
    fig.add_vline(x=pred, line_dash="dash", annotation_text="Your estimate")
    fig.update_layout(height=380, margin=dict(l=20,r=20,t=55,b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab2:
    st.markdown("### Model benchmark")
    m = bundle.metrics.copy()
    money_cols = [c for c in m.columns if "MAE" in c or "RMSE" in c]
    for c in money_cols:
        m[c] = m[c].round(0)
    for c in ["CV R²", "Holdout R²"]:
        m[c] = m[c].round(3)
    st.dataframe(m, use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    with a:
        fig = px.bar(m, x="Model", y="CV RMSE ($)", title="5-fold CV RMSE — lower is better")
        fig.update_layout(height=350, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with b:
        fig = px.bar(m, x="Model", y="Holdout R²", title="Holdout R² — higher is better", range_y=[0,1])
        fig.update_layout(height=350, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Dataset explorer")
    e1, e2, e3 = st.columns(3)
    e1.metric("Mean charge", money(summary.loc["mean", TARGET]))
    e2.metric("Median charge", money(summary.loc["50%", TARGET]))
    e3.metric("Max charge", money(summary.loc["max", TARGET]))

    filter_smoker = st.multiselect("Filter smoking status", sorted(data["smoker"].unique()), default=sorted(data["smoker"].unique()))
    filtered = data[data["smoker"].isin(filter_smoker)]
    fig = px.scatter(filtered, x="bmi", y="charges", color="smoker", hover_data=["age", "children", "region"], title="BMI vs insurance charge")
    fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab3:
    st.markdown("### Why did the model produce this estimate?")
    baseline = pd.DataFrame([{
        "age": data["age"].median(), "bmi": data["bmi"].median(), "children": data["children"].median(),
        "sex": data["sex"].mode()[0], "smoker": data["smoker"].mode()[0], "region": data["region"].mode()[0],
    }])
    impacts = scenario_impacts(bundle.best_model, row, baseline)
    impacts["Feature"] = impacts["Feature"].map(format_feature_label)

    x, y = st.columns([1.05, 1])
    with x:
        chart = px.bar(impacts.sort_values("Impact ($)"), x="Impact ($)", y="Feature", orientation="h", title="Local counterfactual impact")
        chart.add_vline(x=0, line_width=1)
        chart.update_layout(height=420, margin=dict(l=10,r=10,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Change in predicted charge vs reference profile", yaxis_title=None)
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    with y:
        st.markdown("#### Reference profile")
        st.json({
            "age": int(baseline.loc[0, "age"]),
            "bmi": round(float(baseline.loc[0, "bmi"]), 1),
            "children": int(baseline.loc[0, "children"]),
            "sex": baseline.loc[0, "sex"],
            "smoker": baseline.loc[0, "smoker"],
            "region": baseline.loc[0, "region"],
        })
        st.caption("A positive impact means the current feature value increases the prediction relative to the reference profile; a negative impact means it lowers it. This is an intuitive counterfactual explanation, not a causal claim.")

    st.markdown("### Global feature signal")
    global_ref = data.sample(min(250, len(data)), random_state=42)
    global_pred = bundle.best_model.predict(global_ref.drop(columns=[TARGET], errors="ignore"))
    importances = []
    for feature in ["age", "bmi", "children", "sex", "smoker", "region"]:
        changed = global_ref.drop(columns=[TARGET], errors="ignore").copy()
        changed[feature] = baseline.loc[0, feature]
        alt_pred = bundle.best_model.predict(changed)
        importances.append({"Feature": format_feature_label(feature), "Mean abs impact ($)": float(np.mean(np.abs(global_pred - alt_pred)))})
    gi = pd.DataFrame(importances).sort_values("Mean abs impact ($)", ascending=False)
    st.dataframe(gi.round(0), use_container_width=True, hide_index=True)

with tab4:
    st.markdown("### Project architecture")
    st.code(
        "Data → Validation → One-Hot Encoding → 4 Regressors → 5-Fold CV → Best Model → Prediction + Uncertainty → Explainability",
        language="text",
    )
    st.markdown("""
    **What changed from the internship prototype**

    - Replaced manual category encoding with a reusable `ColumnTransformer` pipeline.
    - Benchmarked multiple regressors instead of presenting three unvalidated predictions.
    - Added cross-validation, holdout metrics, model selection, and a persisted model artifact.
    - Added an empirical prediction interval from out-of-fold residuals.
    - Added local counterfactual explanations and global feature-impact analysis.
    - Rebuilt the Streamlit experience around a product-style dashboard rather than a basic form.
    - Added dataset exploration, scenario analysis, and clear model limitations.
    """)
    st.markdown("### Resume-ready positioning")
    st.markdown("**MedPredict AI — Explainable Medical Insurance Cost Intelligence**")
    st.write("An end-to-end regression product that compares four ML models using 5-fold cross-validation, selects the best estimator, serves predictions through Streamlit, quantifies empirical uncertainty, and generates counterfactual explanations from a real insurance dataset.")
    st.markdown("### Repository structure")
    st.code("""app.py
src/
  config.py
  data.py
  model.py
  explain.py
scripts/
  train.py
artifacts/
  model_bundle.joblib
insurance.csv
requirements.txt
README.md
""", language="text")

st.markdown("<div class='small-note'>Built as an educational machine-learning portfolio project. Model outputs should not be used as a substitute for insurer quotes, underwriting decisions, or medical advice.</div>", unsafe_allow_html=True)
