import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from analysis.stats import (correlation_heatmap, pca_scatter,
                              feature_importance_bar, pupil_distribution,
                              run_anova)

st.set_page_config(
    page_title="EyeFatigue Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURE_COLS = [
    "pupil_mean","pupil_std","pupil_min","pupil_trend",
    "blink_mean","blink_max","blink_volatility",
    "saccade_mean","saccade_min","saccade_std",
    "fixation_mean","fixation_max","fixation_std",
    "gaze_dispersion_x","gaze_dispersion_y","gaze_path_length",
    "blink_pupil_ratio","fixation_entropy"
]

COLOR_MAP = {0: "🟢 Fresh", 1: "🟡 Moderate", 2: "🔴 Fatigued"}
LABELS    = ["Fresh", "Moderate", "Fatigued"]

@st.cache_resource
def load_model():
    return joblib.load("models/saved/random_forest.pkl")

@st.cache_data
def load_data():
    feat = pd.read_csv("data/features.csv")
    raw  = pd.read_csv("data/raw/eye_tracking_raw.csv")
    return feat, raw

@st.cache_data
def load_results():
    with open("models/results.json") as f:
        return json.load(f)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("EyeFatigue Analyzer 🧠")
st.sidebar.markdown("*Quantifying brain fatigue from eye-tracking signals*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "Live Predictor", "Dataset Explorer",
    "Model Performance", "Statistical Analysis"])
st.sidebar.markdown("---")
st.sidebar.markdown("Built for **NeuralPort ZEN EYE Pro** research")
st.sidebar.markdown("[GitHub](https://github.com/Amankumarsingh23/EyeFatigue-Analyzer)")

# ── Page: Live Predictor ──────────────────────────────────────────────────
if page == "Live Predictor":
    st.title("Live Fatigue Predictor")
    st.markdown("Simulate an eye-tracking session and get an instant fatigue classification.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Pupil signals")
        pupil_mean  = st.slider("Avg pupil diameter", 0.3, 1.0, 0.65, 0.01)
        pupil_std   = st.slider("Pupil variability",  0.01, 0.15, 0.04, 0.01)
        pupil_min   = st.slider("Min pupil diameter", 0.2, 0.9, 0.55, 0.01)
        pupil_trend = st.slider("Pupil trend (slope)",-0.005, 0.005, -0.001, 0.0001)

    with col2:
        st.subheader("Blink signals")
        blink_mean       = st.slider("Avg blink rate (bpm)", 5.0, 40.0, 15.0, 0.5)
        blink_max        = st.slider("Max blink rate (bpm)", 5.0, 50.0, 22.0, 0.5)
        blink_volatility = st.slider("Blink variability",    0.5, 8.0, 3.0, 0.1)

    with col3:
        st.subheader("Saccade + fixation")
        saccade_mean  = st.slider("Avg saccade vel (deg/s)", 100, 500, 320, 5)
        saccade_min   = st.slider("Min saccade vel (deg/s)", 80,  400, 220, 5)
        saccade_std   = st.slider("Saccade variability",     5,   80,  30,  1)
        fixation_mean = st.slider("Avg fixation dur (ms)",   80,  500, 210, 5)
        fixation_max  = st.slider("Max fixation dur (ms)",   100, 700, 320, 5)
        fixation_std  = st.slider("Fixation variability",    5,   80,  28,  1)

    # Auto-compute derived features
    gaze_dispersion_x = 0.8 + (1.0 - pupil_mean) * 1.2
    gaze_dispersion_y = 0.6 + (1.0 - pupil_mean) * 1.0
    gaze_path_length  = 45 + (40 - saccade_mean * 0.1)
    blink_pupil_ratio = blink_mean / (pupil_mean + 1e-6)
    fixation_entropy  = 3.2 + fixation_std * 0.01

    features = np.array([[
        pupil_mean, pupil_std, pupil_min, pupil_trend,
        blink_mean, blink_max, blink_volatility,
        saccade_mean, saccade_min, saccade_std,
        fixation_mean, fixation_max, fixation_std,
        gaze_dispersion_x, gaze_dispersion_y, gaze_path_length,
        blink_pupil_ratio, fixation_entropy
    ]])

    model = load_model()
    pred  = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    st.markdown("---")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Fatigue Level", COLOR_MAP[pred])
    with col_r2:
        st.metric("Confidence", f"{proba[pred]*100:.1f}%")
    with col_r3:
        st.metric("Blink-Pupil Ratio", f"{blink_pupil_ratio:.2f}")

    import plotly.graph_objects as go
    prob_fig = go.Figure(go.Bar(
        x=LABELS, y=proba*100,
        marker_color=["#1D9E75","#EF9F27","#E24B4A"],
        text=[f"{p:.1f}%" for p in proba*100],
        textposition="outside"))
    prob_fig.update_layout(
        title="Class probability distribution",
        yaxis_title="Probability (%)", yaxis_range=[0,110],
        height=300, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(prob_fig, use_container_width=True)

# ── Page: Dataset Explorer ────────────────────────────────────────────────
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    feat_df, raw_df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total sessions",  feat_df["session_id"].nunique())
    col2.metric("Features",        len(FEATURE_COLS))
    col3.metric("Raw readings",    len(raw_df))
    col4.metric("Fatigue classes", 3)

    st.subheader("Pupil diameter by fatigue level")
    st.plotly_chart(pupil_distribution(feat_df), use_container_width=True)

    st.subheader("PCA — feature space")
    st.plotly_chart(pca_scatter(feat_df), use_container_width=True)

    st.subheader("Raw data sample")
    st.dataframe(raw_df.head(100), use_container_width=True)

# ── Page: Model Performance ───────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance")
    results = load_results()

    import plotly.graph_objects as go
    models = list(results["models"].keys())
    accs   = [results["models"][m]["accuracy"] for m in models]
    aucs   = [results["models"][m]["roc_auc"]  for m in models]
    cvs    = [results["models"][m]["cv_mean"]  for m in models]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Test accuracy", x=models, y=accs,
                              marker_color="#1D9E75"))
        fig.add_trace(go.Bar(name="CV accuracy",   x=models, y=cvs,
                              marker_color="#378ADD"))
        fig.update_layout(title="Model accuracy comparison",
                          barmode="group", yaxis_range=[0,1.05],
                          height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure(go.Bar(
            x=models, y=aucs, marker_color="#7F77DD",
            text=[f"{a:.3f}" for a in aucs], textposition="outside"))
        fig2.update_layout(title="ROC-AUC (one-vs-rest)",
                           yaxis_range=[0,1.05], height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Feature importance — Random Forest")
    st.plotly_chart(feature_importance_bar(), use_container_width=True)

    st.subheader("Raw results")
    st.json(results["models"])

# ── Page: Statistical Analysis ────────────────────────────────────────────
elif page == "Statistical Analysis":
    st.title("Statistical Analysis")
    feat_df, _ = load_data()

    st.subheader("ANOVA — which features differ across fatigue groups?")
    anova = run_anova(feat_df)
    anova_df = pd.DataFrame([
        {"feature": k, "F-stat": v["f_stat"], "p-value": v["p_value"],
         "significant": "Yes" if v["p_value"] < 0.05 else "No"}
        for k,v in anova.items()
    ]).sort_values("F-stat", ascending=False)
    st.dataframe(anova_df, use_container_width=True)

    st.subheader("Feature correlation matrix")
    st.plotly_chart(correlation_heatmap(feat_df), use_container_width=True)