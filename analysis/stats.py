import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

FEATURE_COLS = [
    "pupil_mean","pupil_std","pupil_min","pupil_trend",
    "blink_mean","blink_max","blink_volatility",
    "saccade_mean","saccade_min","saccade_std",
    "fixation_mean","fixation_max","fixation_std",
    "gaze_dispersion_x","gaze_dispersion_y","gaze_path_length",
    "blink_pupil_ratio","fixation_entropy"
]

COLOR_MAP = {"fresh": "#1D9E75", "moderate": "#EF9F27", "fatigued": "#E24B4A"}


def run_anova(df: pd.DataFrame) -> dict:
    """One-way ANOVA for each feature across fatigue groups."""
    results = {}
    groups = [df[df["fatigue_level"]==i] for i in [0,1,2]]
    for col in FEATURE_COLS:
        f_stat, p_val = stats.f_oneway(
            *[g[col].values for g in groups])
        results[col] = {"f_stat": round(f_stat,3), "p_value": round(p_val,6)}
    return results


def pca_analysis(df: pd.DataFrame):
    """PCA to 2D for visualisation."""
    X = StandardScaler().fit_transform(df[FEATURE_COLS])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    return coords, explained


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df[FEATURE_COLS].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values,2), texttemplate="%{text}",
        showscale=True))
    fig.update_layout(title="Feature correlation matrix",
                      height=600, margin=dict(l=20,r=20,t=40,b=20))
    return fig


def pca_scatter(df: pd.DataFrame) -> go.Figure:
    coords, explained = pca_analysis(df)
    fig = px.scatter(
        x=coords[:,0], y=coords[:,1],
        color=df["fatigue_label"],
        color_discrete_map=COLOR_MAP,
        labels={"x": f"PC1 ({explained[0]:.1%} var)",
                "y": f"PC2 ({explained[1]:.1%} var)"},
        title="PCA — eye-tracking feature space by fatigue level",
        opacity=0.7)
    fig.update_traces(marker_size=5)
    fig.update_layout(height=500, margin=dict(l=20,r=20,t=40,b=20))
    return fig


def feature_importance_bar(results_path: str = "models/results.json") -> go.Figure:
    with open(results_path) as f:
        data = json.load(f)
    imp = data["feature_importances"]
    features = list(imp.keys())[:10]
    values   = [imp[k] for k in features]
    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color="#1D9E75"))
    fig.update_layout(
        title="Top 10 features — Random Forest importance",
        xaxis_title="Importance", yaxis=dict(autorange="reversed"),
        height=400, margin=dict(l=20,r=20,t=40,b=20))
    return fig


def pupil_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in COLOR_MAP.items():
        subset = df[df["fatigue_label"]==label]["pupil_mean"]
        fig.add_trace(go.Violin(
            y=subset, name=label, fillcolor=color,
            line_color=color, opacity=0.7, box_visible=True,
            meanline_visible=True))
    fig.update_layout(
        title="Pupil diameter distribution by fatigue level",
        yaxis_title="Mean pupil diameter (normalized)",
        height=400, margin=dict(l=20,r=20,t=40,b=20))
    return fig


if __name__ == "__main__":
    df = pd.read_csv("data/features.csv")
    print("Running ANOVA...")
    anova = run_anova(df)
    sig = {k:v for k,v in anova.items() if v["p_value"] < 0.05}
    print(f"{len(sig)}/{len(FEATURE_COLS)} features significant (p<0.05)")
    for k,v in list(sig.items())[:5]:
        print(f"  {k}: F={v['f_stat']}, p={v['p_value']}")

    Path("analysis/output").mkdir(parents=True, exist_ok=True)
    with open("analysis/output/anova_results.json","w") as f:
        json.dump(anova, f, indent=2)
    print("ANOVA results saved.")