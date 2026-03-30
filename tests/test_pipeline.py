import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.generate import generate_dataset, generate_session
from features.engineer import engineer_features

def test_generate_session_shape():
    session = generate_session(0, fatigue_level=0)
    df = pd.DataFrame(session)
    assert len(df) == 60
    assert "pupil_diameter" in df.columns
    assert "blink_rate" in df.columns

def test_generate_session_fatigue_levels():
    for level in [0, 1, 2]:
        session = generate_session(0, fatigue_level=level)
        df = pd.DataFrame(session)
        assert df["fatigue_level"].iloc[0] == level

def test_dataset_balance():
    df = generate_dataset(n_sessions_per_class=10)
    counts = df.groupby("fatigue_level")["session_id"].nunique()
    assert len(counts) == 3
    assert counts.min() == 10

def test_feature_engineering_output():
    raw = generate_dataset(n_sessions_per_class=5)
    feat = engineer_features(raw)
    assert len(feat) == 15  # 5 sessions x 3 classes
    assert "pupil_mean" in feat.columns
    assert "blink_pupil_ratio" in feat.columns
    assert "fixation_entropy" in feat.columns
    assert feat["pupil_mean"].isna().sum() == 0

def test_pupil_decreases_with_fatigue():
    raw = generate_dataset(n_sessions_per_class=50)
    feat = engineer_features(raw)
    fresh_pupil    = feat[feat["fatigue_level"]==0]["pupil_mean"].mean()
    fatigued_pupil = feat[feat["fatigue_level"]==2]["pupil_mean"].mean()
    assert fresh_pupil > fatigued_pupil

def test_blink_increases_with_fatigue():
    raw = generate_dataset(n_sessions_per_class=50)
    feat = engineer_features(raw)
    fresh_blink    = feat[feat["fatigue_level"]==0]["blink_mean"].mean()
    fatigued_blink = feat[feat["fatigue_level"]==2]["blink_mean"].mean()
    assert fatigued_blink > fresh_blink