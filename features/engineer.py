import pandas as pd
import numpy as np
from pathlib import Path


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-session raw readings into 18 ML-ready features.
    """
    features = []

    for session_id, grp in df.groupby("session_id"):
        p  = grp["pupil_diameter"]
        b  = grp["blink_rate"]
        s  = grp["saccade_velocity"]
        f  = grp["fixation_duration"]
        gx = grp["gaze_x"]
        gy = grp["gaze_y"]

        # Pupil features
        pupil_mean     = p.mean()
        pupil_std      = p.std()
        pupil_min      = p.min()
        pupil_trend    = np.polyfit(range(len(p)), p, 1)[0]  # slope over session

        # Blink features
        blink_mean     = b.mean()
        blink_max      = b.max()
        blink_volatility = b.std()

        # Saccade features
        saccade_mean   = s.mean()
        saccade_min    = s.min()
        saccade_std    = s.std()

        # Fixation features
        fixation_mean  = f.mean()
        fixation_max   = f.max()
        fixation_std   = f.std()

        # Gaze dispersion
        gaze_dispersion_x = gx.std()
        gaze_dispersion_y = gy.std()
        gaze_path_length  = np.sqrt(np.diff(gx)**2 + np.diff(gy)**2).sum()

        # Composite: blink-pupil ratio (high = more fatigued)
        blink_pupil_ratio = blink_mean / (pupil_mean + 1e-6)

        # Fixation entropy (variability in attention)
        fix_norm = f / (f.sum() + 1e-6)
        fixation_entropy = -np.sum(fix_norm * np.log(fix_norm + 1e-9))

        label = grp["fatigue_level"].iloc[0]
        label_name = grp["fatigue_label"].iloc[0]

        features.append({
            "session_id":         session_id,
            "fatigue_level":      label,
            "fatigue_label":      label_name,
            "pupil_mean":         pupil_mean,
            "pupil_std":          pupil_std,
            "pupil_min":          pupil_min,
            "pupil_trend":        pupil_trend,
            "blink_mean":         blink_mean,
            "blink_max":          blink_max,
            "blink_volatility":   blink_volatility,
            "saccade_mean":       saccade_mean,
            "saccade_min":        saccade_min,
            "saccade_std":        saccade_std,
            "fixation_mean":      fixation_mean,
            "fixation_max":       fixation_max,
            "fixation_std":       fixation_std,
            "gaze_dispersion_x":  gaze_dispersion_x,
            "gaze_dispersion_y":  gaze_dispersion_y,
            "gaze_path_length":   gaze_path_length,
            "blink_pupil_ratio":  blink_pupil_ratio,
            "fixation_entropy":   fixation_entropy,
        })

    return pd.DataFrame(features)


if __name__ == "__main__":
    raw = pd.read_csv("data/raw/eye_tracking_raw.csv")
    print(f"Engineering features from {raw['session_id'].nunique()} sessions...")
    feat_df = engineer_features(raw)
    out = Path("data/features.csv")
    feat_df.to_csv(out, index=False)
    print(f"Saved {len(feat_df)} sessions x {feat_df.shape[1]} columns to {out}")
    print(feat_df.groupby("fatigue_label").size())