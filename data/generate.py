import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

def generate_session(session_id: int, fatigue_level: int) -> dict:
    """
    Generate one 60-second eye-tracking session.
    fatigue_level: 0=fresh, 1=moderate, 2=fatigued
    """
    n_samples = 60  # one reading per second

    # Pupil diameter shrinks with fatigue
    pupil_base  = [0.72, 0.62, 0.50][fatigue_level]
    pupil_noise = 0.04
    pupil = np.clip(
        np.random.normal(pupil_base, pupil_noise, n_samples) +
        np.sin(np.linspace(0, 2*np.pi, n_samples)) * 0.02,
        0.3, 1.0)

    # Blink rate increases with fatigue (blinks per minute)
    blink_base  = [12, 18, 26][fatigue_level]
    blink_rate  = np.clip(
        np.random.normal(blink_base, 3, n_samples), 5, 40)

    # Saccade velocity decreases with fatigue (deg/sec)
    sacc_base   = [380, 300, 210][fatigue_level]
    saccade_vel = np.clip(
        np.random.normal(sacc_base, 30, n_samples), 100, 600)

    # Fixation duration increases with fatigue (ms)
    fix_base    = [180, 240, 320][fatigue_level]
    fixation_dur= np.clip(
        np.random.normal(fix_base, 25, n_samples), 80, 600)

    # Gaze dispersion increases with fatigue
    gaze_x = np.random.normal(0, [0.8, 1.2, 1.8][fatigue_level], n_samples)
    gaze_y = np.random.normal(0, [0.6, 1.0, 1.5][fatigue_level], n_samples)

    return {
        "session_id":   [session_id] * n_samples,
        "second":       list(range(n_samples)),
        "fatigue_level":[fatigue_level] * n_samples,
        "fatigue_label":[["fresh","moderate","fatigued"][fatigue_level]] * n_samples,
        "pupil_diameter":   pupil.tolist(),
        "blink_rate":       blink_rate.tolist(),
        "saccade_velocity": saccade_vel.tolist(),
        "fixation_duration":fixation_dur.tolist(),
        "gaze_x":           gaze_x.tolist(),
        "gaze_y":           gaze_y.tolist(),
    }


def generate_dataset(n_sessions_per_class: int = 500) -> pd.DataFrame:
    all_rows = []
    session_id = 0

    for fatigue_level in [0, 1, 2]:
        for _ in range(n_sessions_per_class):
            session = generate_session(session_id, fatigue_level)
            df = pd.DataFrame(session)
            all_rows.append(df)
            session_id += 1

    full_df = pd.concat(all_rows, ignore_index=True)
    return full_df


if __name__ == "__main__":
    print("Generating synthetic eye-tracking dataset...")
    df = generate_dataset(n_sessions_per_class=500)
    out = Path("data/raw/eye_tracking_raw.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df.groupby("fatigue_label").size())