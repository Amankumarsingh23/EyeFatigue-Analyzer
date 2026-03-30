# EyeFatigue Analyzer 🧠

> *The eyes are a window into the brain. This project reads that window.*

A complete machine learning pipeline that ingests eye-tracking signals — gaze position, pupil diameter, blink rate, saccade velocity, fixation duration — engineers 18 scientifically grounded features, trains three classifiers, and serves everything through an interactive Streamlit dashboard with live fatigue prediction.

Built to mirror the data science challenges behind **NeuralPort's ZEN EYE Pro** — a system that scores brain fatigue from eye data in under one minute.

---

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![CI](https://github.com/Amankumarsingh23/EyeFatigue-Analyzer/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

**[Live Demo →](https://eyefatigue-analyzer.streamlit.app)**

---

## The problem this solves

Brain fatigue is invisible — until it's too late. Athletes lose focus mid-game. Drivers miss signals. Students stop absorbing. The challenge NeuralPort is solving is not just measuring fatigue, but measuring it *fast*, *non-invasively*, and *from a signal people already produce naturally* — their eye movements.

This project builds a proof-of-concept for the statistical and ML pipeline that sits behind such a system: from raw gaze readings to a classified fatigue state in milliseconds.

---

## Pipeline overview

```
Raw eye-tracking data (60 samples/session)
            │
            ▼
┌─────────────────────────┐
│   data/generate.py      │  Synthetic sessions — 3 fatigue classes
│   5,000 sessions        │  pupil · blink · saccade · fixation · gaze
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  features/engineer.py   │  18 features per session
│                         │  rolling stats · entropy · composite ratios
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   models/train.py       │  3 classifiers — cross-validated
│                         │  Random Forest · Logistic Regression · SVM
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   analysis/stats.py     │  Multivariate statistics
│                         │  PCA · ANOVA · correlation heatmap
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   app/dashboard.py      │  Streamlit — live interactive dashboard
│                         │  predict · explore · analyse · compare
└─────────────────────────┘
```

---

## Features engineered

The pipeline extracts 18 features from each 60-second eye-tracking session, grounded in psychophysiology research:

| Category | Feature | Why it matters |
|----------|---------|---------------|
| Pupil | `pupil_mean` | Smaller pupil → higher cognitive load |
| Pupil | `pupil_std` | High variability → unstable arousal |
| Pupil | `pupil_min` | Floor of pupil response |
| Pupil | `pupil_trend` | Slope over session — fatigue building up |
| Blink | `blink_mean` | Blink rate rises with fatigue |
| Blink | `blink_max` | Peak blink episodes |
| Blink | `blink_volatility` | Erratic blinking = cognitive disruption |
| Saccade | `saccade_mean` | Slower saccades → reduced alertness |
| Saccade | `saccade_min` | Lowest velocity — extreme fatigue moments |
| Saccade | `saccade_std` | Inconsistency in eye movement speed |
| Fixation | `fixation_mean` | Longer fixations → slower processing |
| Fixation | `fixation_max` | Extreme dwelling — attention stalling |
| Fixation | `fixation_std` | Variability in attention allocation |
| Gaze | `gaze_dispersion_x` | Wider scatter → loss of focus |
| Gaze | `gaze_dispersion_y` | Vertical drift under fatigue |
| Gaze | `gaze_path_length` | Total gaze travel — search efficiency |
| Composite | `blink_pupil_ratio` | Combined arousal signal |
| Composite | `fixation_entropy` | Information-theoretic attention measure |

---

## Model results

Three classifiers trained with 5-fold stratified cross-validation on 1,500 sessions (80/20 split):

| Model | CV Accuracy | Test Accuracy | ROC-AUC |
|-------|-------------|---------------|---------|
| Random Forest | ~0.94 ± 0.01 | ~0.95 | ~0.99 |
| Logistic Regression | ~0.88 ± 0.02 | ~0.89 | ~0.97 |
| SVM (RBF) | ~0.91 ± 0.01 | ~0.92 | ~0.98 |

*Results vary slightly by random seed. Run `python models/train.py` to reproduce.*

**Top predictive features (Random Forest importance):**
1. `blink_pupil_ratio` — strongest single signal
2. `pupil_mean` — core fatigue indicator
3. `saccade_mean` — velocity degradation
4. `fixation_mean` — attention slowing
5. `gaze_path_length` — search efficiency loss

---

## Dashboard pages

### Live Predictor
Adjust 11 eye-tracking sliders in real time. The Random Forest model classifies your simulated session instantly, showing the predicted fatigue level, confidence score, and a probability bar chart across all three classes.

### Dataset Explorer
Violin plots of pupil distribution across fatigue classes. PCA scatter showing how well the 18 features separate the three groups in 2D. Raw data table with 100-row preview.

### Model Performance
Side-by-side accuracy and ROC-AUC comparison across all three models. Random Forest feature importance bar chart. Raw results JSON.

### Statistical Analysis
One-way ANOVA table — which features show statistically significant differences across fatigue groups (spoiler: all 18, p < 0.001). Full 18×18 feature correlation heatmap.

---

## Project structure

```
EyeFatigue-Analyzer/
│
├── data/
│   ├── generate.py              # Synthetic eye-tracking data generator
│   └── raw/
│       └── eye_tracking_raw.csv # Generated dataset (5000 sessions × 60s)
│
├── features/
│   └── engineer.py              # 18-feature extraction pipeline
│
├── models/
│   ├── train.py                 # RF + LogReg + SVM training + evaluation
│   ├── results.json             # CV scores, accuracy, ROC-AUC, importances
│   └── saved/
│       ├── random_forest.pkl
│       ├── logistic_regression.pkl
│       └── svm.pkl
│
├── analysis/
│   ├── stats.py                 # PCA, ANOVA, correlation, Plotly charts
│   └── output/
│       └── anova_results.json
│
├── app/
│   ├── dashboard.py             # Streamlit application
│   └── startup.py               # Auto-runs pipeline if data missing
│
├── tests/
│   └── test_pipeline.py         # 6 pytest tests
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions — full pipeline + tests
│
├── requirements.txt
└── README.md
```

---

## Getting started

### Install dependencies

```bash
git clone https://github.com/Amankumarsingh23/EyeFatigue-Analyzer.git
cd EyeFatigue-Analyzer

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac / Linux

pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Step 1 — generate data
python data/generate.py

# Step 2 — engineer features
python features/engineer.py

# Step 3 — train models
python models/train.py

# Step 4 — run statistical analysis
python analysis/stats.py

# Step 5 — launch dashboard
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

### Run tests

```bash
pytest tests/ -v
```

Expected output:
```
test_generate_session_shape          PASSED
test_generate_session_fatigue_levels PASSED
test_dataset_balance                 PASSED
test_feature_engineering_output      PASSED
test_pupil_decreases_with_fatigue    PASSED
test_blink_increases_with_fatigue    PASSED
6 passed in 4.2s
```

---

## The science behind the data

The synthetic data is not random noise — it encodes real psychophysiological relationships:

**Pupil diameter** shrinks under cognitive load (the pupillary light reflex is overridden by the locus coeruleus during mental effort). Fresh subjects show normalized diameter ~0.72; fatigued subjects drop to ~0.50.

**Blink rate** follows a U-shaped curve with alertness — it rises from ~12 bpm when fresh to ~26 bpm when fatigued, as the inhibitory mechanism that suppresses blinking during focused tasks weakens.

**Saccade velocity** declines with fatigue. The main sequence relationship (amplitude vs peak velocity) degrades as the superior colliculus firing rate decreases under fatigue — modelled here as a drop from ~380 to ~210 deg/sec.

**Fixation duration** lengthens as cognitive processing slows — fresh subjects show ~180ms fixations; fatigued subjects extend to ~320ms as the brain requires more time per visual unit.

---

## Connecting to real hardware

The pipeline expects a CSV with columns matching the raw format. To swap in real data from an eye tracker:

```python
# From Tobii Pro SDK
gaze_data = tobii_controller.get_gaze_data()
pupil = gaze_data.left_eye.pupil_diameter

# From PICO 4 Enterprise
PXR_EyeTracking.GetCombineEyeGazePoint(out gazePoint)

# From SRanipal (HTC Vive Pro Eye)
SRanipal_Eye_API.GetEyeData(ref eyeData)
```

Format your readings as:
```
session_id, second, pupil_diameter, blink_rate, saccade_velocity, fixation_duration, gaze_x, gaze_y
```

Then run `features/engineer.py` and `models/train.py` on your real data.

---

## CI/CD

GitHub Actions runs the entire pipeline on every push:

```yaml
Generate data → Engineer features → Train models → Run stats → pytest
```

The CI badge at the top of this README reflects the current pipeline health.

---

## Roadmap

- [ ] LSTM-based sequential model on raw 60-frame time series
- [ ] Unsupervised clustering (K-means + DBSCAN) for ZONE detection
- [ ] Real biometric data integration (Tobii / PICO / SRanipal)
- [ ] REST API endpoint wrapping the trained classifier
- [ ] Per-session longitudinal fatigue tracking over multiple sessions
- [ ] Export reports as PDF for athlete performance review

---

## Related project

**[GazeID VR](https://github.com/Amankumarsingh23/GazeID-VR)** — Unity (C#) VR application that identifies users by gaze patterns and computes fatigue scores in real time using OpenXR and XR Interaction Toolkit. The ML pipeline in this repo is the data science foundation; GazeID VR is the VR product layer on top.

---

## Built by

**Aman Kumar Singh**
3rd Year B.Tech, Material Science and Engineering — IIT Kanpur
Codeforces Specialist (peak 1582) · 400+ problems solved

[LinkedIn](https://linkedin.com/in/aman-singh-iitkanpur) &nbsp;·&nbsp; [GitHub](https://github.com/Amankumarsingh23) &nbsp;·&nbsp; [NeuroVR Dashboard](https://github.com/Amankumarsingh23/NeuroVR-Dashboard)

---

*"NeuralPort quantifies the invisible phenomenon of brain fatigue and scientifically recreates the moment humans enter their ZONE of peak focus and alertness."*
*This project is built to contribute to exactly that mission.*
