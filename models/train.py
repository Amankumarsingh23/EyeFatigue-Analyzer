import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score)
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    "pupil_mean","pupil_std","pupil_min","pupil_trend",
    "blink_mean","blink_max","blink_volatility",
    "saccade_mean","saccade_min","saccade_std",
    "fixation_mean","fixation_max","fixation_std",
    "gaze_dispersion_x","gaze_dispersion_y","gaze_path_length",
    "blink_pupil_ratio","fixation_entropy"
]


def train_all_models(feat_path: str = "data/features.csv"):
    df = pd.read_csv(feat_path)
    X  = df[FEATURE_COLS].values
    y  = df["fatigue_level"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=12,
                random_state=42, n_jobs=-1))
        ]),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=1.0, random_state=42))
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0,
                        probability=True, random_state=42))
        ]),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    Path("models/saved").mkdir(parents=True, exist_ok=True)

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=cv, scoring="accuracy")
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        acc     = accuracy_score(y_test, y_pred)
        auc     = roc_auc_score(y_test, y_proba, multi_class="ovr")

        print(f"  CV Accuracy:  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Test Accuracy: {acc:.3f}")
        print(f"  ROC-AUC (OVR): {auc:.3f}")
        print(classification_report(y_test, y_pred,
              target_names=["fresh","moderate","fatigued"]))

        joblib.dump(pipeline, f"models/saved/{name}.pkl")
        results[name] = {
            "cv_mean":  round(float(cv_scores.mean()), 4),
            "cv_std":   round(float(cv_scores.std()),  4),
            "accuracy": round(float(acc), 4),
            "roc_auc":  round(float(auc), 4),
        }

    # Save feature importances from RF
    rf_clf = models["random_forest"].named_steps["clf"]
    importances = dict(zip(FEATURE_COLS, rf_clf.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(),
                               key=lambda x: x[1], reverse=True))

    with open("models/results.json", "w") as f:
        json.dump({"models": results, "feature_importances": importances}, f, indent=2)

    print("\nAll models saved to models/saved/")
    print("Results saved to models/results.json")
    return results


if __name__ == "__main__":
    train_all_models()