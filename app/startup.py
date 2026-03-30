import subprocess, sys
from pathlib import Path

def run_pipeline():
    if not Path("data/raw/eye_tracking_raw.csv").exists():
        subprocess.run([sys.executable, "data/generate.py"], check=True)
    if not Path("data/features.csv").exists():
        subprocess.run([sys.executable, "features/engineer.py"], check=True)
    if not Path("models/saved/random_forest.pkl").exists():
        subprocess.run([sys.executable, "models/train.py"], check=True)
    if not Path("analysis/output/anova_results.json").exists():
        subprocess.run([sys.executable, "analysis/stats.py"], check=True)