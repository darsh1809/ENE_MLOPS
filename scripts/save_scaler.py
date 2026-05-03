"""
scripts/save_scaler.py
-----------------------
Re-trains and saves the StandardScaler used in train_model.py
so the Flask API can apply the same transformation at inference time.

Run this AFTER train_model.py has produced customer_segments.csv:
    python scripts/save_scaler.py

Output:  models/scaler.pkl
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

SEGMENTS_PATH = os.getenv("SEGMENTS_PATH", "data/customer_segments.csv")
MODELS_DIR    = "models"
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    if not os.path.exists(SEGMENTS_PATH):
        print(f"ERROR: '{SEGMENTS_PATH}' not found. Run train_model.py first.")
        sys.exit(1)

    print(f"Loading RFM data from {SEGMENTS_PATH}...")
    rfm = pd.read_csv(SEGMENTS_PATH)

    features = ["Recency", "Frequency", "Monetary"]
    missing  = [f for f in features if f not in rfm.columns]
    if missing:
        print(f"ERROR: Columns missing from segments CSV: {missing}")
        sys.exit(1)

    X = rfm[features]
    scaler = StandardScaler()
    scaler.fit(X)

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"  mean_  = {scaler.mean_}")
    print(f"  scale_ = {scaler.scale_}")


if __name__ == "__main__":
    main()
