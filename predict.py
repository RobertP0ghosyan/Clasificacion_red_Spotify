#!/usr/bin/env python3
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================

MODEL_FILE = "rf_final_v3.joblib"
NEW_FEATURES_FILE = "features_v2FINALPREDICT.csv"
OUTPUT_FILE = "predictions.csv"

# =========================
# LOAD MODEL
# =========================

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

print(f"[i] Loaded model from {MODEL_FILE}")
print(f"[i] Model expects {len(feature_columns)} features")

# =========================
# LOAD NEW DATA
# =========================

df_new = pd.read_csv(NEW_FEATURES_FILE)

print(f"[i] Loaded {len(df_new)} new samples")

# Keep pcap name if present
pcap_names = df_new["pcap"] if "pcap" in df_new.columns else None

# Drop non-feature columns
X_new = df_new.drop(columns=["pcap"], errors="ignore")

# =========================
# ALIGN FEATURE ORDER
# =========================

X_new = X_new[feature_columns]

# =========================
# PREDICT
# =========================

y_pred = model.predict(X_new)

# =========================
# SAVE RESULTS
# =========================

results = pd.DataFrame({
    "pcap": pcap_names if pcap_names is not None else range(len(y_pred)),
    "predicted_label": y_pred
})

results.to_csv(OUTPUT_FILE, index=False)

print(f"[✓] Predictions saved to '{OUTPUT_FILE}'")
