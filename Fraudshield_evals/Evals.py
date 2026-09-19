"""
FraudShield — Golden Dataset Builder
=====================================
Run this LOCALLY in your FraudShield repo (where creditcard.csv lives).

What it does:
1. Loads creditcard.csv
2. Runs your trained XGBoost model to get a fraud probability per transaction
3. Stratifies transactions into risk bands so your golden set isn't just
   "all obvious fraud" or "all obvious clean" — it covers the hard cases
   your agents actually need to be tested on
4. Samples N transactions per band
5. Outputs golden_dataset_template.csv — a hand-labeling template you fill in

Usage:
    python build_golden_dataset.py
"""

import pandas as pd
import joblib  # or use your existing model-loading code

# ---- CONFIG ----
DATA_PATH = "../creditcard.csv"
MODEL_PATH = "../fraudshield_model.pkl"   # update to your actual saved model path
SAMPLES_PER_BAND = 6               # 6 bands x 6 = ~36 rows, good golden-set size
RANDOM_STATE = 42

# Risk bands based on model probability — this is the key idea.
# You want cases in EVERY band, not just "clear fraud" and "clear clean",
# because that's where agent disagreement/errors actually happen.
BANDS = [
    ("very_low",  0.00, 0.10),
    ("low",       0.10, 0.30),
    ("borderline",0.30, 0.55),   # the hardest, most valuable band
    ("elevated",  0.55, 0.70),
    ("high",      0.70, 0.90),
    ("very_high", 0.90, 1.00),
]


def main():
    df = pd.read_csv(DATA_PATH)
    y_true = df ["Class"]
    X = df.drop(columns=["Class"])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X["Amount_scaled"] = scaler.fit_transform(X[["Amount"]])
    X["Time_scaled"] = scaler.fit_transform(X[["Time"]])
    X = X.drop(columns=["Amount" , "Time"])

    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(X)[:, 1]
    df["model_fraud_probability"] = proba
    df["ground_truth_class"] = y_true  # 1 = actual fraud, 0 = actual clean

    samples = []
    for band_name, lo, hi in BANDS:
        if hi == 1.00:
            band_df = df[(df["model_fraud_probability"] >= lo) &
                         (df["model_fraud_probability"] <= hi)]
        else:
            band_df = df [(df["model_fraud_probability"] >= lo) &
                          (df["model_fraud_probability"] < hi)]
        n = min(SAMPLES_PER_BAND, len(band_df))
        sampled = band_df.sample(n=n, random_state=RANDOM_STATE)
        sampled = sampled.copy()
        sampled["risk_band"] = band_name
        samples.append(sampled)
        print(f"{band_name:12s}: sampled {n} / {len(band_df)} available")

    golden = pd.concat(samples).reset_index(drop=True)
    golden.insert(0, "case_id", [f"GOLD-{i+1:03d}" for i in range(len(golden))])

    # Columns you (the human) fill in by hand — this IS the golden label set
    golden["expected_document_verification"] = ""   # valid / invalid / missing
    golden["expected_risk_tier"] = ""                # low / medium / high
    golden["expected_compliance_flag"] = ""          # pass / review / block
    golden["expected_escalation"] = ""               # auto_approve / human_review / auto_reject
    golden["expected_case_note_summary"] = ""         # 1-2 sentence gold summary for LLM-judge comparison
    golden["labeling_notes"] = ""                     # why you labeled it this way — useful for interviews too

    expected_cols = [c for c in golden.columns if c.startswith("expected_")]
    other_cols = [c for c in golden.columns
                  if c not in ["case_id" , "risk_band" , "model_fraud_probability" , "ground_truth_class" , "labeling_notes"]
                  and not c.startswith("expected_")]
    out_cols = (["case_id" , "risk_band" , "model_fraud_probability" , "ground_truth_class"]
                + expected_cols
                +["labeling_notes"]
                + other_cols)

    golden = golden[out_cols]
    golden.to_csv("golden_dataset_template.csv", index=False)
    print(f"\nSaved golden_dataset_template.csv with {len(golden)} rows.")
    print("Next: open it, and for each case_id fill in the 'expected_*' columns by hand.")


if __name__ == "__main__":
    main()