"""
FraudShield — Golden Dataset Builder (v2)

Replaces the original Evals.py. Two substantive changes:

1. SAMPLES FROM HELD-OUT ROWS ONLY.
   The original loaded the whole creditcard.csv and sampled from all of it.
   About 80% of those rows were rows the model trained on, where its
   probabilities are optimistically biased by memorisation. A "borderline"
   case picked from training data is not borderline for a fresh transaction --
   so the hardest band, the one the whole exercise exists to test, was the
   least trustworthy. This version reads the test index written by
   train_pipeline.py and samples only from there.

2. NO SCALER. Scoring goes through serving.score(), the same path Streamlit
   and LangGraph use, so the probabilities in the golden set are by
   construction the probabilities production would produce.

Run:
    python train_pipeline.py      # writes artifacts/test_index_*.json
    python build_golden_dataset.py
"""

import json
import pandas as pd

import fraudshield_config as cfg
from serving import score, load_bundle

SAMPLES_PER_BAND = 6
RANDOM_STATE = 42

# Bands on model probability. The point is to cover every regime, not just
# obvious-fraud and obvious-clean -- agent disagreement lives in the middle.
BANDS = [
    ("very_low",   0.00, 0.10),
    ("low",        0.10, 0.30),
    ("borderline", 0.30, 0.55),   # the hardest and most valuable band
    ("elevated",   0.55, 0.70),
    ("high",       0.70, 0.90),
    ("very_high",  0.90, 1.00),
]


def main():
    bundle = load_bundle()
    df = pd.read_csv(cfg.DATA_PATH)

    test_index_path = cfg.ARTIFACT_DIR / f"test_index_{cfg.ARTIFACT_VERSION}.json"
    if not test_index_path.exists():
        raise FileNotFoundError(
            f"{test_index_path.name} not found. Run train_pipeline.py first -- "
            "the golden set must come from held-out rows."
        )
    test_idx = json.loads(test_index_path.read_text())
    df = df.loc[test_idx].copy()
    print(f"Sampling from {len(df):,} held-out rows (model version {bundle['version']})")

    scored = score(df, bundle)
    df["model_fraud_probability"] = scored["fraud_probability"]
    df["hour_of_day"] = scored["hour_of_day"]
    df["ground_truth_class"] = df["Class"]

    samples = []
    for name, lo, hi in BANDS:
        mask = (df["model_fraud_probability"] >= lo)
        mask &= (df["model_fraud_probability"] <= hi) if hi == 1.00 else (df["model_fraud_probability"] < hi)
        band_df = df[mask]
        n = min(SAMPLES_PER_BAND, len(band_df))
        if n == 0:
            print(f"{name:12s}: 0 available -- band not represented")
            continue
        sampled = band_df.sample(n=n, random_state=RANDOM_STATE).copy()
        sampled["risk_band"] = name
        samples.append(sampled)
        print(f"{name:12s}: sampled {n} / {len(band_df)} available")

    golden = pd.concat(samples).reset_index(drop=True)
    golden.insert(0, "case_id", [f"GOLD-{i+1:03d}" for i in range(len(golden))])

    # Columns you fill in by hand. This IS the golden label set.
    golden["expected_document_verification"] = ""   # valid / invalid
    golden["expected_risk_tier"] = ""               # low / medium / high
    golden["expected_compliance_flag"] = ""         # pass / review / block
    golden["expected_escalation"] = ""              # auto_approve / human_review / auto_reject
    golden["expected_case_note_summary"] = ""       # 1-2 sentence gold summary for the LLM judge
    golden["labeling_notes"] = ""                   # why you labeled it this way

    lead = ["case_id", "risk_band", "model_fraud_probability",
            "ground_truth_class", "Amount", "hour_of_day"]
    expected_cols = [c for c in golden.columns if c.startswith("expected_")]
    rest = [c for c in golden.columns
            if c not in lead + expected_cols + ["labeling_notes", "Class"]]
    golden = golden[lead + expected_cols + ["labeling_notes"] + rest]

    out = cfg.EVAL_DIR / "golden_dataset_template.csv"
    golden.to_csv(out, index=False)

    # Provenance: which model produced these probabilities. Without this, a
    # golden set labelled against v1 scores silently gets reused against v3.
    (cfg.EVAL_DIR / "golden_dataset_provenance.json").write_text(json.dumps({
        "model_version": bundle["version"],
        "feature_version": bundle["feature_version"],
        "threshold": bundle["threshold"],
        "sampled_from": "held-out test index",
        "n_cases": int(len(golden)),
    }, indent=2))

    print(f"\nSaved {out.name} with {len(golden)} rows.")
    print("Next: fill in the expected_* columns by hand, then run eval_runner.py.")


if __name__ == "__main__":
    main()
