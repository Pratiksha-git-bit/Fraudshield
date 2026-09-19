"""
FraudShield — training pipeline.

Replaces EDA cells 9 through 16.

What changed vs the notebook:
  - Time -> hour_of_day (stationary), Amount kept raw, nothing scaled.
  - No fitted transform anywhere in feature engineering, so training/serving
    skew is structurally impossible rather than merely avoided.
  - SMOTE is fit on TRAIN ONLY (it already was, but it is worth being explicit:
    oversampling the test set would invent fraud cases and inflate recall).
  - Full threshold sweep instead of five hardcoded prints.
  - A promotion gate, and a saved test-index file so evals can sample from
    held-out rows only.

Run:
    pip install mlflow
    python train_pipeline.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_score, recall_score, f1_score,
    brier_score_loss,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import fraudshield_config as cfg
from features import engineer, feature_names, CYCLIC_HOUR


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        # Brier will look bad on a SMOTE-trained model. That is the point:
        # SMOTE distorts the base rate, so these are ranking scores, not
        # calibrated probabilities. Know this before calling 0.7 "70% likely".
        "brier": float(brier_score_loss(y_true, proba)),
    }


def main():
    df = pd.read_csv(cfg.DATA_PATH)
    y = df["Class"]

    # Engineer first (stateless, so order vs the split does not matter here --
    # that is exactly the property we wanted), then split.
    X = engineer(df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y,
    )

    # Persist the held-out index so the golden dataset can be sampled from
    # test rows only. Sampling eval cases from rows the model memorised makes
    # the whole agent evaluation optimistic.
    test_index_path = cfg.ARTIFACT_DIR / f"test_index_{cfg.ARTIFACT_VERSION}.json"
    test_index_path.write_text(json.dumps([int(i) for i in X_te.index]))

    sm = SMOTE(random_state=cfg.RANDOM_STATE)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)

    model = XGBClassifier(**cfg.XGB_PARAMS)
    model.fit(X_tr_sm, y_tr_sm)

    proba = model.predict_proba(X_te)[:, 1]

    sweep = [evaluate(y_te, proba, t) for t in np.round(np.arange(0.05, 1.00, 0.05), 2)]
    sweep_df = pd.DataFrame(sweep)
    sweep_path = cfg.ARTIFACT_DIR / f"threshold_sweep_{cfg.ARTIFACT_VERSION}.csv"
    sweep_df.to_csv(sweep_path, index=False)

    metrics = evaluate(y_te, proba, cfg.DECISION_THRESHOLD)
    cm = confusion_matrix(y_te, (proba >= cfg.DECISION_THRESHOLD).astype(int))
    tn, fp, fn, tp = cm.ravel()

    print("\n--- Threshold sweep ---")
    print(sweep_df[["threshold", "precision", "recall", "f1"]].to_string(index=False))
    print(f"\n--- At operating threshold {cfg.DECISION_THRESHOLD} ---")
    print(classification_report(y_te, (proba >= cfg.DECISION_THRESHOLD).astype(int)))
    print(f"Fraud caught: {tp} | Fraud missed: {fn} | False alarms: {fp}")
    print(f"Analyst queue per 100k txns: {(tp + fp) / len(y_te) * 100_000:.0f}")

    gate = cfg.PROMOTION_GATE
    failures = [
        f"{k} {metrics[k.replace('min_', '')]:.3f} < {v}"
        for k, v in gate.items()
        if metrics[k.replace("min_", "")] < v
    ]
    passed = not failures
    print(f"\nPromotion gate: {'PASS' if passed else 'FAIL -- ' + '; '.join(failures)}")

    bundle = {
        "version": cfg.ARTIFACT_VERSION,
        "feature_version": cfg.FEATURE_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "feature_names": feature_names(CYCLIC_HOUR),
        "cyclic_hour": CYCLIC_HOUR,
        "threshold": cfg.DECISION_THRESHOLD,
        "metrics": metrics,
        "train_rows": int(len(X_tr)),
        "train_fraud_rate": float(y_tr.mean()),
        "sampling": "SMOTE",
        "calibrated": False,
    }
    joblib.dump(bundle, cfg.BUNDLE_PATH)
    joblib.dump(bundle, cfg.BUNDLE_LATEST)
    print(f"Saved bundle -> {cfg.BUNDLE_PATH.name}")

    baseline = {
        "version": cfg.ARTIFACT_VERSION,
        "feature_version": cfg.FEATURE_VERSION,
        "n": int(len(X_tr)),
        "features": {
            c: {
                "mean": float(X_tr[c].mean()),
                "std": float(X_tr[c].std()),
                "deciles": [float(v) for v in np.quantile(X_tr[c], np.arange(0, 1.1, 0.1))],
            }
            for c in X_tr.columns
        },
        "score_deciles": [
            float(v) for v in np.quantile(model.predict_proba(X_tr)[:, 1], np.arange(0, 1.1, 0.1))
        ],
        "fraud_rate": float(y_tr.mean()),
    }
    cfg.BASELINE_STATS_PATH.write_text(json.dumps(baseline))
    print(f"Saved baseline stats -> {cfg.BASELINE_STATS_PATH.name}")
    print(f"Saved test index    -> {test_index_path.name}")

    try:
        import mlflow
        mlflow.set_experiment("fraudshield")
        with mlflow.start_run(run_name=f"xgb_{cfg.ARTIFACT_VERSION}"):
            mlflow.log_params({**cfg.XGB_PARAMS, "sampling": "SMOTE",
                               "threshold": cfg.DECISION_THRESHOLD,
                               "artifact_version": cfg.ARTIFACT_VERSION,
                               "feature_version": cfg.FEATURE_VERSION,
                               "cyclic_hour": CYCLIC_HOUR})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_metrics({"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                "gate_passed": int(passed)})
            mlflow.log_artifact(str(sweep_path))
            mlflow.log_artifact(str(cfg.BUNDLE_PATH))
            mlflow.log_artifact(str(cfg.BASELINE_STATS_PATH))
        print("Logged run to MLflow. View with: mlflow ui")
    except ImportError:
        print("mlflow not installed -- skipped tracking (pip install mlflow)")


if __name__ == "__main__":
    main()
