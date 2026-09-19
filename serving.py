"""
FraudShield — the one and only scoring path.

Streamlit, the LangGraph pipeline, and the eval runner all import `score()`
from here. Nothing else touches the model directly.

The bug this replaces: every caller previously did
    scaler = StandardScaler(); scaler.fit_transform(df[["Amount"]])
On a batch of 284,807 rows that silently matched training. On a SINGLE
transaction from the Streamlit form, variance is zero, so Amount_scaled and
Time_scaled both became 0.0 -- the model stopped seeing amount at all, and
still returned a confident-looking probability.

Since v2 there is no fitted transform at all (see features.py), so the class
of bug is gone rather than patched. `self_check()` keeps it gone.
"""

import functools
import joblib
import numpy as np
import pandas as pd

import fraudshield_config as cfg
from features import engineer


@functools.lru_cache(maxsize=1)
def load_bundle(path: str = None):
    bundle = joblib.load(path or cfg.BUNDLE_LATEST)
    required = {"model", "feature_names", "threshold", "version", "feature_version"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(
            f"Bundle is missing keys: {missing}. This looks like a v1 bundle -- "
            f"retrain with train_pipeline.py."
        )
    if bundle["feature_version"] != cfg.FEATURE_VERSION:
        raise ValueError(
            f"Feature version mismatch: bundle was trained with "
            f"{bundle['feature_version']}, features.py is now "
            f"{cfg.FEATURE_VERSION}. Retrain before serving."
        )
    return bundle


def score(raw: pd.DataFrame, bundle=None) -> pd.DataFrame:
    """raw: dataframe with V1..V28, Amount, Time (Class optional, ignored).
    Returns fraud_probability, the flag, and the versions used."""
    bundle = bundle or load_bundle()
    X = engineer(raw, cyclic=bundle["cyclic_hour"])

    expected = bundle["feature_names"]
    if list(X.columns) != expected:
        raise ValueError(f"Feature mismatch.\n expected {expected}\n got      {list(X.columns)}")

    proba = bundle["model"].predict_proba(X)[:, 1]
    return pd.DataFrame({
        "fraud_probability": proba,
        "flagged": proba >= bundle["threshold"],
        "hour_of_day": X["hour_of_day"] if "hour_of_day" in X else np.nan,
        "threshold": bundle["threshold"],
        "model_version": bundle["version"],
    }, index=raw.index)


def score_one(transaction: dict) -> dict:
    """Convenience wrapper for the Streamlit single-transaction form."""
    return score(pd.DataFrame([transaction])).iloc[0].to_dict()


def self_check():
    """Run after every retrain. Asserts a transaction scores identically alone
    and in a batch -- the exact invariant the old code violated. Also asserts
    hour_of_day wraps correctly across the day boundary."""
    bundle = load_bundle()
    df = pd.read_csv(cfg.DATA_PATH)
    sample = df.sample(50, random_state=0)

    batch = score(sample, bundle)["fraud_probability"].to_numpy()
    singles = np.array([
        score(sample.iloc[[i]], bundle)["fraud_probability"].iloc[0]
        for i in range(len(sample))
    ])
    delta = float(np.abs(batch - singles).max())
    print(f"[1/2] batch vs single-row max delta : {delta:.2e} -> {'PASS' if delta < 1e-9 else 'FAIL'}")
    if delta >= 1e-9:
        raise AssertionError("Training/serving skew detected -- a transform is being re-fit.")

    probe = sample.iloc[[0]].copy()
    a = score(probe, bundle)["hour_of_day"].iloc[0]
    probe["Time"] = probe["Time"] + 86_400
    b = score(probe, bundle)["hour_of_day"].iloc[0]
    ok = abs(a - b) < 1e-9
    print(f"[2/2] hour_of_day stable +24h       : {a:.3f} vs {b:.3f} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError("hour_of_day is not wrapping -- Time is still leaking in.")
    return True


if __name__ == "__main__":
    self_check()
