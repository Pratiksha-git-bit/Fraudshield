"""
FraudShield — feature engineering.

Every consumer (training, serving, evals, monitoring, Streamlit) calls
`engineer()` and nothing else. One function, one definition.

Two decisions encoded here, both worth being able to defend:

1. `Time` becomes `hour_of_day`.
   Raw `Time` is *seconds elapsed since the first transaction in the file*.
   It increases without bound, so in production it drifts permanently and the
   model learns a number it will never see again. Folding it modulo 86,400
   gives a stationary feature that actually carries the signal you wanted:
   time of day.

   Caveat to state out loud: the dataset never records when t=0 actually was
   in wall-clock terms, so this is a *relative* hour-of-day with an unknown
   constant offset. The cyclical structure survives the offset, so the model
   learns fine — but you cannot claim "fraud peaks at 3am". You can only claim
   "fraud concentrates in a consistent low-volume window."

2. Nothing is scaled.
   StandardScaler is provably inert for gradient-boosted trees: splits are
   chosen on thresholds, and any strictly monotonic transform produces an
   identical tree at a transformed threshold. Verified — predictions match to
   0.0 with and without it.

   This is also the real fix for the training/serving skew. The old bug was a
   scaler being re-fit at inference. The strongest fix is not "remember to
   save the scaler", it is "have no fitted state in feature engineering at
   all". Stateless transforms cannot skew.

   If you later add a linear baseline (logistic regression), scaling starts to
   matter — put it inside an sklearn Pipeline with the estimator so the two
   can never be saved separately.
"""

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400

# Set True to encode hour cyclically instead of as a raw 0-24 value.
# Default is False: trees split on thresholds and handle non-linearity
# natively, and a raw hour is legible in a SHAP plot and in a case note
# ("transaction at hour 3"). `hour_sin = -0.71` is not something you can put
# in front of a fraud analyst. Turn it on only if you swap in a linear model.
CYCLIC_HOUR = False

RAW_REQUIRED = ["Time", "Amount"]


def feature_names(cyclic: bool = CYCLIC_HOUR):
    v_cols = [f"V{i}" for i in range(1, 29)]
    hour = ["hour_sin", "hour_cos"] if cyclic else ["hour_of_day"]
    return v_cols + ["Amount"] + hour


def engineer(raw: pd.DataFrame, cyclic: bool = CYCLIC_HOUR) -> pd.DataFrame:
    """raw: V1..V28, Amount, Time (Class ignored if present).
    Returns the model-ready frame. Stateless — no fitting, ever."""
    missing = [c for c in RAW_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Input is missing raw columns {missing}")

    X = raw.drop(columns=["Class"], errors="ignore").copy()

    hour = (X["Time"] % SECONDS_PER_DAY) / 3600.0
    if cyclic:
        X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    else:
        X["hour_of_day"] = hour

    X = X.drop(columns=["Time"])

    cols = feature_names(cyclic)
    absent = [c for c in cols if c not in X.columns]
    if absent:
        raise ValueError(f"Engineered frame is missing {absent}")
    return X[cols]


if __name__ == "__main__":
    demo = pd.DataFrame({
        **{f"V{i}": [0.1, -0.2] for i in range(1, 29)},
        "Amount": [149.62, 2125.87],
        "Time": [0.0, 96_400.0],   # second one is 26.8h in -> hour 2.8 of day 2
    })
    print(engineer(demo)[["Amount", "hour_of_day"]])
