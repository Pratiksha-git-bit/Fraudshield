"""
FraudShield — SHAP explainability.

Replaces EDA cells 13-14. Regenerates SHAP against the v2 bundle and saves the
artifacts the RAG layer expects.

Why this is a file and not a notebook cell: the old cells computed SHAP from
`xgb_model` and `X_test`, variables that only existed mid-notebook. The saved
pickles carried no record of which model or which feature set produced them,
so a stale shap_values.pkl would happily load against a new model and
attribute importance to the wrong columns -- silently, with plausible-looking
numbers. This version stamps the versions in and asserts they match on load.

Run:
    pip install shap
    py -3.14 explain.py
"""

import joblib
import numpy as np
import pandas as pd

import fraudshield_config as cfg
from features import engineer
from serving import load_bundle, score

SAMPLE_SIZE = 500          # SHAP is slow; 500 rows is plenty for a summary plot
ARTIFACT = cfg.ARTIFACT_DIR / f"shap_{cfg.ARTIFACT_VERSION}.pkl"
ARTIFACT_LATEST = cfg.ARTIFACT_DIR / "shap_latest.pkl"


def build(sample_size: int = SAMPLE_SIZE, plot: bool = False):
    import shap

    bundle = load_bundle()
    df = pd.read_csv(cfg.DATA_PATH)

    # Explain held-out rows only. SHAP on training rows describes what the
    # model memorised, not how it generalises.
    import json
    idx_path = cfg.ARTIFACT_DIR / f"test_index_{cfg.ARTIFACT_VERSION}.json"
    if idx_path.exists():
        df = df.loc[json.loads(idx_path.read_text())]
        source = "held-out test rows"
    else:
        source = "full dataset (no test index found -- run train_pipeline.py)"

    raw_sample = df.sample(min(sample_size, len(df)), random_state=cfg.RANDOM_STATE)
    X_sample = engineer(raw_sample, cyclic=bundle["cyclic_hour"])

    if list(X_sample.columns) != bundle["feature_names"]:
        raise ValueError("Feature mismatch between engineer() and the bundle.")

    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(X_sample)

    scored = score(raw_sample, bundle)

    payload = {
        "model_version": bundle["version"],
        "feature_version": bundle["feature_version"],
        "feature_names": bundle["feature_names"],
        "shap_values": shap_values,
        "X_sample": X_sample,
        "amounts": raw_sample["Amount"].to_numpy(),
        "hours": scored["hour_of_day"].to_numpy(),
        "probabilities": scored["fraud_probability"].to_numpy(),
        "expected_value": float(explainer.expected_value),
        "source": source,
    }
    joblib.dump(payload, ARTIFACT)
    joblib.dump(payload, ARTIFACT_LATEST)

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(bundle["feature_names"], mean_abs), key=lambda x: -x[1])

    print(f"Model {bundle['version']} | {len(X_sample)} rows from {source}")
    print(f"Saved -> {ARTIFACT.name}\n")
    print("Top 10 features by mean |SHAP|:")
    for name, val in ranking[:10]:
        print(f"  {name:14s} {val:.4f}")

    if plot:
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(cfg.ARTIFACT_DIR / f"shap_bar_{cfg.ARTIFACT_VERSION}.png", dpi=140)
        plt.close()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(cfg.ARTIFACT_DIR / f"shap_beeswarm_{cfg.ARTIFACT_VERSION}.png", dpi=140)
        plt.close()
        print("\nSaved shap_bar and shap_beeswarm PNGs to artifacts/")

    return payload


def load():
    """Load SHAP artifacts, refusing stale ones. This assertion is the whole
    point of the file: a mismatched pickle fails loudly instead of quietly
    labelling the wrong columns."""
    if not ARTIFACT_LATEST.exists():
        raise FileNotFoundError("No SHAP artifact. Run: py -3.14 explain.py")
    payload = joblib.load(ARTIFACT_LATEST)
    bundle = load_bundle()
    if payload["model_version"] != bundle["version"]:
        raise ValueError(
            f"Stale SHAP artifact: built for model {payload['model_version']}, "
            f"current model is {bundle['version']}. Re-run explain.py."
        )
    # Version strings alone are not enough. If features.py changes and the
    # model is retrained without bumping ARTIFACT_VERSION, the check above
    # passes while the SHAP values describe a different column set -- which is
    # the original Amount_scaled bug wearing a new hat. Compare the columns
    # themselves, not just the label on the tin.
    if payload["feature_version"] != bundle["feature_version"]:
        raise ValueError(
            f"Stale SHAP artifact: built with {payload['feature_version']}, "
            f"bundle uses {bundle['feature_version']}. Re-run explain.py."
        )
    if list(payload["feature_names"]) != list(bundle["feature_names"]):
        raise ValueError(
            "SHAP artifact feature names do not match the bundle.\n"
            f" artifact: {list(payload['feature_names'])}\n"
            f" bundle  : {list(bundle['feature_names'])}\n"
            "Re-run explain.py."
        )
    return payload


def top_features(i: int, k: int = 3, payload=None) -> str:
    """Replaces get_top3_shap_features(). Returns RAG-ready context.

    Note the framing: V1-V28 are PCA outputs, so naming them as risk factors
    would be meaningless to an analyst. They are described as relative
    contribution only.
    """
    p = payload or load()
    vals = p["shap_values"][i]
    pairs = sorted(zip(p["feature_names"], vals), key=lambda x: abs(x[1]), reverse=True)[:k]

    lines = [f"TOP {k} MODEL DRIVERS (SHAP contribution to this decision):"]
    for n, (feat, val) in enumerate(pairs, 1):
        direction = "pushes toward fraud" if val > 0 else "pushes toward legitimate"
        lines.append(f"{n}. {feat}: {val:+.4f} ({direction})")
    lines.append(f"\nAmount: EUR {p['amounts'][i]:.2f}")
    lines.append(f"Hour of day: {p['hours'][i]:.1f} (relative to dataset start)")
    lines.append(f"Model fraud probability: {p['probabilities'][i]:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    build(plot="--plot" in sys.argv)
    print("\n--- Sample context for transaction 0 ---")
    print(top_features(0))
