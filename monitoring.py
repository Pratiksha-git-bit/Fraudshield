"""
FraudShield — drift monitoring (v2).

Answers the question a hiring manager will actually ask: "how would you know
your model had gone stale in production?"

Three signals, in the order they become available in real life:

1. Prediction drift  — the score distribution shifts. Visible immediately.
2. Feature drift     — inputs shift. Visible immediately.
3. Performance decay — recall drops. Only visible once labels arrive, which
                       for fraud means 30-90 days later (chargeback lag).

Because of (3), 1 and 2 are your early-warning system, not a nice-to-have.

--------------------------------------------------------------------------
FIXED IN v2
--------------------------------------------------------------------------

1. THE NO-DRIFT CONTROL WAS SAMPLING TRAINING ROWS.
   `full.sample(frac=0.2)` drew from all 284,807 rows, so roughly 80% of the
   "live" batch was data the model memorised. That is why the control run
   reported live recall 0.980 against train 0.888 — the model appearing to
   IMPROVE after deployment, which never happens. A decay monitor with a
   contaminated baseline cannot detect decay; it reads healthy right up until
   it doesn't. The control now samples from the held-out test index only.

2. --simulate PRINTED PERFORMANCE METRICS THAT MEANT NOTHING.
   Simulation mutates V14, V17 and Amount, then scored those mutated rows
   against the ORIGINAL `Class` labels. Those labels describe transactions
   that no longer exist. Recall and precision computed across that join are
   not a measurement of anything. Performance decay is now suppressed under
   --simulate, with an explicit note saying why. Drift still reports, because
   drift is exactly what the simulation is designed to exercise.

3. RETRAIN FIRED ON DRIFT ALONE, FROM A SINGLE BATCH.
   One reading of prediction PSI above 0.25 produced "RETRAIN — open a
   retraining ticket". Three things wrong with that:
     - Drift is evidence of change, not evidence of degradation. The model may
       be fine on the new population.
     - The most common cause of a sudden PSI spike is a broken upstream
       pipeline, not a real population shift. Retraining on corrupted input
       bakes the corruption into the model.
     - One batch is a sample. Drift has to persist to be real.
   Retraining is now recommended only when labelled performance has actually
   fallen below the promotion gate, or when severe drift persists across
   consecutive runs. Everything else routes to INVESTIGATE, pipeline first.

4. PSI ASSUMED EVERY BASELINE BIN HOLDS 1/n OF THE MASS.
   True only when all 11 decile edges are distinct. `np.unique` silently
   collapses ties, and the code then assumed uniform mass across the survivors
   — so a feature with a concentrated point mass got its expected percentages
   understated and its PSI inflated, biasing toward false alarms. Expected
   mass is now derived from how many decile intervals collapsed into each bin.
   This is defensive: it may not be firing on the current features, and the
   run will tell you if it is.

Run:
    python monitoring.py                 # no-drift control on held-out rows
    python monitoring.py new_batch.csv   # score a real incoming batch
    python monitoring.py --simulate      # prove the alarm fires
"""

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import fraudshield_config as cfg
from features import engineer
from serving import load_bundle, score

# Industry-standard PSI bands.
PSI_STABLE = 0.10       # below this: no action
PSI_INVESTIGATE = 0.25  # 0.10-0.25: investigate. above 0.25: retrain candidate.

# A PSI computed on a few hundred rows is mostly sampling noise.
MIN_BATCH_ROWS = 1_000

# Severe drift must persist across this many consecutive runs before it is
# treated as real rather than as a bad batch.
RUNS_BEFORE_RETRAIN = 3
DRIFT_HISTORY_PATH = cfg.ARTIFACT_DIR / "drift_history.json"


# ------------------------------------------------------------------ PSI
def psi(baseline_deciles, actual_values, eps=1e-6):
    """Population Stability Index against fixed baseline decile edges.

    Expected mass per bin is derived from the decile array rather than assumed
    uniform. With 11 edges there are 10 intervals, each holding 10% of the
    baseline by construction; if edges tie, the intervals that collapse into a
    surviving bin have their mass summed into it. Assuming 1/n across the
    survivors instead understates the expected share of a concentrated bin and
    inflates PSI.
    """
    raw = np.asarray(baseline_deciles, dtype=float)
    if len(raw) < 3:
        return 0.0

    mass_per_interval = 1.0 / (len(raw) - 1)
    edges, inverse = np.unique(raw, return_inverse=True)
    if len(edges) < 3:
        return 0.0

    n_bins = len(edges) - 1
    exp_pct = np.zeros(n_bins)
    for i in range(len(raw) - 1):
        exp_pct[min(int(inverse[i]), n_bins - 1)] += mass_per_interval

    bin_edges = edges.astype(float).copy()
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    act_counts, _ = np.histogram(np.asarray(actual_values, dtype=float), bins=bin_edges)
    act_pct = act_counts / max(act_counts.sum(), 1)

    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def verdict(value):
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_INVESTIGATE:
        return "investigate"
    return "retrain candidate"     # candidate, not instruction


# ------------------------------------------------------------------ batches
def held_out_control(frac=0.2, random_state=7):
    """The no-drift control. Must come from rows the model never saw, or the
    comparison measures memorisation instead of generalisation."""
    full = pd.read_csv(cfg.DATA_PATH)
    idx_path = cfg.ARTIFACT_DIR / f"test_index_{cfg.ARTIFACT_VERSION}.json"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"{idx_path.name} not found. Run train_pipeline.py first — the "
            "control batch must be drawn from held-out rows, not from the "
            "whole file."
        )
    test_idx = json.loads(idx_path.read_text())
    held = full.loc[test_idx]
    batch = held.sample(frac=frac, random_state=random_state)
    print(f"No batch given -- no-drift control: {len(batch):,} rows sampled "
          f"from {len(held):,} held-out rows (never trained on).\n")
    return batch


def inject_drift(batch):
    """Shifts the two features the model leans on hardest plus the amount
    distribution — what a merchant-mix change or an upstream pipeline bug
    looks like in practice."""
    batch = batch.copy()
    batch["V14"] = batch["V14"] - 1.5
    batch["V17"] = batch["V17"] * 1.8
    batch["Amount"] = batch["Amount"] * 2.5
    print("--simulate: injected synthetic drift into V14, V17 and Amount.\n")
    return batch


# ------------------------------------------------------------------ drift
def run_drift_report(new_df: pd.DataFrame, top_n=10):
    baseline = json.loads(cfg.BASELINE_STATS_PATH.read_text())
    bundle = load_bundle()

    raw = new_df.drop(columns=["Class"], errors="ignore")
    scored = score(raw, bundle)

    pred_psi = psi(baseline["score_deciles"], scored["fraud_probability"])
    flag_rate = float(scored["flagged"].mean())

    engineered = engineer(raw, cyclic=bundle["cyclic_hour"])

    rows, collapsed = [], []
    for feat, stats in baseline["features"].items():
        if feat not in engineered.columns:
            rows.append({"feature": feat, "psi": np.nan, "status": "MISSING"})
            continue
        if len(np.unique(stats["deciles"])) < len(stats["deciles"]):
            collapsed.append(feat)
        value = psi(stats["deciles"], engineered[feat])
        rows.append({
            "feature": feat,
            "psi": round(value, 4),
            "baseline_mean": round(stats["mean"], 4),
            "current_mean": round(float(engineered[feat].mean()), 4),
            "status": verdict(value),
        })

    feat_df = pd.DataFrame(rows).sort_values("psi", ascending=False)

    print(f"Model version: {bundle['version']}  |  batch rows: {len(raw):,}")
    print("\n=== PREDICTION DRIFT ===")
    print(f"Score PSI            : {pred_psi:.4f}  -> {verdict(pred_psi).upper()}")
    print(f"Flag rate            : {flag_rate:.4%}")
    print(f"Training fraud rate  : {baseline['fraud_rate']:.4%}")

    print(f"\n=== FEATURE DRIFT (top {top_n} by PSI) ===")
    print(feat_df.head(top_n).to_string(index=False))

    unstable = feat_df[feat_df["psi"] >= PSI_STABLE]
    severe = feat_df[feat_df["psi"] >= PSI_INVESTIGATE]
    print(f"\nFeatures above PSI {PSI_STABLE}: {len(unstable)} of {len(feat_df)}")

    if collapsed:
        print(f"Note: tied baseline deciles in {collapsed} — expected mass was "
              f"derived from interval multiplicity, not assumed uniform.")

    out = cfg.ARTIFACT_DIR / "drift_report.csv"
    feat_df.to_csv(out, index=False)
    print(f"Saved -> {out.name}")

    return {
        "prediction_psi": float(pred_psi),
        "flag_rate": flag_rate,
        "n_unstable": int(len(unstable)),
        "n_severe": int(len(severe)),
        "n_rows": int(len(raw)),
    }


# ------------------------------------------------------------------ decay
def performance_decay(new_df: pd.DataFrame):
    """Only callable once labels land. Compares live recall/precision to the
    metrics stored in the bundle at training time."""
    if "Class" not in new_df.columns:
        print("\nNo labels in this batch — performance decay not computable yet. "
              "For fraud this is the normal state: chargebacks arrive 30-90 days "
              "later, which is why drift is the early-warning signal.")
        return None

    from sklearn.metrics import precision_score, recall_score

    bundle = load_bundle()
    scored = score(new_df.drop(columns=["Class"]), bundle)
    pred = scored["flagged"].astype(int)
    y = new_df["Class"]

    live = {
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "n_fraud": int(y.sum()),
    }
    train = bundle["metrics"]
    print("\n=== PERFORMANCE DECAY ===")
    for k in ("recall", "precision"):
        print(f"{k:10s} train={train[k]:.3f}  live={live[k]:.3f}  "
              f"delta={live[k] - train[k]:+.3f}")
    print(f"(live metrics computed on {live['n_fraud']} actual fraud cases)")

    if live["n_fraud"] < 30:
        print("Caution: too few fraud cases in this batch for the live figures "
              "to be stable. Treat as indicative, not as a trigger.")

    live["below_gate"] = bool(live["recall"] < cfg.PROMOTION_GATE["min_recall"]
                              and live["n_fraud"] >= 30)
    return live


# ------------------------------------------------------------------ action
def load_history():
    if DRIFT_HISTORY_PATH.exists():
        try:
            return json.loads(DRIFT_HISTORY_PATH.read_text())
        except json.JSONDecodeError:
            return []
    return []


def save_history(history, entry, keep=20):
    history = (history + [entry])[-keep:]
    DRIFT_HISTORY_PATH.write_text(json.dumps(history, indent=2))
    return history


def decide_action(drift, decay, history):
    """Drift alone never orders a retrain.

    Retraining is expensive, and it is the wrong first move when the cause is
    an upstream data problem — you would be fitting the model to corrupted
    input. So: confirmed degradation retrains, sustained severe drift retrains,
    and a single drifty batch opens an investigation.
    """
    if drift["n_rows"] < MIN_BATCH_ROWS:
        return ("INSUFFICIENT DATA",
                f"{drift['n_rows']:,} rows is below the {MIN_BATCH_ROWS:,}-row "
                "floor; PSI on a batch this small is mostly sampling noise.")

    if decay and decay.get("below_gate"):
        return ("RETRAIN",
                f"live recall {decay['recall']:.3f} is below the "
                f"{cfg.PROMOTION_GATE['min_recall']} promotion gate on labelled "
                "data — this is measured degradation, not a leading indicator.")

    severe_now = (drift["prediction_psi"] >= PSI_INVESTIGATE
                  or drift["n_severe"] >= 3)

    if severe_now:
        streak = 1
        for past in reversed(history):
            if past.get("severe"):
                streak += 1
            else:
                break
        if streak >= RUNS_BEFORE_RETRAIN:
            return ("RETRAIN",
                    f"severe drift sustained across {streak} consecutive runs — "
                    "no longer attributable to a single bad batch.")
        return ("INVESTIGATE",
                f"severe drift, run {streak} of {RUNS_BEFORE_RETRAIN}. Check the "
                "upstream pipeline FIRST — a schema change or a broken feed is a "
                "more common cause than a real population shift, and retraining "
                "on corrupted input bakes the fault into the model.")

    if drift["prediction_psi"] >= PSI_STABLE or drift["n_unstable"] >= 5:
        return ("INVESTIGATE",
                "moderate drift — worth a look at the upstream pipeline, no "
                "action on the model yet.")

    return ("NONE", "all signals stable.")


# ------------------------------------------------------------------ main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    simulate = "--simulate" in sys.argv

    if args:
        batch = pd.read_csv(args[0])
    else:
        # A time-based split cannot demonstrate drift on this dataset: it spans
        # only ~48 hours, so the tail is one narrow slice of hour_of_day and any
        # "drift" it shows is a sampling artifact, not a real shift.
        batch = held_out_control()

    if simulate:
        batch = inject_drift(batch)

    drift = run_drift_report(batch)

    if simulate:
        # The labels describe the ORIGINAL transactions. After mutating V14,
        # V17 and Amount, those rows are synthetic and the labels no longer
        # correspond to them, so recall and precision across that join measure
        # nothing. Suppressed deliberately.
        decay = None
        print("\n=== PERFORMANCE DECAY ===")
        print("Skipped under --simulate. The injected rows are synthetic while "
              "the Class labels still describe the original transactions, so "
              "live recall/precision computed against them would not be a "
              "measurement of anything.")
    else:
        decay = performance_decay(batch)

    history = load_history()
    action, reason = decide_action(drift, decay, history)

    print(f"\nRecommended action: {action}")
    print(f"  Why: {reason}")

    save_history(history, {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": cfg.ARTIFACT_VERSION,
        "prediction_psi": round(drift["prediction_psi"], 4),
        "n_severe": drift["n_severe"],
        "severe": bool(drift["prediction_psi"] >= PSI_INVESTIGATE
                       or drift["n_severe"] >= 3),
        "simulated": simulate,
        "action": action,
    })
    print(f"  History -> {DRIFT_HISTORY_PATH.name}")


if __name__ == "__main__":
    main()
