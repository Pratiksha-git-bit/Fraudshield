"""
FraudShield — evaluation harness (the LLMOps half).

Your notebook produces golden_dataset_results.csv with actual_* columns, but
nothing ever scores them. This closes that loop.

Two kinds of assertion, and the distinction matters in an interview:

  DETERMINISTIC  — decision, escalation, compliance flag. Exact match. These
                   are the ones that caught the Risk Scoring agent returning
                   mock values, and they should run on every commit.

  GRADED         — the case note narrative. No exact match exists, so an LLM
                   judge scores it against your hand-written gold summary.
                   Judges are noisy: temperature 0, a fixed prompt version,
                   and spot-check ~10 judgments by hand before you trust it.

Run:
    python eval_runner.py                       # deterministic only, free, fast
    python eval_runner.py --judge               # adds LLM-as-judge (uses Groq)
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd

import fraudshield_config as cfg

RESULTS_PATH = cfg.EVAL_DIR / "golden_dataset_results.csv"

# expected_column -> actual_column, plus a normaliser so "Approve" == "approve"
DETERMINISTIC_CHECKS = {
    "expected_escalation": "actual_decision",
    "expected_compliance_flag": "actual_compliance_passed",
    "expected_document_verification": "actual_document_verified",
}

NORMALISE = {
    "auto_approve": "approve",
    "auto_reject": "reject",
    "human_review": "escalate_to_human",
    "pass": "true",
    "block": "false",
    "valid": "true",
    "invalid": "false",
    "yes": "true",
    "no": "false",
}


def norm(v):
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    if s in ("nan", "none", "<na>"):
        return ""
    return NORMALISE.get(s, s)


def is_labelled(series: pd.Series) -> pd.Series:
    """Empty cells in a CSV read back as NaN, not "". Getting this wrong marks
    every unlabelled row as a failure and makes the whole report meaningless."""
    return series.map(norm) != ""


# ------------------------------------------------------------ deterministic
def run_deterministic(df: pd.DataFrame):
    rows, summary = [], {}
    for exp_col, act_col in DETERMINISTIC_CHECKS.items():
        if exp_col not in df.columns or act_col not in df.columns:
            print(f"  skip {exp_col}: column not present")
            continue
        labelled = df[is_labelled(df[exp_col])]
        if labelled.empty:
            print(f"  skip {exp_col}: no hand labels filled in yet")
            continue
        match = labelled[exp_col].map(norm) == labelled[act_col].map(norm)
        summary[exp_col] = {
            "n_labelled": int(len(labelled)),
            "n_pass": int(match.sum()),
            "accuracy": round(float(match.mean()), 3),
        }
        for idx, ok in match.items():
            if not ok:
                rows.append({
                    "case_id": df.at[idx, "case_id"],
                    "check": exp_col,
                    "risk_band": df.at[idx, "risk_band"],
                    "risk": round(float(df.at[idx, "model_fraud_probability"]), 3),
                    "expected": df.at[idx, exp_col],
                    "actual": df.at[idx, act_col],
                })
    return summary, pd.DataFrame(rows)


def coverage_check(df: pd.DataFrame):
    """A check the golden set alone cannot give you: did every branch of every
    agent actually execute? A rule that never fires is untested, not passing.
    This is how you find dead rules like `amount > 100000` on a dataset whose
    maximum amount is about 25,691."""
    notes = []
    if "actual_document_verified" in df.columns:
        vals = df["actual_document_verified"].map(norm).unique()
        if len(vals) == 1:
            notes.append(
                f"Document verification returned '{vals[0]}' for all "
                f"{len(df)} cases — the other branch is never exercised."
            )
    if "actual_decision" in df.columns:
        seen = set(df["actual_decision"].map(norm))
        for d in ("approve", "reject", "escalate_to_human"):
            if d not in seen:
                notes.append(f"Decision '{d}' never produced across the golden set.")
    return notes


# ------------------------------------------------------------ LLM judge
def call_llm(prompt: str) -> str:
    import requests
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set — put it in your .env")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": cfg.LLM_MODEL,
            "temperature": cfg.LLM_TEMPERATURE,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_json(text: str):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        return json.loads(m.group(0)) if m else None


def run_judge(df: pd.DataFrame):
    spec = cfg.PROMPTS["llm_judge"]
    col = df.get("expected_case_note_summary", pd.Series(dtype=str))
    labelled = df[is_labelled(col)] if len(col) else df.iloc[0:0]
    if labelled.empty:
        print("  no gold case-note summaries filled in — judge skipped")
        return pd.DataFrame()

    out = []
    for idx, row in labelled.iterrows():
        prompt = spec["template"].format(
            expected=row["expected_case_note_summary"],
            actual=str(row.get("actual_case_notes", ""))[:1500],
        )
        try:
            scores = parse_json(call_llm(prompt)) or {}
        except Exception as e:  # noqa: BLE001
            scores = {"error": str(e)[:120]}
        out.append({
            "case_id": row["case_id"],
            "judge_prompt_version": spec["version"],
            **scores,
        })
        print(f"  judged {row['case_id']}")
    return pd.DataFrame(out)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="run the LLM judge")
    ap.add_argument("--path", default=str(RESULTS_PATH))
    args = ap.parse_args()

    # LangSmith tracing: set these in .env and every LangGraph invoke is
    # captured automatically — per-node latency, inputs, outputs, token cost.
    # No code change needed inside the graph itself.
    #   LANGCHAIN_TRACING_V2=true
    #   LANGCHAIN_API_KEY=ls__...
    #   LANGCHAIN_PROJECT=fraudshield
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print(f"LangSmith tracing active -> project "
              f"{os.getenv('LANGCHAIN_PROJECT', cfg.LANGSMITH_PROJECT)}")

    df = pd.read_csv(args.path)
    print(f"Loaded {len(df)} golden cases from {args.path}\n")

    print("=== DETERMINISTIC CHECKS ===")
    summary, failures = run_deterministic(df)
    if not summary:
        print("  nothing to check -- fill in the expected_* columns in "
              "golden_dataset_template.csv first, then re-run the graph.")
    for check, s in summary.items():
        print(f"{check:34s} {s['n_pass']}/{s['n_labelled']}  acc={s['accuracy']:.1%}")

    if not failures.empty:
        print("\n--- FAILURES ---")
        print(failures.to_string(index=False))
        print("\nFailures by risk band:")
        print(failures["risk_band"].value_counts().to_string())
    else:
        print("\nNo deterministic failures.")

    print("\n=== COVERAGE ===")
    notes = coverage_check(df)
    print("\n".join(f"  - {n}" for n in notes) if notes else "  all branches exercised")

    judge_df = pd.DataFrame()
    if args.judge:
        print("\n=== LLM JUDGE (case notes) ===")
        judge_df = run_judge(df)
        if not judge_df.empty:
            for col in ("factual_consistency", "decision_match", "usefulness"):
                if col in judge_df.columns:
                    print(f"{col:22s} mean={judge_df[col].mean():.2f}/5")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "run_at": stamp,
        "model_version": cfg.ARTIFACT_VERSION,
        "case_notes_prompt": cfg.PROMPTS["case_notes"]["version"],
        "judge_prompt": cfg.PROMPTS["llm_judge"]["version"],
        "deterministic": summary,
        "coverage_notes": notes,
        "n_failures": int(len(failures)),
    }
    path = cfg.EVAL_DIR / f"eval_report_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    if not failures.empty:
        failures.to_csv(cfg.EVAL_DIR / f"eval_failures_{stamp}.csv", index=False)
    if not judge_df.empty:
        judge_df.to_csv(cfg.EVAL_DIR / f"eval_judge_{stamp}.csv", index=False)
    print(f"\nSaved -> {path.name}")


if __name__ == "__main__":
    main()
