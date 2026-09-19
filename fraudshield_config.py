"""
FraudShield — central configuration.

Everything that can drift between training, evaluation and serving lives here
and ONLY here. If a number appears in two places, it will eventually disagree
with itself (see: threshold 0.6 in the tuning cell vs 0.7 in the save cell vs
0.8 in the compliance agent — three different numbers for one decision).

Bump ARTIFACT_VERSION on every retrain. Bump FEATURE_VERSION whenever
features.py changes. Bump the prompt version strings whenever you edit a
prompt. All three get stamped into every eval report, so you can always answer
"which model, which features, which prompt produced this result?"
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
# Resolved relative to this file, not the cwd, so notebooks and scripts agree
# and nothing breaks when the repo moves off C:\Users\Abcom\Desktop.
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "creditcard.csv"
ARTIFACT_DIR = ROOT / "artifacts"
EVAL_DIR = ROOT / "Fraudshield_evals"
FAISS_DIR = ROOT / "faiss_index"

ARTIFACT_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- versioning
ARTIFACT_VERSION = "v2.0.0"     # v2 = hour_of_day features, no scaler
FEATURE_VERSION = "features.v2"
BUNDLE_PATH = ARTIFACT_DIR / f"fraudshield_bundle_{ARTIFACT_VERSION}.pkl"
BUNDLE_LATEST = ARTIFACT_DIR / "fraudshield_bundle_latest.pkl"
BASELINE_STATS_PATH = ARTIFACT_DIR / f"baseline_stats_{ARTIFACT_VERSION}.json"

# ---------------------------------------------------------------- modelling
RANDOM_STATE = 42
TEST_SIZE = 0.2

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "eval_metric": "aucpr",
}

# The one threshold. Chosen for recall: missing fraud costs far more than a
# false positive an analyst clears in 30 seconds.
DECISION_THRESHOLD = 0.70

# Minimum bar a retrained model must clear to be promoted.
PROMOTION_GATE = {
    "min_recall": 0.85,
    "min_precision": 0.40,
    "min_pr_auc": 0.80,
}

# ---------------------------------------------------------------- agents
# Amounts are in euros and top out near 25,691, so the old 100_000
# document-verification rule could never fire — that branch was dead across
# all 36 golden cases. These are set where they can actually separate cases.
AGENT_RULES = {
    "doc_verification_amount_ceiling": 200.0,
    "human_review_risk_floor": 0.70,    # == DECISION_THRESHOLD, deliberately
    "auto_reject_risk_floor": 0.90,     # was a bare 0.8 inside the agent
}

# ---------------------------------------------------------------- prompts
# Prompt registry. Treat these like model weights: versioned, diffable,
# never edited in place inside a notebook cell.
PROMPTS = {
    "case_notes": {
        "version": "case_notes.v1",
        "template": (
            "You are a fraud analyst writing a case note for a reviewer who has "
            "15 seconds to read it.\n\n"
            "Transaction ID: {transaction_id}\n"
            "Amount: EUR {amount:.2f}\n"
            "Hour of day: {hour_of_day:.1f} (relative to dataset start)\n"
            "Model fraud probability: {risk_score:.3f} "
            "(decision threshold {threshold})\n"
            "Document verification: {document_verified}\n"
            "Compliance: {compliance_passed}\n"
            "Decision: {decision}\n"
            "Top model drivers: {shap_summary}\n\n"
            "Write 1-2 sentences. State the decision and the single strongest "
            "reason for it. Do not speculate beyond the evidence above. Do not "
            "invent merchant names, locations or cardholder details. Do not "
            "state an absolute clock time -- the hour above is relative."
        ),
    },
    "llm_judge": {
        "version": "llm_judge.v1",
        "template": (
            "You are grading a fraud case note against a reference note.\n\n"
            "REFERENCE (gold): {expected}\n"
            "CANDIDATE (model): {actual}\n\n"
            "Score the candidate 1-5 on each dimension:\n"
            "- factual_consistency: does it contradict the reference or invent facts?\n"
            "- decision_match: does it convey the same decision?\n"
            "- usefulness: would a reviewer know what to do next?\n\n"
            "Respond with ONLY a JSON object, no markdown fences, no preamble:\n"
            '{{"factual_consistency": <int>, "decision_match": <int>, '
            '"usefulness": <int>, "reason": "<one short sentence>"}}'
        ),
    },
}

# ---------------------------------------------------------------- LLMOps
LLM_MODEL = "llama-3.3-70b-versatile"   # Groq
LLM_TEMPERATURE = 0.0                   # deterministic for reproducible evals
LANGSMITH_PROJECT = "fraudshield"
