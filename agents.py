"""
FraudShield — agent nodes.

Drop-in replacement for cells 1-6 of fraudshield_langgraph.ipynb. Import these
into the notebook instead of redefining them in cells:

    from agents import build_graph
    app = build_graph()

What changed:

- The document-verification ceiling moves from a hardcoded 100_000 to
  AGENT_RULES. At 100_000 the rule could never fire on this dataset (amounts
  top out near 25,691), so that branch was dead across all 36 golden cases and
  the eval could not tell "correctly passes" from "never ran".

- The compliance agent's bare `risk_score > 0.8` now reads
  auto_reject_risk_floor. Three different numbers (0.6, 0.7, 0.8) were
  governing one decision across three files.

- `decision` starts as "pending" rather than "approve". Defaulting to approve
  means a crash mid-graph leaves you with an approval you never made.

- Every node stamps a structured entry into `trace` instead of print(). Prints
  vanish; a trace can be asserted on, and it is what LangSmith will show you
  per node once you turn tracing on.
"""

from typing import TypedDict, Literal, List, Dict, Any

from langgraph.graph import StateGraph, END

import fraudshield_config as cfg

R = cfg.AGENT_RULES


class FraudShieldState(TypedDict):
    transaction_id: str
    amount: float
    hour_of_day: float
    risk_score: float
    model_version: str
    document_verified: bool
    compliance_passed: bool
    case_notes: str
    decision: Literal["pending", "approve", "reject", "escalate_to_human"]
    human_review_needed: bool
    trace: List[Dict[str, Any]]


def new_state(transaction_id, amount, risk_score, hour_of_day=None, model_version="") -> FraudShieldState:
    """Build a valid initial state. Use this instead of hand-writing the dict
    in a notebook cell, so a field is never silently omitted."""
    return {
        "transaction_id": str(transaction_id),
        "amount": float(amount),
        "hour_of_day": float(hour_of_day) if hour_of_day is not None else -1.0,
        "risk_score": float(risk_score),
        "model_version": model_version,
        "document_verified": False,
        "compliance_passed": False,
        "case_notes": "",
        "decision": "pending",
        "human_review_needed": False,
        "trace": [],
    }


def _log(state, node, **fields):
    state["trace"].append({"node": node, **fields})


# ------------------------------------------------------------ agents
def document_verification_agent(state: FraudShieldState) -> FraudShieldState:
    ceiling = R["doc_verification_amount_ceiling"]
    state["document_verified"] = state["amount"] <= ceiling
    state["case_notes"] = (
        "Document verification passed."
        if state["document_verified"]
        else f"Amount EUR {state['amount']:.2f} exceeds the EUR {ceiling:.0f} "
             f"ceiling for automatic document verification."
    )
    _log(state, "document_verification",
         amount=state["amount"], ceiling=ceiling, verified=state["document_verified"])
    return state


def risk_scoring_agent(state: FraudShieldState) -> FraudShieldState:
    # The score comes from the XGBoost bundle via serving.score(); this node
    # applies policy to it, it does not recompute it. An earlier version
    # returned mock values here and the golden dataset caught it.
    floor = R["human_review_risk_floor"]
    state["human_review_needed"] = state["risk_score"] >= floor
    if state["human_review_needed"]:
        state["case_notes"] += f" | Risk {state['risk_score']:.3f} at or above the {floor} review floor."
    _log(state, "risk_scoring",
         risk_score=state["risk_score"], floor=floor,
         review=state["human_review_needed"], model_version=state["model_version"])
    return state


def compliance_agent(state: FraudShieldState) -> FraudShieldState:
    reject_floor = R["auto_reject_risk_floor"]
    if state["risk_score"] >= reject_floor:
        state["compliance_passed"] = False
        state["decision"] = "reject"
        state["case_notes"] += f" | Risk at or above the {reject_floor} auto-reject floor."
    elif not state["document_verified"]:
        # Failed docs escalate rather than auto-reject: a large legitimate
        # transaction is common, and auto-rejecting it is a false positive
        # the customer feels immediately.
        state["compliance_passed"] = False
        state["decision"] = "escalate_to_human"
        state["human_review_needed"] = True
        state["case_notes"] += " | Document verification incomplete - routed to an analyst."
    elif state["human_review_needed"]:
        state["compliance_passed"] = True
        state["decision"] = "escalate_to_human"
        state["case_notes"] += " | Escalated to human analyst."
    else:
        state["compliance_passed"] = True
        state["decision"] = "approve"
        state["case_notes"] += " | Compliance passed - approved."
    _log(state, "compliance",
         passed=state["compliance_passed"], decision=state["decision"])
    return state


def case_notes_agent(state: FraudShieldState) -> FraudShieldState:
    """Deterministic version. Swap the body for an LLM call using
    cfg.PROMPTS['case_notes'] when you move to generated notes -- keep this
    one as the fallback for when the LLM call fails or times out."""
    state["case_notes"] = (
        f"[{state['transaction_id']}] EUR {state['amount']:.2f} at hour "
        f"{state['hour_of_day']:.1f}. Model risk {state['risk_score']:.3f} "
        f"(threshold {cfg.DECISION_THRESHOLD}, model {state['model_version']}). "
        f"Docs verified: {state['document_verified']}. "
        f"Decision: {state['decision'].upper()}. "
        f"Detail: {state['case_notes']}"
    )
    _log(state, "case_notes", chars=len(state["case_notes"]))
    return state


def build_graph():
    wf = StateGraph(FraudShieldState)
    wf.add_node("document_verification", document_verification_agent)
    wf.add_node("risk_scoring", risk_scoring_agent)
    wf.add_node("compliance", compliance_agent)
    wf.add_node("case_notes", case_notes_agent)

    wf.set_entry_point("document_verification")
    wf.add_edge("document_verification", "risk_scoring")
    wf.add_edge("risk_scoring", "compliance")
    wf.add_edge("compliance", "case_notes")
    wf.add_edge("case_notes", END)
    return wf.compile()


# ------------------------------------------------------------ golden run
def run_golden_set(app=None):
    """Replaces notebook cell 12. Adds error handling, provenance, and the
    hour_of_day the model actually used."""
    import pandas as pd
    from serving import load_bundle

    app = app or build_graph()
    bundle = load_bundle()
    path = cfg.EVAL_DIR / "golden_dataset_template.csv"
    golden = pd.read_csv(path)
    golden = golden.drop(columns=[c for c in golden.columns if c.startswith("actual_")])

    rows = []
    for _, row in golden.iterrows():
        state = new_state(
            transaction_id=row["case_id"],
            amount=row["Amount"],
            risk_score=row["model_fraud_probability"],
            hour_of_day=row.get("hour_of_day"),
            model_version=bundle["version"],
        )
        try:
            res = app.invoke(state)
            rows.append({
                "actual_document_verified": res["document_verified"],
                "actual_compliance_passed": res["compliance_passed"],
                "actual_decision": res["decision"],
                "actual_human_review_needed": res["human_review_needed"],
                "actual_case_notes": res["case_notes"],
                "actual_nodes_run": len(res["trace"]),
                "actual_error": "",
            })
        except Exception as e:  # noqa: BLE001
            # A crash must not stop the run -- you want to see all 36 failures
            # at once, not the first one.
            rows.append({
                "actual_document_verified": None, "actual_compliance_passed": None,
                "actual_decision": "ERROR", "actual_human_review_needed": None,
                "actual_case_notes": "", "actual_nodes_run": 0,
                "actual_error": str(e)[:200],
            })

    out = pd.concat([golden, pd.DataFrame(rows, index=golden.index)], axis=1)
    out_path = cfg.EVAL_DIR / "golden_dataset_results.csv"
    out.to_csv(out_path, index=False)

    errors = int((out["actual_decision"] == "ERROR").sum())
    print(f"Ran {len(out)} cases ({errors} errors) -> {out_path.name}")
    print("\nDecision distribution:")
    print(out["actual_decision"].value_counts().to_string())
    print("\nDocument verification distribution:")
    print(out["actual_document_verified"].value_counts(dropna=False).to_string())
    return out


if __name__ == "__main__":
    run_golden_set()
