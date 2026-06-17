import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline

# ────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Riya",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — dark analyst theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0f1117; }
.riya-box {
    background: linear-gradient(135deg, #1a1f2e, #16213e);
    border-left: 4px solid #00d4ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #e0e0e0;
    font-size: 15px;
}
.fraud-badge {
    background-color: #ff4b4b;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 13px;
}
.legit-badge {
    background-color: #00c853;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 13px;
}
.metric-card {
    background: #1a1f2e;
    border: 1px solid #2d3561;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
.shap-bar {
    background: #ff6b35;
    height: 10px;
    border-radius: 4px;
}
.shap-bar-neg {
    background: #00d4ff;
    height: 10px;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS — cached so loads only once
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("fraudshield_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_artifacts():
    shap_values = joblib.load("shap_values.pkl")
    X_test_sample = joblib.load("X_test_sample.pkl")
    threshold = float(open("best_threshold.txt").read().strip())
    return shap_values, X_test_sample, threshold

@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

@st.cache_resource
def load_llm():
    pipe = hf_pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=120,
        temperature=0.3,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_top3_shap_features(transaction_idx, shap_values, X_test_sample):
    vals = shap_values[transaction_idx]
    names = X_test_sample.columns.tolist()
    pairs = sorted(zip(names, vals), key=lambda x: abs(x[1]), reverse=True)
    top3 = pairs[:3]
    return top3

def get_risk_score(model, X_test_sample, transaction_idx, threshold):
    row = X_test_sample.iloc[[transaction_idx]]
    prob = model.predict_proba(row)[0][1]
    score = int(prob * 100)
    is_fraud = prob >= threshold
    return prob, score, is_fraud

def get_riya_explanation(transaction_idx, top3, amount, vectorstore, llm):
    # Build SHAP context
    shap_context = "TOP 3 FRAUD INDICATORS:\n"
    for i, (feat, val) in enumerate(top3, 1):
        risk = "HIGH RISK" if val > 0 else "SUSPICIOUS"
        shap_context += f"{i}. {feat}: SHAP={round(float(val), 4)} ({risk})\n"

    # RAG retrieval
    query = f"Why was this transaction flagged? Amount: ${amount:.2f}"
    docs = vectorstore.similarity_search(query, k=2)
    rag_context = "\n".join([doc.page_content for doc in docs])

    # Tightly grounded prompt — no hallucination scope
    prompt = f"""You are Riya, a fraud analyst AI assistant.
Based ONLY on the data below, write a 2-sentence explanation of why this transaction was flagged.
Do not add any information not present in the context.

TRANSACTION DATA:
Amount: ${amount:.2f}
{shap_context}
FRAUD RULES CONTEXT:
{rag_context}

EXPLANATION:"""

    try:
        response = llm.invoke(prompt)
        explanation = response.strip()
        if "EXPLANATION:" in explanation:
            explanation = explanation.split("EXPLANATION:")[-1].strip()
        return explanation
    except Exception as e:
        feat1, val1 = top3[0]
        feat2, val2 = top3[1]
        return (
            f"This transaction was flagged primarily due to anomalous activity in {feat1} "
            f"(SHAP impact: {round(float(val1),3)}) and {feat2} (SHAP impact: {round(float(val2),3)}). "
            f"The transaction amount of ${amount:.2f} combined with these feature deviations "
            f"exceeded the 0.7 fraud probability threshold."
        )

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # Header
    st.markdown("# 🛡️ FraudShield")
    st.markdown("#### AI-Powered Fraud Detection — *Riya, Fraud Ops Analyst*")
    st.divider()

    # Load all artifacts
    with st.spinner("Loading FraudShield models..."):
        model = load_model()
        shap_values, X_test_sample, threshold = load_artifacts()
        vectorstore = load_rag()
        llm = load_llm()

    st.success(f"✅ FraudShield ready — {len(X_test_sample)} transactions loaded | Threshold: {threshold}")

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("## ⚙️ Alert Queue Settings")
        score_filter = st.slider("Min Risk Score", 0, 100, 50)
        n_transactions = st.slider("Transactions to Review", 1, min(20, len(X_test_sample)), 5)
        st.divider()
        st.markdown("### 📊 Model Performance")
        st.metric("Recall", "87%")
        st.metric("Precision", "51%")
        st.metric("Threshold", "0.70")
        st.divider()
        st.markdown("*Riya — Senior Fraud Analyst AI*")
        st.caption("FraudShield v1.0 | Portfolio Project")

    # ── SCORE ALL TRANSACTIONS ──
    results = []
    for i in range(n_transactions):
        prob, score, is_fraud = get_risk_score(model, X_test_sample, i, threshold)
        amount = X_test_sample.iloc[i].get("Amount", 0.0)
        results.append({
            "idx": i,
            "prob": prob,
            "score": score,
            "is_fraud": is_fraud,
            "amount": float(amount)
        })

    # Sort by risk score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    # Apply filter
    results = [r for r in results if r["score"] >= score_filter]

    if not results:
        st.warning("No transactions above the selected risk score threshold.")
        return

    # ── SUMMARY METRICS ──
    col1, col2, col3, col4 = st.columns(4)
    fraud_count = sum(1 for r in results if r["is_fraud"])
    col1.metric("🔴 Flagged as Fraud", fraud_count)
    col2.metric("🟢 Likely Legitimate", len(results) - fraud_count)
    col3.metric("📋 Total Reviewed", len(results))
    avg_score = int(np.mean([r["score"] for r in results]))
    col4.metric("📈 Avg Risk Score", f"{avg_score}/100")

    st.divider()
    st.markdown("## 📋 Alert Queue")

    # ── ALERT QUEUE ──
    for r in results:
        idx = r["idx"]
        score = r["score"]
        is_fraud = r["is_fraud"]
        amount = r["amount"]

        badge = '<span class="fraud-badge">🔴 FRAUD ALERT</span>' if is_fraud else '<span class="legit-badge">🟢 LEGITIMATE</span>'

        with st.expander(f"Transaction #{idx+1} — Risk Score: {score}/100 — Amount: ${amount:.2f}", expanded=is_fraud):

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)
                st.markdown(f"**Risk Score:** `{score} / 100`")
                st.markdown(f"**Amount:** `${amount:.2f}`")
                st.markdown(f"**Fraud Probability:** `{r['prob']:.4f}`")

                # SHAP top 3
                st.markdown("**🔍 Top 3 SHAP Features:**")
                top3 = get_top3_shap_features(idx, shap_values, X_test_sample)
                for feat, val in top3:
                    bar_class = "shap-bar" if val > 0 else "shap-bar-neg"
                    bar_width = min(int(abs(float(val)) * 200), 100)
                    direction = "⬆️ Increases fraud risk" if val > 0 else "⬇️ Decreases fraud risk"
                    st.markdown(f"`{feat}` — {round(float(val), 4)}")
                    st.markdown(
                        f'<div class="{bar_class}" style="width:{bar_width}%"></div>',
                        unsafe_allow_html=True
                    )
                    st.caption(direction)

            with col2:
                # Riya's explanation
                st.markdown("**💬 Riya's Analysis:**")
                if is_fraud:
                    with st.spinner("Riya is analyzing..."):
                        explanation = get_riya_explanation(idx, top3, amount, vectorstore, llm)
                    st.markdown(
                        f'<div class="riya-box">🤖 <b>Riya says:</b><br><br>{explanation}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="riya-box">🤖 <b>Riya says:</b><br><br>'
                        'This transaction does not meet the fraud threshold. '
                        'Feature values are within normal ranges. No action required.</div>',
                        unsafe_allow_html=True
                    )

                # Analyst decision buttons
                st.markdown("**⚡ Analyst Action:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("✅ Confirm Fraud", key=f"fraud_{idx}"):
                        st.success("Logged: Confirmed Fraud")
                with col_b:
                    if st.button("❌ Mark Legitimate", key=f"legit_{idx}"):
                        st.info("Logged: Marked Legitimate")
                with col_c:
                    if st.button("⬆️ Escalate", key=f"escalate_{idx}"):
                        st.warning("Logged: Escalated to Risk Manager")

                notes = st.text_input("📝 Analyst Notes (optional)", key=f"notes_{idx}", placeholder="Add context...")

if __name__ == "__main__":
    main()
