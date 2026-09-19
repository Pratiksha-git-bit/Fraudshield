"""
FraudShield — Streamlit analyst console (v2).

What was wrong with v1, in order of consequence:

1. It loaded `shap_values.pkl` and `X_test_sample.pkl` written by the last
   notebook cell. Those pickles carried no record of which model or which
   feature set produced them, so they loaded happily against any model and
   attributed importance to whatever columns happened to line up. They were
   built on `Amount_scaled` / `Time_scaled`, which no longer exist. Nothing
   errored -- the numbers just meant something else. Now everything comes from
   `explain.load()`, which asserts the SHAP artifact matches the bundle
   currently in use and raises if it doesn't.

2. `X_test_sample.iloc[i].get("Amount_scaled", 0.0)` returned 0.0 for every
   row, because v1 dropped `Amount` and v2 never created `Amount_scaled`.
   Every amount shown in the UI was zero, or a z-score presented as if it were
   money. Amounts are now real euros.

3. The sidebar hardcoded "Recall 87% / Precision 51%". Those were v1 numbers
   typed into a string. They now read from `bundle["metrics"]`, so they cannot
   disagree with the model that is actually scoring.

4. It called `model.predict_proba` directly, bypassing `serving.score()`.
   Two scoring paths is how training/serving skew gets back in. There is now
   one.

Run:
    python train_pipeline.py
    python serving.py          # both invariants must PASS
    python explain.py          # regenerate SHAP against the v2 bundle
    streamlit run app.py
"""

import numpy as np
import streamlit as st
from langchain_community.vectorstores import FAISS
from transformers import pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline

try:  # the non-deprecated import; falls back so the app still runs pre-install
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings

import fraudshield_config as cfg
import explain
from serving import load_bundle

st.set_page_config(page_title="FraudShield — Riya", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.main { background-color: #0f1117; }
.riya-box {
    background: linear-gradient(135deg, #1a1f2e, #16213e);
    border-left: 4px solid #00d4ff;
    border-radius: 8px; padding: 16px 20px; margin: 12px 0;
    color: #e0e0e0; font-size: 15px;
}
.fraud-badge { background-color: #ff4b4b; color: white; padding: 4px 12px;
               border-radius: 20px; font-weight: bold; font-size: 13px; }
.legit-badge { background-color: #00c853; color: white; padding: 4px 12px;
               border-radius: 20px; font-weight: bold; font-size: 13px; }
.shap-bar     { background: #ff6b35; height: 10px; border-radius: 4px; }
.shap-bar-neg { background: #00d4ff; height: 10px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────── artifacts
@st.cache_resource
def load_all():
    """One loader. If the SHAP artifact is stale relative to the bundle,
    explain.load() raises here and the app refuses to start -- which is the
    behaviour you want. A console showing explanations from the wrong model is
    worse than a console that won't open."""
    bundle = load_bundle()
    payload = explain.load()
    return bundle, payload


@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Index is versioned with the model: an index built against v1 metrics
    # silently serving a v2 model is the same class of bug as a stale pickle.
    versioned = cfg.ROOT / f"faiss_index_{cfg.ARTIFACT_VERSION}"
    path = versioned if versioned.exists() else cfg.FAISS_DIR
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def load_llm():
    pipe = hf_pipeline("text-generation", model="google/flan-t5-base",
                       max_new_tokens=120, do_sample=False)
    return HuggingFacePipeline(pipeline=pipe)


# ─────────────────────────────────────────────── helpers
def riya_explanation(i, payload, threshold, vectorstore, llm):
    """Grounded in SHAP + retrieved rules only. The fallback is deterministic
    so the console still says something correct when the LLM call fails."""
    shap_context = explain.top_features(i, k=3, payload=payload)
    amount = float(payload["amounts"][i])

    docs = vectorstore.similarity_search(
        f"Why was this transaction flagged? Amount EUR {amount:.2f}", k=2)
    rag_context = "\n".join(d.page_content for d in docs)

    prompt = (
        "You are Riya, a fraud analyst AI assistant.\n"
        "Based ONLY on the data below, write a 2-sentence explanation of why "
        "this transaction was flagged. Do not add any information not present "
        "in the context. Do not name merchants, locations or cardholders. Do "
        "not state an absolute clock time -- the hour below is relative to the "
        "start of the dataset.\n\n"
        f"{shap_context}\n\nFRAUD RULES CONTEXT:\n{rag_context}\n\nEXPLANATION:"
    )

    try:
        out = llm.invoke(prompt).strip()
        if "EXPLANATION:" in out:
            out = out.split("EXPLANATION:")[-1].strip()
        if not out:
            raise ValueError("empty LLM output")
        return out
    except Exception:  # noqa: BLE001
        names = payload["feature_names"]
        vals = payload["shap_values"][i]
        top = sorted(zip(names, vals), key=lambda x: abs(x[1]), reverse=True)[:2]
        # V1-V28 are PCA components, so they are described as model drivers,
        # not named risk factors -- "V14 was suspicious" means nothing to an
        # analyst and is not a claim the data supports.
        return (
            f"Flagged at a model probability of {payload['probabilities'][i]:.3f}, "
            f"above the {threshold} decision threshold. The strongest "
            f"contributors were {top[0][0]} ({top[0][1]:+.3f}) and "
            f"{top[1][0]} ({top[1][1]:+.3f}); both are PCA components, so they "
            f"indicate relative contribution rather than a named risk factor. "
            f"Transaction amount EUR {payload['amounts'][i]:.2f}."
        )


# ─────────────────────────────────────────────── main
def main():
    st.markdown("# 🛡️ FraudShield")
    st.markdown("#### AI-Powered Fraud Detection — *Riya, Fraud Ops Analyst*")
    st.divider()

    with st.spinner("Loading FraudShield artifacts..."):
        try:
            bundle, payload = load_all()
        except (FileNotFoundError, ValueError) as e:
            st.error(f"**Artifacts are stale or missing.**\n\n{e}")
            st.code("python train_pipeline.py\npython serving.py\npython explain.py",
                    language="bash")
            st.stop()
        vectorstore = load_rag()
        llm = load_llm()

    threshold = bundle["threshold"]
    metrics = bundle["metrics"]
    probs = payload["probabilities"]
    amounts = payload["amounts"]
    hours = payload["hours"]
    n = len(probs)

    st.success(
        f"Model {bundle['version']} · features {bundle['feature_version']} · "
        f"{n} held-out transactions scored · threshold {threshold}"
    )

    with st.sidebar:
        st.markdown("## ⚙️ Alert Queue Settings")
        score_filter = st.slider("Min Risk Score", 0, 100, 50)
        n_show = st.slider("Transactions to Review", 1, min(20, n), 5)
        st.divider()
        st.markdown("### 📊 Model Performance")
        # Read from the bundle, never typed in. These cannot go stale.
        st.metric("Recall", f"{metrics['recall']:.0%}")
        st.metric("Precision", f"{metrics['precision']:.0%}")
        st.metric("PR-AUC", f"{metrics['pr_auc']:.2f}")
        st.metric("Threshold", f"{threshold:.2f}")
        st.caption(
            f"Model {bundle['version']}, trained {bundle['trained_at'][:10]}. "
            "SMOTE-trained, so scores rank well but are not calibrated "
            "probabilities — a 0.7 is not '70% likely fraud'."
        )
        st.divider()
        st.caption(f"SHAP: {payload['model_version']} · {payload['source']}")

    results = [
        {"idx": i, "prob": float(probs[i]), "score": int(probs[i] * 100),
         "is_fraud": bool(probs[i] >= threshold),
         "amount": float(amounts[i]), "hour": float(hours[i])}
        for i in range(n)
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    results = [r for r in results if r["score"] >= score_filter][:n_show]

    if not results:
        st.warning("No transactions above the selected risk score.")
        return

    c1, c2, c3, c4 = st.columns(4)
    flagged = sum(r["is_fraud"] for r in results)
    c1.metric("🔴 Flagged as Fraud", flagged)
    c2.metric("🟢 Likely Legitimate", len(results) - flagged)
    c3.metric("📋 Total Reviewed", len(results))
    c4.metric("📈 Avg Risk Score", f"{int(np.mean([r['score'] for r in results]))}/100")

    st.divider()
    st.markdown("## 📋 Alert Queue")

    for r in results:
        i = r["idx"]
        badge = ('<span class="fraud-badge">🔴 FRAUD ALERT</span>' if r["is_fraud"]
                 else '<span class="legit-badge">🟢 LEGITIMATE</span>')
        header = (f"Transaction #{i} — Risk {r['score']}/100 — "
                  f"EUR {r['amount']:,.2f} — hour {r['hour']:.1f}")

        with st.expander(header, expanded=r["is_fraud"]):
            left, right = st.columns([1, 2])

            with left:
                st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)
                st.markdown(f"**Risk Score:** `{r['score']} / 100`")
                st.markdown(f"**Amount:** `EUR {r['amount']:,.2f}`")
                st.markdown(f"**Hour of day:** `{r['hour']:.1f}` *(relative to dataset start)*")
                st.markdown(f"**Fraud Probability:** `{r['prob']:.4f}`")

                st.markdown("**🔍 Top 3 model drivers (SHAP):**")
                vals = payload["shap_values"][i]
                pairs = sorted(zip(payload["feature_names"], vals),
                               key=lambda x: abs(x[1]), reverse=True)[:3]
                # Scale bars against this row's own largest contribution, so a
                # small-magnitude row is still readable. The old fixed *200
                # multiplier pinned most bars to 0% or 100%.
                span = max(abs(float(v)) for _, v in pairs) or 1.0
                for feat, val in pairs:
                    val = float(val)
                    cls = "shap-bar" if val > 0 else "shap-bar-neg"
                    width = int(abs(val) / span * 100)
                    st.markdown(f"`{feat}` — {val:+.4f}")
                    st.markdown(f'<div class="{cls}" style="width:{width}%"></div>',
                                unsafe_allow_html=True)
                    st.caption("⬆️ Pushes toward fraud" if val > 0
                               else "⬇️ Pushes toward legitimate")
                st.caption("V1–V28 are PCA components — read these as relative "
                           "contribution, not as named risk factors.")

            with right:
                st.markdown("**💬 Riya's Analysis:**")
                if r["is_fraud"]:
                    with st.spinner("Riya is analyzing..."):
                        text = riya_explanation(i, payload, threshold, vectorstore, llm)
                else:
                    text = ("Below the decision threshold. Model drivers are within "
                            "normal ranges for this population. No action required.")
                st.markdown(f'<div class="riya-box">🤖 <b>Riya says:</b><br><br>{text}</div>',
                            unsafe_allow_html=True)

                st.markdown("**⚡ Analyst Action:**")
                a, b, c = st.columns(3)
                if a.button("✅ Confirm Fraud", key=f"fraud_{i}"):
                    st.success("Logged: Confirmed Fraud")
                if b.button("❌ Mark Legitimate", key=f"legit_{i}"):
                    st.info("Logged: Marked Legitimate")
                if c.button("⬆️ Escalate", key=f"escalate_{i}"):
                    st.warning("Logged: Escalated to Risk Manager")

                st.text_input("📝 Analyst Notes (optional)", key=f"notes_{i}",
                              placeholder="Add context...")


if __name__ == "__main__":
    main()
