# FraudShield — MLOps & LLMOps layer

**Start with `FIXES.md`** — it lists everything that needs changing and why.
This file is the sequencing and the rationale behind the design.

Drop these five files in your FraudShield repo root (next to `creditcard.csv`).

```
Fraudshield/
├── creditcard.csv
├── fraudshield_config.py     ← all config, thresholds, versioned prompts
├── features.py               ← stateless feature engineering (hour_of_day)
├── train_pipeline.py         ← replaces EDA cells 9–16
├── serving.py                ← the ONLY scoring path
├── agents.py                 ← replaces LangGraph cells 1–6 and 12
├── build_golden_dataset.py   ← replaces Evals.py
├── monitoring.py             ← drift detection
├── eval_runner.py            ← golden-set scorer + LLM judge
├── artifacts/                ← created automatically
└── Fraudshield_evals/
```

```bash
pip install mlflow
python train_pipeline.py     # retrain, bundle, log to MLflow
python serving.py            # skew self-check — must PASS
python monitoring.py         # drift report
python eval_runner.py        # deterministic golden-set checks
python eval_runner.py --judge  # + LLM-as-judge on case notes
mlflow ui                    # http://localhost:5000
```

---

## Order of work

### Week 1 — correctness (do this before any new feature)

**1. Retrain with `train_pipeline.py`.** It fits the scaler on train only and
saves it *inside* the model bundle. Your old `fraudshield_model.pkl` has no
scaler attached, which is why every caller was fitting its own.

**2. Point everything at `serving.py`.** Replace, in the LangGraph notebook,
`Evals.py`, and the Streamlit app:

```python
# before — re-fits a scaler on whatever data happens to be in front of it
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
risk_score = model.predict_proba(features_df)[0][1]

# after
from serving import score_one
risk_score = score_one(transaction)["fraud_probability"]
```

**3. Run `python serving.py`.** It asserts a single transaction scores
identically alone and in a batch. That assertion is the whole point: it fails
loudly the next time someone re-introduces a `fit_transform` at inference.

**4. Fix the dead agent rule.** `amount > 100000` never fires — this dataset's
amounts are in euros and max out near 25,691. `AGENT_RULES` in the config sets
it to 2000. Re-run the golden set afterwards; the coverage check in
`eval_runner.py` will tell you whether both branches now execute.

### Week 2 — MLOps

**5. MLflow.** Already wired into `train_pipeline.py`. Every retrain logs
params, metrics, the threshold sweep, and the bundle. This is what lets you
answer "which run produced the model currently in the Streamlit app?"

**6. The promotion gate.** `PROMOTION_GATE` in the config is the bar a retrain
must clear before it ships. Right now it prints PASS/FAIL; in a real system it
would be the CI step that blocks the deploy.

**7. Drift monitoring.** `monitoring.py` computes PSI per feature and on the
score distribution. Run it against a held-out time slice.

> **A real finding from running this:** `Time_scaled` shows PSI above 10 on a
> later time slice. That is not noise — `Time` is *seconds since the first
> transaction in the file*, so in production it increases without bound and
> drifts permanently. Either drop it or convert it to hour-of-day
> (`Time % 86400 / 3600`), which is the signal you actually wanted. This is a
> good thing to have found yourself.

**8. Calibration — know the limitation.** SMOTE rebalances the training base
rate, so your model's outputs are good *rankings* but are not true
probabilities. A 0.7 output is not "a 70% chance of fraud." If someone asks,
say that, and mention `CalibratedClassifierCV` fit on an untouched validation
split as the fix. Being able to name this is worth more than fixing it.

### Week 3 — LLMOps

Your four agents are currently pure Python rules — there's no LLM in the graph
yet, so there's nothing to trace. LLMOps starts once the Case Notes agent
actually calls Groq. Sequence:

**9. Move Case Notes to an LLM** using `PROMPTS["case_notes"]` from the config.
Prompts live in version control with a version string, never inline in a
notebook cell.

**10. LangSmith tracing.** Four environment variables, zero code change:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=fraudshield
GROQ_API_KEY=your_key
```

Every `app.invoke()` is then captured with per-node latency, inputs, outputs
and token cost. The per-node latency view is what tells you which agent is
your bottleneck.

**11. LLM-as-judge.** `eval_runner.py --judge` grades generated case notes
against your hand-written `expected_case_note_summary` column on factual
consistency, decision match, and usefulness. Temperature 0, fixed prompt
version, both recorded in the report.

> Judges are themselves unvalidated models. Before quoting a judge score,
> hand-grade ~10 cases yourself and check the judge agrees. If it doesn't, the
> judge prompt is the bug, not the agent.

**12. Regression suite.** Deterministic checks are fast and free — run them on
every commit. The judge costs tokens — run it before releases.

---

## The distinction to draw in interviews

**MLOps** = the model is a versioned artifact with a reproducible training
path, a promotion gate, and drift alarms. Failures are numeric and detectable
by assertion.

**LLMOps** = the *prompt* is also a versioned artifact, outputs are
non-deterministic, and failures are semantic — a fluent, confident, wrong case
note. That's why the eval split is deterministic checks for decisions and a
graded judge for narrative, and why tracing matters more than it does for a
classifier: you can't unit-test a paragraph.

The strongest thing you have here is the training/serving skew: a bug that was
invisible in batch evaluation and would have been silent in production,
caught by an invariant rather than by looking at output. That is a better
story than the golden-dataset one because *you* found it and it explains why
the eval harness exists.
