# FraudShield — what to fix

Ordered by consequence. Items marked **[done]** are already handled by the
files in this folder; items marked **[you]** need an edit in your notebooks
that I can't make from here.

---

## A. Correctness — these change your numbers

### A1. `Time` → `hour_of_day` **[done — `features.py`]**

`Time` is seconds elapsed since the first transaction in the file. It grows
without bound, so the model learned a number it will never see again and the
drift monitor screamed (PSI > 10) the moment it saw a later time slice.
Folding it modulo 86,400 gives a stationary feature carrying the signal you
actually wanted. After the fix, `hour_of_day` PSI came out at 0.0015.

Say this out loud when you present it: the dataset never records what t=0 was
in wall-clock terms, so this is a **relative** hour-of-day with an unknown
constant offset. The cyclical structure survives the offset so the model
learns fine, but you cannot claim "fraud peaks at 3am" — only "fraud
concentrates in a consistent low-volume window." Claiming the former is the
kind of thing an interviewer will pull on.

I left `hour_of_day` as a raw 0–24 value rather than sin/cos. Trees split on
thresholds and handle the non-linearity natively, and a raw hour is legible in
a SHAP plot and in a case note. `hour_sin = -0.71` is not something you can
put in front of a fraud analyst. `CYCLIC_HOUR = True` in `features.py` if you
ever swap in a linear model.

### A2. Scaling removed entirely **[done — `features.py`]**

I tested this: `StandardScaler` changes XGBoost's predictions by exactly
`0.0`. Trees split on thresholds, and any strictly monotonic transform gives
an identical tree at a transformed threshold. The scaling step in EDA cell 9
was doing nothing for your model.

This is also the real fix for the training/serving skew, and it's a better
answer than the one I gave you last time. The weak fix is "remember to save
the scaler." The strong fix is "have no fitted state in feature engineering at
all" — **stateless transforms cannot skew.** That's the version to say in an
interview.

If you later add a logistic-regression baseline, scaling starts to matter —
put it inside an `sklearn.Pipeline` with the estimator so the two can't be
saved separately.

### A3. Golden dataset was sampled from training rows **[done — `build_golden_dataset.py`]**

This one is worth dwelling on. The old `Evals.py` loaded all of
`creditcard.csv` and sampled from it, so roughly 80% of your 36 golden cases
were rows the model trained on. Its probabilities there are optimistically
biased by memorisation — a "borderline" case drawn from training data isn't
borderline for a fresh transaction. The hardest band, the one the whole
exercise exists to stress, was the least trustworthy.

`train_pipeline.py` now writes `artifacts/test_index_*.json` and the builder
samples only from held-out rows. **Rebuild and re-label your golden set**
after retraining. Painful, but the old labels were attached to inflated
probabilities.

### A4. The dead document-verification rule **[done — `agents.py`]**

`amount > 100000` could never fire: amounts are in euros and top out near
25,691. That branch never executed across all 36 cases, so your eval couldn't
distinguish "correctly passes" from "never ran." Ceiling is now 2,000 via
`AGENT_RULES`.

`eval_runner.py` has a coverage check that catches exactly this class of
problem — a rule that never fires is untested, not passing. Run it after you
re-run the graph and confirm both branches now execute.

### A5. Three thresholds for one decision **[done — `fraudshield_config.py`]**

EDA cell 15 printed the final report at 0.6, cell 16 saved 0.7, and the
compliance agent used a bare 0.8. All three now read from
`DECISION_THRESHOLD` / `AGENT_RULES`.

### A6. `tx["Amount", "N/A"]` in EDA cell 21 **[you]**

```python
amount = tx["Amount", "N/A"] if "Amount" in tx.index else 0.0
```

`X_test_sample` had `Amount` dropped in favour of `Amount_scaled`, so
`"Amount" in tx.index` is always False and this always returns `0.0`. Every
RAG context you've generated says `Amount: $0.00`. Under v2 `Amount` survives
feature engineering, so:

```python
amount = float(tx["Amount"])
```

---

## B. Notebook edits you need to make

### B1. LangGraph cell 10 — replace the scaler block **[you]**

```python
# before
scaler_amount = StandardScaler()
df['Amount_scaled'] = scaler_amount.fit_transform(df[['Amount']])
risk_score = model.predict_proba(features_df)[0][1]

# after
from serving import score
scored = score(df.loc[[fraud_tx.name]])
risk_score = scored["fraud_probability"].iloc[0]
```

### B2. LangGraph cells 1–6 — import instead of redefining **[done — `agents.py`]**

```python
from agents import build_graph, new_state, run_golden_set
app = build_graph()
results = run_golden_set(app)     # replaces cell 12
```

`new_state()` also fixes `decision` defaulting to `"approve"` before any agent
has run. A crash mid-graph previously left you holding an approval you never
made; it's now `"pending"`.

### B3. Hardcoded `C:\Users\Abcom\Desktop\...` paths **[done — `fraudshield_config.py`]**

Everything resolves relative to the config file. Nothing breaks when the repo
moves, and nobody reviewing your GitHub sees your desktop path.

### B4. The RAG knowledge base hardcodes stale metrics **[you]**

EDA cell 18 types these into a string literal:

> precision: 51% … recall: 87% … 0.7 threshold

The moment you retrain, your RAG layer confidently cites numbers that are no
longer true, with retrieval scores that look perfectly healthy. This is a
genuinely LLMOps-flavoured failure: nothing errors, the answer is just wrong.

Generate that section instead of typing it:

```python
from serving import load_bundle
b = load_bundle()
m = b["metrics"]
metrics_section = f"""
5. CURRENT MODEL PERFORMANCE (model {b['version']}, threshold {b['threshold']}):
Precision: {m['precision']:.0%} - of flagged transactions, this share are actual fraud.
Recall: {m['recall']:.0%} - this share of all actual fraud is caught.
We prioritise recall: a missed fraud costs more than a false positive an analyst clears.
"""
fraud_knowledge = STATIC_RULES + metrics_section
```

Then version the FAISS index with the model — save to
`faiss_index_{bundle['version']}` and load the matching one. An index built
against v1 metrics silently serving a v2 model is the same bug wearing a hat.

### B5. `HuggingFaceEmbeddings` import is deprecated **[you]**

```python
# before
from langchain_community.embeddings import HuggingFaceEmbeddings
# after
from langchain_huggingface import HuggingFaceEmbeddings   # pip install langchain-huggingface
```

### B6. `ask_fraudshield` never calls an LLM **[you]**

It retrieves chunks and returns a formatted prompt string. There's no
generation step, so what you have today is retrieval, not RAG. Same for the
Case Notes agent — it's string concatenation. Wire `GROQ_API_KEY` (already
loaded in your last cell) into both, using `cfg.PROMPTS["case_notes"]`.

Keep the deterministic `case_notes_agent` in `agents.py` as the fallback path
for when the LLM call times out. Having a non-LLM fallback is a real design
answer to "what happens when the model provider is down."

---

## C. Limitations to state, not fix

These are worth more as things you volunteer than as things you solve.

**C1. SMOTE leaves the probabilities uncalibrated.** Oversampling distorts the
base rate, so your outputs rank well but aren't probabilities. A 0.7 is not
"70% likely fraud." `train_pipeline.py` logs a Brier score so you can show
you've measured it. The fix if pressed: `CalibratedClassifierCV` fit on an
untouched validation split.

**C2. `V1`–`V28` are PCA outputs.** You can't name them in a case note. "V14
was the strongest driver" means nothing to a fraud analyst. Frame SHAP output
as relative contribution, not as named risk factors — and note that in a real
deployment you'd have the pre-PCA features and this problem disappears.

**C3. Performance decay is only observable with a lag.** Fraud labels arrive
via chargebacks, 30–90 days later. That's precisely why feature and prediction
drift are your early-warning system rather than a nice-to-have — they're the
only signal available in week one.

---

## Run order after all of this

```bash
python train_pipeline.py        # retrain on v2 features
python serving.py               # both invariants must PASS
python build_golden_dataset.py  # resample from held-out rows
#    -> hand-label the expected_* columns (this is the slow part)
python agents.py                # run the graph over the golden set
python eval_runner.py           # deterministic checks + branch coverage
python monitoring.py            # no-drift control
python monitoring.py --simulate # prove the alarm fires
mlflow ui
```
