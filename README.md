# 🛡️ FraudShield — AI-Powered Fraud Detection System

## 🚨 The Problem

Digital payments in India are growing at an unprecedented pace — UPI alone
processed over 18,000 crore transactions in FY2024. But with this growth
comes a dark side: financial fraud is rising sharply. Banks and fintechs
lose billions annually to fraudulent transactions, and traditional
rule-based systems struggle to keep up with evolving fraud patterns.
Most fraud goes undetected until it's too late.

## 💡 The Need

Financial institutions need intelligent, real-time fraud detection that
goes beyond static rules. They need systems that can learn from patterns,
explain their decisions, and allow analysts to interact with flagged
transactions naturally — not just receive a binary yes/no alert.

## 🛡️ What FraudShield Does

FraudShield is an AI-powered fraud detection system built for the fintech
ecosystem. It uses XGBoost to detect fraudulent transactions with 87%
recall, SHAP for explainability, and a LangChain RAG layer that lets
fraud analysts ask natural language questions like
*"Why was this transaction flagged?"* — powered by an AI analyst
persona called **Riya**.

---

## ✨ Key Features

- 🔍 **Real-time fraud detection** on credit card transactions
- 🤖 **XGBoost ML model** — 87% recall, optimized threshold at 0.7
- 📊 **SHAP Explainability** — understand *why* a transaction is flagged
- 💬 **Natural Language Interface** via LangChain RAG + Riya (AI Analyst)
- 🖥️ **Streamlit UI** — interactive dashboard for fraud analysts
- ✅ **LangGraph Multi-Agent System**- 4 agents (Document Verification , Risk Scoring , Compliance , Case Notes) with human-in-the-loop escalation

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost |
| Explainability | SHAP |
| NLP / RAG | LangChain + FAISS |
| UI | Streamlit |
| Language | Python |
| Dataset | Kaggle Credit Card Fraud (284,807 transactions) |
|Multi-Agent | LangGraph

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Pratiksha-git-bit/Fraudshield.git
cd Fraudshield

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py


## Project structure

FraudShield/
├── app.py                       # Streamlit UI
├── fraudshield_eda.ipynb        # Exploratory Data Analysis
├── fraudshield_model.pkl        # Trained XGBoost model
├── shap_values.pkl              # SHAP explainability values
├── faiss_index/                 # RAG vector store
├── best_threshold.txt           # Optimal classification threshold
└── README.md
├── fraudshield_langgraph.ipynb  # LangGraph Multi-Agent Pipeline

