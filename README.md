<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&pause=1000&color=00A86B&center=true&vCenter=true&width=600&lines=VeriFact+%F0%9F%94%8D;AI-Powered+Fact+Checking+System" alt="VeriFact" />

# VeriFact 🔍
### Autonomous Research Agent for Information Verification

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-Llama--3.3--70B-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![FAST-NUCES](https://img.shields.io/badge/FAST--NUCES-Karachi-005C2E?style=for-the-badge)](https://nu.edu.pk)
[![AI Course](https://img.shields.io/badge/Artificial%20Intelligence-Spring%202026-1A3A5C?style=for-the-badge)](https://github.com/zia19068/VeriFact---Artificial-Intelligence-Project)

> **VeriFact** is a multi-stage AI pipeline that evaluates the truthfulness of any claim using real-time web retrieval, machine learning clustering, combinatorial optimisation, and adversarial LLM reasoning — delivering a verdict of **VERIFIED**, **FALSE**, **PARTIALLY TRUE**, or **INCONCLUSIVE** with full source citations.

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Pipeline Architecture](#-pipeline-architecture)
- [Core Algorithms](#-core-algorithms)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Sample Output](#-sample-output)
- [Team](#-team)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Agentic Query Generation** | LLM decomposes each claim into 5 angle-varied search queries |
| 🌐 **Live Web Retrieval** | DuckDuckGo search fetches up to 40 real-time snippets per claim |
| 🧠 **ML Evidence Clustering** | TF-IDF + K-Means removes redundant snippets, keeps diverse evidence |
| ⚖️ **LLM Relevance Scoring** | Each snippet scored on Relevance (0–10) and Stance (−10 to +10) |
| 🎒 **Knapsack Optimisation** | OR-Tools selects the optimal evidence subset within token limits |
| 🥊 **Minimax Debate** | Proponent vs Opponent adversarial reasoning before final verdict |
| 📊 **Structured Verdict** | JSON output with verdict, confidence, key findings, and citations |
| 💬 **Interactive Debate Mode** | Argue a position against the AI in real-time, evidence-backed debate |

---

## 🏗 Pipeline Architecture

```
User Claim
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1 │  Query Generation      │  Groq LLM (Llama-3.3-70B)  │
│  Stage 2 │  Web Retrieval         │  DuckDuckGo DDGS            │
│  Stage 3 │  Clustering            │  TF-IDF + K-Means           │
│  Stage 4 │  LLM Scoring           │  Relevance & Stance         │
│  Stage 5 │  Knapsack Selection    │  Google OR-Tools            │
│  Stage 6 │  Minimax Debate        │  Proponent / Opponent       │
│  Stage 7 │  Verdict Synthesis     │  Chain-of-Thought JSON      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
 Verdict: VERIFIED / FALSE / PARTIALLY TRUE / INCONCLUSIVE
 Confidence: HIGH / MEDIUM / LOW
 Sources: [cited URLs]
```

---

## 🧪 Core Algorithms

### 1. TF-IDF + K-Means Clustering
Vectorises all retrieved snippets and groups them into up to **12 clusters**. The best representative from each cluster is kept — maximising evidence diversity while removing duplicates.

### 2. LLM Relevance & Stance Scoring
Each snippet receives:
- **Relevance** `(0–10)` — how directly it addresses the claim
- **Stance** `(−10 to +10)` — supports (+) or refutes (−) the claim

Composite score formula:
```
score = stance × (relevance / 10) × domain_credibility × quality_penalty
```

Domain tiers: `Reuters / BBC / WHO / PubMed → ×1.4` | `Wikipedia / Bloomberg → ×1.1`

### 3. 0/1 Knapsack (Google OR-Tools)
Solves snippet selection as a **Branch-and-Bound Knapsack problem** — maximises total evidence value within a **6,000-character token budget**.

### 4. Minimax Debate
Before issuing a verdict, the system simulates adversarial reasoning:
- **Proponent** → Top-3 supporting snippets argue *for* the claim
- **Opponent** → Top-3 refuting snippets argue *against* the claim
- **Context** → High-relevance neutral snippets provide balance

---

## 🛠 Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| LLM Inference | `Groq API` (Llama-3.3-70B) | Query gen, scoring, debate, verdict |
| Web Search | `duckduckgo-search` (DDGS) | Real-time evidence retrieval |
| Vectorisation | `scikit-learn` TfidfVectorizer | TF-IDF feature extraction |
| Clustering | `scikit-learn` KMeans | Evidence deduplication |
| Optimisation | `Google OR-Tools` | 0/1 Knapsack snippet selection |
| API Client | `openai` SDK (Groq-compat.) | LLM API calls with retry logic |
| Language | `Python 3.11+` | Core implementation |

---

## ⚙️ Installation

### Prerequisites
- Python 3.11 or higher
- A free [Groq API key](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/zia19068/VeriFact---Artificial-Intelligence-Project.git
cd VeriFact---Artificial-Intelligence-Project/VeriFact-Project
```

### 2. Install dependencies
```bash
pip install groq openai duckduckgo-search scikit-learn ortools
```

### 3. Set your API key

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "your_groq_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set GROQ_API_KEY=your_groq_api_key_here
```

**macOS / Linux:**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

---

## 🚀 Usage

### Run VeriFact
```bash
python main.py
```

You will be prompted with two modes:

```
============================================================
         VERIFACT - AI FACT CHECKING SYSTEM
============================================================
Select mode:
  [1] Fact Check a Claim
  [2] Interactive Debate Mode
  [3] Exit
```

#### Mode 1 — Fact Check a Claim
Enter any claim and receive a full verdict with citations:
```
Enter claim: The Great Wall of China is visible from space.

[Generating queries...]  [Searching web...]  [Clustering...]
[Scoring evidence...]    [Optimising...]     [Debating...]

VERDICT:     FALSE
CONFIDENCE:  HIGH
SOURCES:     nasa.gov, bbc.com, scientificamerican.com ...
```

#### Mode 2 — Interactive Debate Mode
Argue any position against the AI in a multi-turn, evidence-backed debate.

---

## 📁 Project Structure

```
VeriFact---Artificial-Intelligence-Project/
│
├── VeriFact-Project/
│   ├── main.py                  # Core pipeline — all 7 stages
│   ├── static/                  # CSS & JS assets
│   └── templates/               # HTML templates
│
├── README.md                    # This file
├── VeriFact-Project Proposal.pdf
└── VeriFact-Project Report.pdf
```

---

## 📊 Sample Output

| Claim | Verdict | Confidence |
|---|---|---|
| Earth is ~4.5 billion years old | ✅ VERIFIED | HIGH |
| 8 glasses of water daily is mandated | 🟡 PARTIALLY TRUE | MEDIUM |
| COVID-19 vaccines contain microchips | ❌ FALSE | HIGH |
| Great Wall of China visible from space | ❌ FALSE | HIGH |
| Climate change driven by human activity | ✅ VERIFIED | HIGH |

---

## 👥 Team

| Name | Roll Number | GitHub |
|---|---|---|
| **OM** | 24K-0711 | [@OmKaran1111111](https://github.com/OmKaran1111111) |
| **Vishal** | 24K-0625 | [VishalParwani76](https://github.com/VishalParwani76) | |
| **Zia** | 24K-0817 | [@zia19068](https://github.com/zia19068) |

**Course:** Artificial Intelligence (AI) — Section BCS-4K  
**Instructor:** Sir Riaz Ahmed  
**Institution:** FAST-NUCES, Karachi — Spring 2026

---

<div align="center">

Made with ❤️ at **FAST-NUCES Karachi** · Spring 2026

</div>
