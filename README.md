# The-Eye - Reply Mirror Fraud Detection System

## What It Does

**The-Eye** is a multi-agent AI system designed for the Reply Mirror AI Challenge. It detects financial fraud in evolving environments where malicious actors constantly adapt their tactics to evade detection.

### The Challenge

In 2087's Reply Mirror metropolis, financial fraud is sophisticated and adaptive. Attackers:
- Target new merchants and transaction categories
- Shift temporal habits (daytime → late-night)
- Operate across changing geographic regions
- Vary transaction amounts and frequency
- Create new deceptive behavioral sequences

Static models fail. Only dynamic, continuously learning systems succeed.

---

## Architecture

The system implements a **6-layer agent-based architecture**:

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA SOURCES                                              │
│  • Transactions (amount, type, IBAN, timestamp)                    │
│  • GPS Locations (biotag, lat/lng, datetime)                       │
│  • SMS Threads (conversation text)                                 │
│  • Email Messages (full mail threads)                              │
│  • User Profiles (demographics, behavior descriptions)              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: FEATURE ENGINEERING                                       │
│  • Amount z-scores, balance deltas                                  │
│  • Time-of-day, day-of-week patterns                                │
│  • Recipient novelty, velocity anomalies                            │
│  • GPS-transaction distance calculations                            │
│  • Payment method encoding                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: SPECIALIZED AGENTS (6 Experts)                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐                  │
│  │ Transaction Anomaly │  │ Behavioral Profiler │                  │
│  │ • Isolation Forest  │  │ • Per-user baselines│                  │
│  │ • Amount outliers   │  │ • Deviation scoring │                  │
│  │ • Frequency patterns│  │ • Habit modeling   │                  │
│  └─────────────────────┘  └─────────────────────┘                  │
│  ┌─────────────────────┐  ┌─────────────────────┐                  │
│  │    Geospatial       │  │   Graph Network     │                  │
│  │ • GPS vs Txn loc   │  │ • Centrality       │                  │
│  │ • Impossible travel │  │ • Ring detection   │                  │
│  │ • Jurisdiction      │  │ • Money mule paths │                  │
│  └─────────────────────┘  └─────────────────────┘                  │
│  ┌─────────────────────┐  ┌─────────────────────┐                  │
│  │     Temporal       │  │   Communications    │                  │
│  │ • Time-of-day      │  │ • LLM-based NLP    │                  │
│  │ • Burst detection  │  │ • Phishing signals │                  │
│  │ • Schedule shifts  │  │ • Social eng.      │                  │
│  └─────────────────────┘  └─────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4: META-ORCHESTRATOR                                         │
│  • Weighted voting across all agent scores                          │
│  • Asymmetric cost calibration (FP ≠ FN penalty)                   │
│  • Dynamic threshold per challenge level                            │
│  • Final fraud / legitimate decision                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 5: ADAPTATION ENGINE                                         │
│  ┌──────────────────┐  ┌──────────────────┐                        │
│  │  Drift Detection │  │   Memory Bank    │                        │
│  │  • DDM/ADWIN    │  │  • Pattern store │                        │
│  │  • Concept drift │  │  • Few-shot rec. │                        │
│  │  • Alert triggers│  │  • Cross-level  │                        │
│  └──────────────────┘  └──────────────────┘                        │
│  • Per-agent retraining when patterns shift                         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 6: OUTPUT GENERATOR                                          │
│  • Precision-recall threshold optimization                          │
│  • Fraud transaction IDs → output.txt                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why 6 Specialized Agents?

Fraud evolves along **multiple dimensions simultaneously**:
- Amount patterns may look normal while geographic patterns are off
- A single model averaging everything misses uncorrelated signals
- Each agent is an expert in one dimension → independent probability score
- Fusion step becomes more informative

---

## Key Design Decisions

| Component | Why | Impact |
|-----------|-----|--------|
| **6 Agents** | Different fraud dimensions need specialized detection | Better signal separation |
| **Meta-Orchestrator** | Asymmetric FP/FN penalties require cost-aware fusion | Balances precision/recall |
| **Drift Detection** | Attackers change tactics between levels | Triggers retraining |
| **Memory Bank** | Past patterns inform future detection | Few-shot learning |
| **LLM Communications** | Social engineering precedes transactions | Pre-transaction signal |

---

## Run Modes

```bash
# Single level evaluation
python3 main.py

# Single level on training data
python3 main.py --train

# 5-level loop on training data
python3 main.py --train-full

# 5-level loop + final evaluation
python3 main.py --full
```

---

## Output

- **`output.txt`** - UTF-8 plain text, one fraud transaction ID per line
- **`memory_bank.json`** - Persisted fraud patterns for cross-level transfer

---

## Quick Start

```bash
cd The-Eye
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure .env with API keys
python3 main.py
```

See [EXECUTION.md](EXECUTION.md) for detailed instructions.
