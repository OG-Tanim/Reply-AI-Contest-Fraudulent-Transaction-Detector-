# Reply Mirror - Fraud Detection System

## Quick Start

### 1. Install Dependencies

```bash
cd The-Eye
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` with your API keys:

```bash
# Langfuse credentials
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://challenges.reply.com/langfuse

# OpenRouter API key
OPENROUTER_API_KEY=sk-or-...

# Team name
TEAM_NAME=your-team-name

# Do not edit
LANGFUSE_MEDIA_UPLOAD_ENABLED=false
```

### 3. Run Modes

```bash
# Run on EVALUATION data (default)
python3 main.py

# Run on TRAINING data (single level)
python3 main.py --train

# Run 5-level loop on TRAINING data (no final evaluation)
python3 main.py --train-full

# Run 5-level loop on TRAINING data, then EVALUATION
python3 main.py --full
```

### 4. Output

- **Output file:** `output.txt`
- **Memory bank:** `memory_bank.json`
- **Format:** UTF-8 plain text, one transaction ID per line

## Folder Structure

```
The-Eye/
├── Training-Data/       # Training dataset
├── Evaluation-Data/     # Evaluation dataset
├── output.txt          # Fraud predictions
├── memory_bank.json     # Persisted patterns
├── main.py             # Entry point
├── config/             # Configuration
└── src/                # Source code
    ├── data/           # Data loading & features
    ├── agents/         # 6 specialized agents
    ├── orchestrator/   # Meta-orchestrator
    ├── adaptation/      # Drift detection & memory
    └── output/         # Output generator
```

## Run Modes Explained

| Mode | Description |
|------|-------------|
| `python3 main.py` | Single level on evaluation data |
| `python3 main.py --train` | Single level on training data |
| `python3 main.py --train-full` | 5-level loop on training data only |
| `python3 main.py --full` | 5-level loop on training + final evaluation |

## Architecture

```
Layer 1: Data Sources (Transactions, GPS, SMS, Email, Users)
         ↓
Layer 2: Feature Engineering
         ↓
Layer 3: 6 Specialized Agents
         ├─ Transaction Anomaly (Isolation Forest)
         ├─ Behavioral Profiler (per-user baselines)
         ├─ Geospatial (GPS vs transaction location)
         ├─ Graph Network (centrality, ring detection)
         ├─ Temporal (time patterns, bursts)
         └─ Communications (LLM-based NLP)
         ↓
Layer 4: Meta-Orchestrator (weighted voting)
         ↓
Layer 5: Adaptation Engine (drift detection, memory bank)
         ↓
Layer 6: Output Generator → output.txt
```

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

## Notes

- Langfuse tracing is enabled by default
- Session ID is generated as `{TEAM_NAME}-{ULID}`
- Memory bank persists patterns across levels for cross-level transfer
