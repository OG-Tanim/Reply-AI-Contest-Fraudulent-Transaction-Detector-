import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_DATA_DIR = BASE_DIR / "Training-Data"
EVALUATION_DATA_DIR = BASE_DIR / "Evaluation-Data"
OUTPUT_FILE = BASE_DIR / "output.txt"

NUM_LEVELS = 5

AGENT_WEIGHTS = {
    "transaction_anomaly": 0.20,
    "behavioral_profiler": 0.20,
    "geospatial": 0.15,
    "graph_network": 0.15,
    "temporal": 0.15,
    "communications": 0.15,
}

LEVEL_CONFIGS = {
    1: {
        "name": "Level 1",
        "min_fraud_rate": 0.15,
        "agent_weights": {
            "transaction_anomaly": 0.20,
            "behavioral_profiler": 0.20,
            "geospatial": 0.15,
            "graph_network": 0.15,
            "temporal": 0.15,
            "communications": 0.15,
        },
        "retrain_threshold": 0.15,
    },
    2: {
        "name": "Level 2",
        "min_fraud_rate": 0.18,
        "agent_weights": {
            "transaction_anomaly": 0.18,
            "behavioral_profiler": 0.20,
            "geospatial": 0.16,
            "graph_network": 0.16,
            "temporal": 0.15,
            "communications": 0.15,
        },
        "retrain_threshold": 0.17,
    },
    3: {
        "name": "Level 3",
        "min_fraud_rate": 0.20,
        "agent_weights": {
            "transaction_anomaly": 0.16,
            "behavioral_profiler": 0.20,
            "geospatial": 0.17,
            "graph_network": 0.17,
            "temporal": 0.15,
            "communications": 0.15,
        },
        "retrain_threshold": 0.19,
    },
    4: {
        "name": "Level 4",
        "min_fraud_rate": 0.22,
        "agent_weights": {
            "transaction_anomaly": 0.15,
            "behavioral_profiler": 0.20,
            "geospatial": 0.18,
            "graph_network": 0.18,
            "temporal": 0.14,
            "communications": 0.15,
        },
        "retrain_threshold": 0.21,
    },
    5: {
        "name": "Level 5",
        "min_fraud_rate": 0.25,
        "agent_weights": {
            "transaction_anomaly": 0.14,
            "behavioral_profiler": 0.20,
            "geospatial": 0.18,
            "graph_network": 0.18,
            "temporal": 0.15,
            "communications": 0.15,
        },
        "retrain_threshold": 0.23,
    },
}

ANOMALY_THRESHOLD = 0.6
MIN_FRAUD_DETECTION_RATE = 0.15

DRIFT_CONFIG = {
    "ddm_warning_threshold": 2.0,
    "ddm_drift_threshold": 3.0,
    "ddm_min_samples": 30,
    "adwin_delta": 0.002,
    "level_drift_threshold": 0.20,
}

MEMORY_CONFIG = {
    "similarity_threshold": 0.7,
    "max_patterns_per_level": 50,
    "enable_fewshot": True,
}

OPENROUTER_MODEL = "gpt-4o-preview"

COST_MATRIX = {
    "false_positive_cost": 1.0,
    "false_negative_cost": 2.0,
}

LANGFUSE_HOST = "https://challenges.reply.com/langfuse"

MEMORY_BANK_FILE = BASE_DIR / "memory_bank.json"
