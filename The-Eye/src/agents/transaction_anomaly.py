import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import IsolationForest

from src.data.feature_store import FeatureStore


class TransactionAnomalyAgent:
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.model = None
        self._train()
    
    def _train(self):
        features = self.feature_store.compute_all_features()
        
        X = []
        for f in features:
            X.append([
                f["amount"],
                f["amount_zscore"],
                f["balance_delta"],
                f["transaction_type_encoded"],
                f["payment_method_encoded"],
                f["hour"],
                f["day_of_week"],
                f["recipient_novelty"],
                f["amount_deviation"],
            ])
        
        X = np.array(X)
        
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.15,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X)
    
    def score(self, transaction: Dict[str, Any]) -> float:
        features = self.feature_store.extract_features(transaction)
        
        x = np.array([[
            features["amount"],
            features["amount_zscore"],
            features["balance_delta"],
            features["transaction_type_encoded"],
            features["payment_method_encoded"],
            features["hour"],
            features["day_of_week"],
            features["recipient_novelty"],
            features["amount_deviation"],
        ]])
        
        anomaly_score = self.model.decision_function(x)[0]
        fraud_prob = 1.0 - (anomaly_score + 0.5)
        fraud_prob = max(0.0, min(1.0, fraud_prob))
        
        return fraud_prob
    
    def score_all(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            txn["transaction_id"]: self.score(txn)
            for txn in transactions
        }
