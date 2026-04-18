import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from typing import List, Dict, Any

from src.data.feature_store import FeatureStore


class BehavioralProfilerAgent:
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self._build_profiles()
    
    def _build_profiles(self):
        self.user_profiles = {}
        
        for sender_id, baseline in self.feature_store.user_baselines.items():
            self.user_profiles[sender_id] = {
                "mean_amount": baseline["mean_amount"],
                "std_amount": baseline["std_amount"],
                "typical_types": baseline["txn_types"],
                "typical_methods": baseline["payment_methods"],
                "typical_recipients": baseline["recipients"],
                "typical_hours": self._compute_typical_hours(baseline["times"]),
                "typical_days": self._compute_typical_days(baseline["times"]),
            }
    
    def _compute_typical_hours(self, times: List) -> Dict[int, int]:
        hours = {}
        for t in times:
            if t:
                hour = t.hour
                hours[hour] = hours.get(hour, 0) + 1
        return hours
    
    def _compute_typical_days(self, times: List) -> Dict[int, int]:
        days = {}
        for t in times:
            if t:
                day = t.weekday()
                days[day] = days.get(day, 0) + 1
        return days
    
    def score(self, transaction: Dict[str, Any]) -> float:
        sender_id = transaction["sender_id"]
        
        if sender_id not in self.user_profiles:
            return 0.3
        
        profile = self.user_profiles[sender_id]
        scores = []
        
        amount_zscore = abs(transaction["amount"] - profile["mean_amount"]) / (profile["std_amount"] + 1e-6)
        amount_score = min(1.0, amount_zscore / 3.0)
        scores.append(amount_score * 0.3)
        
        if transaction["transaction_type"] not in profile["typical_types"]:
            scores.append(0.2)
        else:
            scores.append(0.0)
        
        if transaction["payment_method"] and transaction["payment_method"] not in profile["typical_methods"]:
            scores.append(0.15)
        else:
            scores.append(0.0)
        
        if transaction["recipient_id"] not in profile["typical_recipients"]:
            recipient_score = 0.25
            scores.append(recipient_score)
        else:
            scores.append(0.0)
        
        if transaction["timestamp"]:
            hour = transaction["timestamp"].hour
            typical_hours = profile["typical_hours"]
            if typical_hours:
                max_hour_count = max(typical_hours.values())
                hour_count = typical_hours.get(hour, 0)
                if hour_count < max_hour_count * 0.1:
                    scores.append(0.1)
                else:
                    scores.append(0.0)
        
        return min(1.0, sum(scores))
    
    def score_all(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            txn["transaction_id"]: self.score(txn)
            for txn in transactions
        }
