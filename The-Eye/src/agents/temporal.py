import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime, timedelta
from typing import List, Dict, Any
from collections import defaultdict

from src.data.feature_store import FeatureStore


class TemporalAgent:
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self._build_temporal_patterns()
    
    def _build_temporal_patterns(self):
        self.hour_distribution = defaultdict(int)
        self.day_distribution = defaultdict(int)
        self.month_distribution = defaultdict(int)
        
        self.user_hour_patterns = defaultdict(lambda: defaultdict(int))
        self.user_day_patterns = defaultdict(lambda: defaultdict(int))
        
        for txn in self.feature_store.transactions:
            if txn["timestamp"]:
                hour = txn["timestamp"].hour
                day = txn["timestamp"].weekday()
                month = txn["timestamp"].month
                sender = txn["sender_id"]
                
                self.hour_distribution[hour] += 1
                self.day_distribution[day] += 1
                self.month_distribution[month] += 1
                
                self.user_hour_patterns[sender][hour] += 1
                self.user_day_patterns[sender][day] += 1
        
        total_hours = sum(self.hour_distribution.values())
        self.hour_probs = {
            h: c / total_hours if total_hours > 0 else 0
            for h, c in self.hour_distribution.items()
        }
    
    def _detect_burst(self, transaction: Dict[str, Any]) -> float:
        sender = transaction["sender_id"]
        timestamp = transaction.get("timestamp")
        
        if not timestamp:
            return 0.0
        
        recent_txns = [
            t for t in self.feature_store.transactions
            if t["sender_id"] == sender
            and t["timestamp"]
            and 0 < (timestamp - t["timestamp"]).total_seconds() < 86400
        ]
        
        if len(recent_txns) > 5:
            return 1.0
        elif len(recent_txns) > 3:
            return 0.6
        elif len(recent_txns) > 1:
            return 0.3
        
        return 0.0
    
    def _unusual_hour_score(self, transaction: Dict[str, Any]) -> float:
        sender = transaction["sender_id"]
        timestamp = transaction.get("timestamp")
        
        if not timestamp:
            return 0.0
        
        hour = timestamp.hour
        
        if sender in self.user_hour_patterns:
            user_hours = self.user_hour_patterns[sender]
            total = sum(user_hours.values())
            if total > 0:
                hour_freq = user_hours.get(hour, 0) / total
                if hour_freq < 0.05 and hour in [0, 1, 2, 3, 4, 5]:
                    return 0.8
                elif hour_freq < 0.02:
                    return 0.4
        
        if self.hour_probs.get(hour, 0) < 0.02:
            return 0.3
        
        return 0.0
    
    def _unusual_day_score(self, transaction: Dict[str, Any]) -> float:
        sender = transaction["sender_id"]
        timestamp = transaction.get("timestamp")
        
        if not timestamp:
            return 0.0
        
        day = timestamp.weekday()
        is_weekend = day >= 5
        
        if sender in self.user_day_patterns:
            user_days = self.user_day_patterns[sender]
            total = sum(user_days.values())
            if total > 0:
                day_freq = user_days.get(day, 0) / total
                if is_weekend and day_freq < 0.1:
                    return 0.5
        
        return 0.0
    
    def _rapid_succession_score(self, transaction: Dict[str, Any]) -> float:
        sender = transaction["sender_id"]
        timestamp = transaction.get("timestamp")
        
        if not timestamp:
            return 0.0
        
        recent_same_hour = [
            t for t in self.feature_store.transactions
            if t["sender_id"] == sender
            and t["timestamp"]
            and 0 < (timestamp - t["timestamp"]).total_seconds() < 3600
        ]
        
        if len(recent_same_hour) >= 3:
            return 1.0
        elif len(recent_same_hour) >= 2:
            return 0.6
        
        return 0.0
    
    def score(self, transaction: Dict[str, Any]) -> float:
        scores = []
        
        burst_score = self._detect_burst(transaction)
        if burst_score > 0:
            scores.append(burst_score * 0.4)
        
        hour_score = self._unusual_hour_score(transaction)
        if hour_score > 0:
            scores.append(hour_score * 0.3)
        
        day_score = self._unusual_day_score(transaction)
        if day_score > 0:
            scores.append(day_score * 0.2)
        
        rapid_score = self._rapid_succession_score(transaction)
        if rapid_score > 0:
            scores.append(rapid_score * 0.1)
        
        return min(1.0, sum(scores))
    
    def score_all(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            txn["transaction_id"]: self.score(txn)
            for txn in transactions
        }
