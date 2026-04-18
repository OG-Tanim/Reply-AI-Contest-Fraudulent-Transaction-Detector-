import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime
from typing import Dict, List, Any, Optional


class FeatureStore:
    def __init__(self, data: Dict[str, Any]):
        self.transactions = data["transactions"]
        self.users = data["users"]
        self.locations = data["locations"]
        self.sms = data["sms"]
        self.emails = data["emails"]
        
        self._compute_user_baselines()
        self._compute_location_baselines()
        
    def _compute_user_baselines(self):
        self.user_baselines = {}
        
        sender_txns = {}
        for txn in self.transactions:
            sender = txn["sender_id"]
            if sender not in sender_txns:
                sender_txns[sender] = []
            sender_txns[sender].append(txn)
        
        for sender_id, txns in sender_txns.items():
            amounts = [t["amount"] for t in txns]
            self.user_baselines[sender_id] = {
                "mean_amount": sum(amounts) / len(amounts) if amounts else 0,
                "std_amount": self._std(amounts) if len(amounts) > 1 else 0,
                "tx_count": len(txns),
                "txn_types": list(set(t["transaction_type"] for t in txns)),
                "payment_methods": list(set(t["payment_method"] for t in txns if t["payment_method"])),
                "recipients": list(set(t["recipient_id"] for t in txns)),
                "times": [t["timestamp"] for t in txns if t["timestamp"]],
            }
    
    def _compute_location_baselines(self):
        self.location_baselines = {}
        
        biotag_locs = {}
        for loc in self.locations:
            biotag = loc["biotag"]
            if biotag not in biotag_locs:
                biotag_locs[biotag] = []
            biotag_locs[biotag].append(loc)
        
        for biotag, locs in biotag_locs.items():
            lats = [l["lat"] for l in locs]
            lngs = [l["lng"] for l in locs]
            self.location_baselines[biotag] = {
                "mean_lat": sum(lats) / len(lats),
                "mean_lng": sum(lngs) / len(lngs),
                "lat_std": self._std(lats),
                "lng_std": self._std(lngs),
                "cities": list(set(l["city"] for l in locs if l["city"])),
            }
    
    def _std(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def extract_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        features = {}
        
        features["amount"] = transaction["amount"]
        features["amount_zscore"] = self._amount_zscore(transaction)
        features["balance_after"] = transaction["balance_after"]
        features["balance_delta"] = self._balance_delta(transaction)
        
        features["transaction_type"] = transaction["transaction_type"]
        features["transaction_type_encoded"] = self._encode_type(transaction["transaction_type"])
        
        features["payment_method"] = transaction["payment_method"]
        features["payment_method_encoded"] = self._encode_payment_method(transaction["payment_method"])
        
        features["has_location"] = transaction["location"] is not None
        features["location"] = transaction["location"]
        
        features["has_sender_iban"] = transaction["sender_iban"] is not None
        features["has_recipient_iban"] = transaction["recipient_iban"] is not None
        
        features["timestamp"] = transaction["timestamp"]
        features["hour"] = transaction["timestamp"].hour if transaction["timestamp"] else 12
        features["day_of_week"] = transaction["timestamp"].weekday() if transaction["timestamp"] else 0
        features["is_weekend"] = features["day_of_week"] >= 5
        features["is_night"] = features["hour"] >= 22 or features["hour"] < 6
        
        features["sender_id"] = transaction["sender_id"]
        features["recipient_id"] = transaction["recipient_id"]
        
        features["recipient_novelty"] = self._recipient_novelty(transaction)
        features["amount_deviation"] = self._amount_deviation(transaction)
        features["time_deviation"] = self._time_deviation(transaction)
        features["velocity_anomaly"] = self._velocity_anomaly(transaction)
        
        features["description"] = transaction.get("description") or ""
        features["has_description"] = transaction.get("description") is not None
        
        return features
    
    def _amount_zscore(self, transaction: Dict[str, Any]) -> float:
        sender = transaction["sender_id"]
        if sender in self.user_baselines:
            baseline = self.user_baselines[sender]
            if baseline["std_amount"] > 0:
                zscore = abs(transaction["amount"] - baseline["mean_amount"]) / baseline["std_amount"]
                return zscore
        return 0.0
    
    def _balance_delta(self, transaction: Dict[str, Any]) -> float:
        return transaction["balance_after"] - transaction["amount"]
    
    def _encode_type(self, txn_type: str) -> int:
        type_map = {"transfer": 0, "e-commerce": 1, "direct_debit": 2, "in-person payment": 3, "withdrawal": 4}
        return type_map.get(txn_type, 0)
    
    def _encode_payment_method(self, method: Optional[str]) -> int:
        if not method:
            return 0
        method_map = {"debit card": 1, "mobile device": 2, "smartwatch": 3, "GooglePay": 4, "PayPal": 5}
        return method_map.get(method.lower(), 0)
    
    def _recipient_novelty(self, transaction: Dict[str, Any]) -> float:
        sender = transaction["sender_id"]
        recipient = transaction["recipient_id"]
        if sender in self.user_baselines:
            recipients = self.user_baselines[sender]["recipients"]
            if recipient in recipients:
                return 0.0
            return 1.0
        return 0.5
    
    def _amount_deviation(self, transaction: Dict[str, Any]) -> float:
        sender = transaction["sender_id"]
        if sender in self.user_baselines:
            baseline = self.user_baselines[sender]
            mean = baseline["mean_amount"]
            if mean > 0:
                return abs(transaction["amount"] - mean) / mean
        return 0.0
    
    def _time_deviation(self, transaction: Dict[str, Any]) -> float:
        sender = transaction["sender_id"]
        if sender in self.user_baselines and transaction["timestamp"]:
            baseline = self.user_baselines[sender]
            if baseline["times"]:
                avg_hour = sum(t.hour for t in baseline["times"]) / len(baseline["times"])
                hour_diff = abs(transaction["timestamp"].hour - avg_hour)
                hour_diff = min(hour_diff, 24 - hour_diff)
                return hour_diff / 12.0
        return 0.0
    
    def _velocity_anomaly(self, transaction: Dict[str, Any]) -> float:
        sender = transaction["sender_id"]
        if sender in self.user_baselines and transaction["timestamp"]:
            baseline = self.user_baselines[sender]
            times = sorted([t for t in baseline["times"] if t is not None])
            if len(times) > 1:
                intervals = [(times[i+1] - times[i]).total_seconds() / 3600 for i in range(len(times)-1)]
                avg_interval = sum(intervals) / len(intervals) if intervals else 24
                txn_hour = transaction["timestamp"]
                recent_times = [t for t in times if (txn_hour - t).total_seconds() < 86400]
                if recent_times:
                    time_since_last = (txn_hour - max(recent_times)).total_seconds() / 3600
                    if time_since_last < avg_interval * 0.1:
                        return 1.0
        return 0.0
    
    def get_user_transactions(self, user_id: str) -> List[Dict[str, Any]]:
        return [t for t in self.transactions if t["sender_id"] == user_id]
    
    def get_user_locations(self, biotag: str) -> List[Dict[str, Any]]:
        return [l for l in self.locations if l["biotag"] == biotag]
    
    def get_user_ibans(self) -> Dict[str, str]:
        ibans = {}
        for iban, user in self.users.items():
            ibans[user.get("first_name", "").lower()] = iban
            ibans[user.get("last_name", "").lower()] = iban
        return ibans
    
    def compute_all_features(self) -> List[Dict[str, Any]]:
        return [self.extract_features(txn) for txn in self.transactions]
