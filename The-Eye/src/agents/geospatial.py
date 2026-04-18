import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from typing import List, Dict, Any, Optional, Tuple

from src.data.feature_store import FeatureStore


class GeospatialAgent:
    EARTH_RADIUS_KM = 6371.0
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self._build_location_profiles()
    
    def _build_location_profiles(self):
        self.location_profiles = {}
        
        for biotag, baseline in self.feature_store.location_baselines.items():
            self.location_profiles[biotag] = {
                "mean_lat": baseline["mean_lat"],
                "mean_lng": baseline["mean_lng"],
                "lat_std": baseline["lat_std"],
                "lng_std": baseline["lng_std"],
                "cities": baseline["cities"],
            }
    
    def haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return self.EARTH_RADIUS_KM * c
    
    def _find_transaction_location(self, transaction: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        location_name = transaction.get("location")
        if not location_name:
            return None
        
        for loc in self.feature_store.locations:
            if loc.get("city") and location_name.lower() in loc.get("city", "").lower():
                return (loc["lat"], loc["lng"])
        
        return None
    
    def _get_user_location_at_time(self, sender_id: str, timestamp, max_hours_before: int = 24) -> Optional[Tuple[float, float]]:
        user_locs = self.feature_store.get_user_locations(sender_id)
        
        if not user_locs:
            return None
        
        best_loc = None
        best_diff = float('inf')
        
        for loc in user_locs:
            if loc["timestamp"]:
                diff_hours = abs((timestamp - loc["timestamp"]).total_seconds()) / 3600
                if diff_hours <= max_hours_before and diff_hours < best_diff:
                    best_diff = diff_hours
                    best_loc = (loc["lat"], loc["lng"])
        
        return best_loc
    
    def score(self, transaction: Dict[str, Any]) -> float:
        sender_id = transaction["sender_id"]
        timestamp = transaction.get("timestamp")
        
        if not timestamp:
            return 0.0
        
        if sender_id not in self.location_profiles:
            if transaction.get("location"):
                return 0.5
            return 0.0
        
        txn_location = self._find_transaction_location(transaction)
        
        if not txn_location:
            return 0.0
        
        user_recent_loc = self._get_user_location_at_time(sender_id, timestamp)
        
        if not user_recent_loc:
            profile = self.location_profiles[sender_id]
            user_recent_loc = (profile["mean_lat"], profile["mean_lng"])
        
        distance = self.haversine_distance(
            user_recent_loc[0], user_recent_loc[1],
            txn_location[0], txn_location[1]
        )
        
        if distance > 500:
            return 1.0
        elif distance > 200:
            return 0.7
        elif distance > 100:
            return 0.4
        elif distance > 50:
            return 0.2
        
        return 0.0
    
    def score_all(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            txn["transaction_id"]: self.score(txn)
            for txn in transactions
        }
