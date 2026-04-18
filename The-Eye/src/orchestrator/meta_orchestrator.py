from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import AGENT_WEIGHTS


class MetaOrchestrator:
    def __init__(
        self,
        agent_scores: Dict[str, Dict[str, float]],
        weights: Dict[str, float] = None
    ):
        self.agent_scores = agent_scores
        self.weights = weights or AGENT_WEIGHTS
        self.transaction_ids = self._get_all_transaction_ids()
    
    def _get_all_transaction_ids(self) -> List[str]:
        all_ids = set()
        for agent_name, scores in self.agent_scores.items():
            all_ids.update(scores.keys())
        return list(all_ids)
    
    def compute_combined_scores(self) -> Dict[str, float]:
        combined = {}
        
        for txn_id in self.transaction_ids:
            weighted_sum = 0.0
            weight_total = 0.0
            
            for agent_name, scores in self.agent_scores.items():
                if txn_id in scores:
                    weight = self.weights.get(agent_name, 0.1)
                    weighted_sum += weight * scores[txn_id]
                    weight_total += weight
            
            if weight_total > 0:
                combined[txn_id] = weighted_sum / weight_total
            else:
                combined[txn_id] = 0.0
        
        return combined
    
    def calibrate_threshold(
        self,
        combined_scores: Dict[str, float],
        min_fraud_rate: float = 0.15,
        max_fraud_rate: float = 0.50
    ) -> float:
        sorted_scores = sorted(combined_scores.values(), reverse=True)
        
        n = len(sorted_scores)
        
        min_idx = int(n * min_fraud_rate)
        max_idx = int(n * max_fraud_rate)
        
        if min_idx >= n:
            min_idx = n - 1
        if max_idx >= n:
            max_idx = n - 1
        if min_idx > max_idx:
            min_idx = max_idx
        
        threshold = (sorted_scores[min_idx] + sorted_scores[max_idx]) / 2
        
        return threshold
    
    def classify(
        self,
        threshold: float = None,
        target_fraud_rate: float = None
    ) -> List[str]:
        combined_scores = self.compute_combined_scores()
        
        if threshold is None:
            if target_fraud_rate:
                threshold = self.calibrate_threshold(
                    combined_scores,
                    min_fraud_rate=target_fraud_rate
                )
            else:
                threshold = self.calibrate_threshold(combined_scores)
        
        fraud_transactions = [
            txn_id for txn_id, score in combined_scores.items()
            if score >= threshold
        ]
        
        return sorted(fraud_transactions, key=lambda x: combined_scores[x], reverse=True)
    
    def get_top_suspicious(
        self,
        n: int = 12,
        combined_scores: Dict[str, float] = None
    ) -> List[str]:
        if combined_scores is None:
            combined_scores = self.compute_combined_scores()
        
        sorted_transactions = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [txn_id for txn_id, _ in sorted_transactions[:n]]
    
    def get_score_summary(self) -> Dict[str, Any]:
        combined = self.compute_combined_scores()
        
        scores = list(combined.values())
        
        return {
            "num_transactions": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "threshold_recommended": self.calibrate_threshold(combined),
        }
