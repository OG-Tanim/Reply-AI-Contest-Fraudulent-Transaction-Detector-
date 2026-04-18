from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import OUTPUT_FILE, COST_MATRIX


class OutputGenerator:
    def __init__(
        self,
        fraud_transactions: List[str],
        combined_scores: Dict[str, float] = None,
        agent_scores: Dict[str, Dict[str, float]] = None
    ):
        self.fraud_transactions = fraud_transactions
        self.combined_scores = combined_scores or {}
        self.agent_scores = agent_scores or {}
    
    def write_output(self, output_path: Path = None) -> Path:
        output_path = output_path or OUTPUT_FILE
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for txn_id in self.fraud_transactions:
                f.write(f"{txn_id}\n")
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        n_transactions = len(self.combined_scores)
        n_fraud = len(self.fraud_transactions)
        
        fraud_rate = n_fraud / n_transactions if n_transactions > 0 else 0
        
        return {
            "total_transactions": n_transactions,
            "fraudulent_transactions": n_fraud,
            "fraud_detection_rate": fraud_rate,
            "output_file": str(OUTPUT_FILE),
        }
    
    def print_summary(self):
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("FRAUD DETECTION SUMMARY")
        print("=" * 60)
        print(f"Total transactions analyzed: {summary['total_transactions']}")
        print(f"Fraudulent transactions flagged: {summary['fraudulent_transactions']}")
        print(f"Fraud detection rate: {summary['fraud_detection_rate']:.1%}")
        print(f"Output file: {summary['output_file']}")
        print("=" * 60)
        
        if self.combined_scores:
            print("\nTop 10 Most Suspicious Transactions:")
            print("-" * 60)
            
            sorted_txns = sorted(
                self.combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for i, (txn_id, score) in enumerate(sorted_txns, 1):
                agent_breakdown = []
                for agent_name, scores in self.agent_scores.items():
                    if txn_id in scores:
                        agent_breakdown.append(f"{agent_name}:{scores[txn_id]:.2f}")
                
                print(f"{i}. Score: {score:.3f} | {txn_id[:20]}...")
                print(f"   Agents: {', '.join(agent_breakdown[:3])}")
