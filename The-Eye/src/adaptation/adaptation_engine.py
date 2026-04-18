import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict
import numpy as np

from src.data.feature_store import FeatureStore


class AgentRetrainer:
    def __init__(self):
        self.retrain_history: List[Dict[str, Any]] = []
        self.agent_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def should_retrain(
        self,
        agent_name: str,
        drift_detector,
        threshold: float = 0.15
    ) -> bool:
        drift_status = drift_detector.detect_agent_drift(agent_name)
        
        if drift_status["drift"]:
            return True
        
        if drift_status.get("current_mean", 0.5) < 0.3 or drift_status.get("current_mean", 0.5) > 0.7:
            if drift_status["warning"]:
                return True
        
        return False
    
    def retrain_transaction_anomaly_agent(
        self,
        agent,
        data: Dict[str, Any],
        memory_bank=None,
        fewshot_examples: Optional[List[Dict]] = None
    ):
        print(f"  Retraining Transaction Anomaly Agent...")
        
        transactions = data["transactions"]
        feature_store = FeatureStore(data)
        
        agent.model = None
        agent.feature_store = feature_store
        agent._train()
        
        self._record_retrain(agent, "transaction_anomaly", "full")
        
        return agent
    
    def retrain_behavioral_profiler(
        self,
        agent,
        data: Dict[str, Any],
        memory_bank=None,
        fewshot_examples: Optional[List[Dict]] = None
    ):
        print(f"  Retraining Behavioral Profiler Agent...")
        
        feature_store = FeatureStore(data)
        
        agent.feature_store = feature_store
        agent.user_profiles = {}
        agent._build_profiles()
        
        self._record_retrain(agent, "behavioral_profiler", "incremental")
        
        return agent
    
    def retrain_geospatial_agent(
        self,
        agent,
        data: Dict[str, Any],
        memory_bank=None,
        fewshot_examples: Optional[List[Dict]] = None
    ):
        print(f"  Retraining Geospatial Agent...")
        
        feature_store = FeatureStore(data)
        
        agent.feature_store = feature_store
        agent.location_profiles = {}
        agent._build_location_profiles()
        
        self._record_retrain(agent, "geospatial", "incremental")
        
        return agent
    
    def retrain_graph_network_agent(
        self,
        agent,
        data: Dict[str, Any],
        memory_bank=None,
        fewshot_examples: Optional[List[Dict]] = None
    ):
        print(f"  Retraining Graph Network Agent...")
        
        feature_store = FeatureStore(data)
        
        agent.feature_store = feature_store
        agent.graph = None
        agent.node_metrics = {}
        agent._build_graph()
        agent._compute_metrics()
        
        self._record_retrain(agent, "graph_network", "full")
        
        return agent
    
    def retrain_temporal_agent(
        self,
        agent,
        data: Dict[str, Any],
        memory_bank=None,
        fewshot_examples: Optional[List[Dict]] = None
    ):
        print(f"  Retraining Temporal Agent...")
        
        feature_store = FeatureStore(data)
        
        agent.feature_store = feature_store
        agent.hour_distribution = defaultdict(int)
        agent.day_distribution = defaultdict(int)
        agent.month_distribution = defaultdict(int)
        agent.user_hour_patterns = defaultdict(lambda: defaultdict(int))
        agent.user_day_patterns = defaultdict(lambda: defaultdict(int))
        agent._build_temporal_patterns()
        
        self._record_retrain(agent, "temporal", "incremental")
        
        return agent
    
    def retrain_communications_agent(
        self,
        agent,
        data: Dict[str, Any],
        memory_bank=None,
        fewshot_examples: Optional[List[Dict]] = None
    ):
        print(f"  Retraining Communications Agent (updating keyword patterns)...")
        
        agent.sms = data["sms"]
        agent.emails = data["emails"]
        agent.users = data["users"]
        agent._build_user_communication_map()
        
        self._record_retrain(agent, "communications", "keyword_update")
        
        return agent
    
    def retrain_agents(
        self,
        agents: Dict[str, Any],
        data: Dict[str, Any],
        agents_to_retrain: List[str],
        memory_bank=None
    ) -> Dict[str, Any]:
        retrained = {}
        
        retrain_map = {
            "transaction_anomaly": self.retrain_transaction_anomaly_agent,
            "behavioral_profiler": self.retrain_behavioral_profiler,
            "geospatial": self.retrain_geospatial_agent,
            "graph_network": self.retrain_graph_network_agent,
            "temporal": self.retrain_temporal_agent,
            "communications": self.retrain_communications_agent,
        }
        
        for agent_name in agents_to_retrain:
            if agent_name in retrain_map and agent_name in agents:
                fewshot = None
                if memory_bank:
                    fewshot = memory_bank.get_fewshot_examples(
                        pattern_type=self._get_pattern_type(agent_name),
                        current_level=len(self.retrain_history),
                    )
                
                retrained[agent_name] = retrain_map[agent_name](
                    agents[agent_name],
                    data,
                    memory_bank,
                    fewshot
                )
        
        return retrained
    
    def _get_pattern_type(self, agent_name: str) -> str:
        mapping = {
            "transaction_anomaly": "amount_anomaly",
            "behavioral_profiler": "behavioral_drift",
            "geospatial": "geographic_anomaly",
            "graph_network": "network_ring",
            "temporal": "temporal_burst",
            "communications": "communication_phishing",
        }
        return mapping.get(agent_name, "unknown")
    
    def _record_retrain(self, agent, agent_name: str, retrain_type: str):
        self.retrain_history.append({
            "agent_name": agent_name,
            "retrain_type": retrain_type,
            "timestamp": str(np.datetime64('now')),
        })
        
        self.agent_states[agent_name] = {
            "last_retrain_type": retrain_type,
            "retrain_count": self.agent_states[agent_name].get("retrain_count", 0) + 1,
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "total_retrains": len(self.retrain_history),
            "agent_states": dict(self.agent_states),
            "recent_retrains": self.retrain_history[-5:] if self.retrain_history else [],
        }


class AdaptationEngine:
    def __init__(
        self,
        drift_detector=None,
        memory_bank=None,
        retrainer=None
    ):
        self.drift_detector = drift_detector
        self.memory_bank = memory_bank
        self.retrainer = retrainer or AgentRetrainer()
        self.level_configs: Dict[int, Dict[str, Any]] = {}
        self.current_level = 0
    
    def register_drift_detector(self, drift_detector):
        self.drift_detector = drift_detector
    
    def register_memory_bank(self, memory_bank):
        self.memory_bank = memory_bank
    
    def process_level(
        self,
        level: int,
        data: Dict[str, Any],
        agents: Dict[str, Any],
        agent_scores: Dict[str, Dict[str, float]],
        fraud_transactions: List[str],
        combined_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        self.current_level = level
        
        print(f"\n{'='*60}")
        print(f"ADAPTATION ENGINE - Level {level}")
        print(f"{'='*60}")
        
        result = {
            "level": level,
            "drift_detected": False,
            "agents_to_retrain": [],
            "new_patterns_added": 0,
            "adaptations_applied": [],
        }
        
        if self.drift_detector:
            for agent_name, scores in agent_scores.items():
                for txn_id, score in scores.items():
                    self.drift_detector.add_agent_score(agent_name, score)
            
            self.drift_detector.add_level_scores(level, agent_scores)
            
            drift_status = self.drift_detector.detect_all_drift()
            
            for agent_name, status in drift_status.items():
                if status["drift"]:
                    result["drift_detected"] = True
                    result["agents_to_retrain"].append(agent_name)
                    result["adaptations_applied"].append(f"Drift detected in {agent_name}")
        
        agents_needing_retrain = []
        if self.drift_detector:
            agents_needing_retrain = self.drift_detector.get_agents_needing_retrain()
            for agent_name in agents_needing_retrain:
                if agent_name not in result["agents_to_retrain"]:
                    result["agents_to_retrain"].append(agent_name)
                    result["adaptations_applied"].append(f"Level comparison drift in {agent_name}")
        
        if result["agents_to_retrain"]:
            print(f"\n  Retraining agents: {result['agents_to_retrain']}")
            agents = self.retrainer.retrain_agents(
                agents,
                data,
                result["agents_to_retrain"],
                self.memory_bank
            )
        
        if self.memory_bank and fraud_transactions:
            patterns_added = self._extract_and_store_patterns(
                level,
                fraud_transactions,
                combined_scores,
                agent_scores,
                data
            )
            result["new_patterns_added"] = patterns_added
            print(f"  Stored {patterns_added} fraud patterns")
        
        if self.memory_bank:
            self.memory_bank.build_level_summary(level)
        
        return result
    
    def _extract_and_store_patterns(
        self,
        level: int,
        fraud_transactions: List[str],
        combined_scores: Dict[str, float],
        agent_scores: Dict[str, Dict[str, float]],
        data: Dict[str, Any]
    ) -> int:
        if not self.memory_bank:
            return 0
        
        feature_store = FeatureStore(data)
        transaction_map = {t["transaction_id"]: t for t in data["transactions"]}
        
        patterns_added = 0
        
        txn_features = []
        for txn_id in fraud_transactions:
            if txn_id in transaction_map:
                txn = transaction_map[txn_id]
                features = feature_store.extract_features(txn)
                
                txn_agent_scores = {
                    agent: scores.get(txn_id, 0)
                    for agent, scores in agent_scores.items()
                }
                
                pattern_type = self._classify_pattern_type(txn_agent_scores)
                
                self.memory_bank.add_pattern(
                    level=level,
                    pattern_type=pattern_type,
                    features={
                        "amount_zscore": features.get("amount_zscore", 0),
                        "amount_deviation": features.get("amount_deviation", 0),
                        "recipient_novelty": features.get("recipient_novelty", 0),
                        "time_deviation": features.get("time_deviation", 0),
                    },
                    agent_scores=txn_agent_scores,
                    transaction_ids=[txn_id],
                    metadata={"combined_score": combined_scores.get(txn_id, 0)}
                )
                patterns_added += 1
        
        return patterns_added
    
    def _classify_pattern_type(self, agent_scores: Dict[str, float]) -> str:
        if not agent_scores:
            return "unknown"
        
        max_agent = max(agent_scores.items(), key=lambda x: x[1])
        
        mapping = {
            "transaction_anomaly": "amount_anomaly",
            "behavioral_profiler": "behavioral_drift",
            "geospatial": "geographic_anomaly",
            "graph_network": "network_ring",
            "temporal": "temporal_burst",
            "communications": "communication_phishing",
        }
        
        return mapping.get(max_agent[0], "unknown")
    
    def get_level_config(self, level: int) -> Dict[str, Any]:
        if level not in self.level_configs:
            self.level_configs[level] = self._default_config_for_level(level)
        return self.level_configs[level]
    
    def _default_config_for_level(self, level: int) -> Dict[str, Any]:
        base_threshold = 0.15
        threshold_increase = 0.02 * level
        
        return {
            "threshold": base_threshold + threshold_increase,
            "agent_weights": {
                "transaction_anomaly": 0.20 - 0.02 * level if level > 0 else 0.20,
                "behavioral_profiler": 0.20,
                "geospatial": 0.15 + 0.01 * level if level > 0 else 0.15,
                "graph_network": 0.15,
                "temporal": 0.15,
                "communications": 0.15 + 0.01 * level if level > 0 else 0.15,
            },
            "retrain_threshold": 0.15 + 0.02 * level,
        }
    
    def update_level_config(self, level: int, config: Dict[str, Any]):
        self.level_configs[level] = config
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "current_level": self.current_level,
            "level_configs": list(self.level_configs.keys()),
            "drift_detector": "active" if self.drift_detector else "inactive",
            "memory_bank": "active" if self.memory_bank else "inactive",
            "retrainer": self.retrainer.get_status(),
        }
