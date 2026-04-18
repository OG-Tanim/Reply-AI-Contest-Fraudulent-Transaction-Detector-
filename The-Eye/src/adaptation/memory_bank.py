import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import json
from datetime import datetime
import numpy as np


class FraudPattern:
    def __init__(
        self,
        pattern_id: str,
        level: int,
        pattern_type: str,
        features: Dict[str, float],
        agent_scores: Dict[str, float],
        transaction_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.pattern_id = pattern_id
        self.level = level
        self.pattern_type = pattern_type
        self.features = features
        self.agent_scores = agent_scores
        self.transaction_ids = transaction_ids
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "level": self.level,
            "pattern_type": self.pattern_type,
            "features": self.features,
            "agent_scores": self.agent_scores,
            "transaction_ids": self.transaction_ids,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FraudPattern":
        pattern = cls(
            pattern_id=data["pattern_id"],
            level=data["level"],
            pattern_type=data["pattern_type"],
            features=data["features"],
            agent_scores=data["agent_scores"],
            transaction_ids=data["transaction_ids"],
            metadata=data.get("metadata", {}),
        )
        pattern.access_count = data.get("access_count", 0)
        if "created_at" in data:
            pattern.created_at = datetime.fromisoformat(data["created_at"])
        return pattern
    
    def compute_similarity(self, other: "FraudPattern") -> float:
        if self.pattern_type != other.pattern_type:
            return 0.0
        
        feature_similarity = self._cosine_similarity(
            list(self.features.values()),
            list(other.features.values())
        )
        
        agent_similarity = self._cosine_similarity(
            list(self.agent_scores.values()),
            list(other.agent_scores.values())
        )
        
        return (feature_similarity + agent_similarity) / 2.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2) or len(vec1) == 0:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def mark_accessed(self):
        self.access_count += 1
        self.last_accessed = datetime.now()


import math


class MemoryBank:
    def __init__(self, similarity_threshold: float = 0.7, max_patterns_per_level: int = 50):
        self.patterns: List[FraudPattern] = []
        self.patterns_by_level: Dict[int, List[FraudPattern]] = defaultdict(list)
        self.patterns_by_type: Dict[str, List[FraudPattern]] = defaultdict(list)
        self.similarity_threshold = similarity_threshold
        self.max_patterns_per_level = max_patterns_per_level
        self.level_summaries: Dict[int, Dict[str, Any]] = {}
        self.fewshot_examples: Dict[str, List[FraudPattern]] = defaultdict(list)
    
    def add_pattern(
        self,
        level: int,
        pattern_type: str,
        features: Dict[str, float],
        agent_scores: Dict[str, float],
        transaction_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        pattern_id = f"pattern_l{level}_{pattern_type}_{len(self.patterns)}"
        
        existing = self._find_similar_pattern(level, pattern_type, features, agent_scores)
        if existing and existing.compute_similarity(
            FraudPattern(pattern_id, level, pattern_type, features, agent_scores, transaction_ids)
        ) > 0.9:
            existing.transaction_ids.extend(transaction_ids)
            return existing.pattern_id
        
        pattern = FraudPattern(
            pattern_id=pattern_id,
            level=level,
            pattern_type=pattern_type,
            features=features,
            agent_scores=agent_scores,
            transaction_ids=transaction_ids,
            metadata=metadata,
        )
        
        self.patterns.append(pattern)
        self.patterns_by_level[level].append(pattern)
        self.patterns_by_type[pattern_type].append(pattern)
        
        self._prune_level_patterns(level)
        
        return pattern_id
    
    def _find_similar_pattern(
        self,
        level: int,
        pattern_type: str,
        features: Dict[str, float],
        agent_scores: Dict[str, float]
    ) -> Optional[FraudPattern]:
        temp_pattern = FraudPattern(
            pattern_id="temp",
            level=level,
            pattern_type=pattern_type,
            features=features,
            agent_scores=agent_scores,
            transaction_ids=[],
        )
        
        candidates = self.patterns_by_type.get(pattern_type, [])
        if level in self.patterns_by_level:
            candidates = [p for p in candidates if p.level == level]
        
        best_match = None
        best_similarity = 0.0
        
        for pattern in candidates:
            similarity = temp_pattern.compute_similarity(pattern)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern
        
        if best_similarity > self.similarity_threshold:
            return best_match
        
        return None
    
    def _prune_level_patterns(self, level: int):
        patterns = self.patterns_by_level.get(level, [])
        if len(patterns) > self.max_patterns_per_level:
            patterns.sort(key=lambda p: p.access_count, reverse=True)
            patterns[:] = patterns[:self.max_patterns_per_level]
    
    def find_similar_patterns(
        self,
        pattern_type: str,
        features: Dict[str, float],
        agent_scores: Dict[str, float],
        max_results: int = 5,
        exclude_levels: Optional[List[int]] = None
    ) -> List[Tuple[FraudPattern, float]]:
        temp_pattern = FraudPattern(
            pattern_id="temp",
            level=-1,
            pattern_type=pattern_type,
            features=features,
            agent_scores=agent_scores,
            transaction_ids=[],
        )
        
        candidates = self.patterns_by_type.get(pattern_type, [])
        
        if exclude_levels:
            candidates = [p for p in candidates if p.level not in exclude_levels]
        
        similarities = []
        for pattern in candidates:
            sim = temp_pattern.compute_similarity(pattern)
            if sim > self.similarity_threshold:
                similarities.append((pattern, sim))
                pattern.mark_accessed()
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:max_results]
    
    def get_fewshot_examples(
        self,
        pattern_type: str,
        current_level: int,
        max_examples: int = 3
    ) -> List[Dict[str, Any]]:
        examples = []
        
        for level in range(current_level):
            patterns = self.patterns_by_level.get(level, [])
            type_patterns = [p for p in patterns if p.pattern_type == pattern_type]
            
            type_patterns.sort(key=lambda p: (p.access_count, len(p.transaction_ids)), reverse=True)
            
            for pattern in type_patterns[:max_examples]:
                examples.append({
                    "level": pattern.level,
                    "features": pattern.features,
                    "agent_scores": pattern.agent_scores,
                    "num_transactions": len(pattern.transaction_ids),
                    "similarity_context": pattern.compute_similarity(pattern),
                })
        
        return examples[:max_examples]
    
    def build_level_summary(self, level: int):
        patterns = self.patterns_by_level.get(level, [])
        
        if not patterns:
            self.level_summaries[level] = {
                "num_patterns": 0,
                "pattern_types": [],
                "avg_agent_scores": {},
            }
            return
        
        pattern_types = list(set(p.pattern_type for p in patterns))
        
        agent_names = set()
        for p in patterns:
            agent_names.update(p.agent_scores.keys())
        
        avg_scores = {}
        for agent in agent_names:
            scores = [p.agent_scores.get(agent, 0) for p in patterns]
            avg_scores[agent] = sum(scores) / len(scores) if scores else 0
        
        self.level_summaries[level] = {
            "num_patterns": len(patterns),
            "pattern_types": pattern_types,
            "avg_agent_scores": avg_scores,
            "total_transactions": sum(len(p.transaction_ids) for p in patterns),
        }
    
    def suggest_retraining(
        self,
        current_level: int,
        new_features: Dict[str, float],
        new_agent_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        suggestions = {
            "should_retrain": False,
            "reason": "",
            "similar_past_patterns": [],
            "retrain_agents": [],
        }
        
        if current_level < 1:
            return suggestions
        
        for pattern_type in ["amount_anomaly", "behavioral_drift", "geographic_anomaly", "network_ring", "temporal_burst"]:
            matches = self.find_similar_patterns(
                pattern_type,
                new_features,
                new_agent_scores,
                exclude_levels=[current_level]
            )
            
            if matches:
                suggestions["similar_past_patterns"].extend(
                    {"type": pattern_type, "pattern": p.pattern_id, "similarity": sim}
                    for p, sim in matches
                )
        
        if len(suggestions["similar_past_patterns"]) >= 3:
            suggestions["should_retrain"] = True
            suggestions["reason"] = "Multiple similar patterns found from past levels"
            suggestions["retrain_agents"] = ["transaction_anomaly", "behavioral_profiler"]
        
        return suggestions
    
    def save(self, filepath: Path):
        data = {
            "patterns": [p.to_dict() for p in self.patterns],
            "level_summaries": self.level_summaries,
            "similarity_threshold": self.similarity_threshold,
            "max_patterns_per_level": self.max_patterns_per_level,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, filepath: Path):
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.patterns = [FraudPattern.from_dict(p) for p in data.get("patterns", [])]
        self.level_summaries = data.get("level_summaries", {})
        self.similarity_threshold = data.get("similarity_threshold", 0.7)
        self.max_patterns_per_level = data.get("max_patterns_per_level", 50)
        
        self.patterns_by_level.clear()
        self.patterns_by_type.clear()
        
        for pattern in self.patterns:
            self.patterns_by_level[pattern.level].append(pattern)
            self.patterns_by_type[pattern.pattern_type].append(pattern)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "total_patterns": len(self.patterns),
            "levels": list(self.level_summaries.keys()),
            "pattern_types": list(self.patterns_by_type.keys()),
            "patterns_per_level": {level: len(patterns) for level, patterns in self.patterns_by_level.items()},
        }
