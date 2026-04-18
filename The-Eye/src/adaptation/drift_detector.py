import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import math


class DriftDetector:
    DDM_WARNING_THRESHOLD = 2.0
    DDM_DRIFT_THRESHOLD = 3.0
    DDM_MIN_SAMPLES = 30
    
    ADWIN_DELTA = 0.002
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.ddm_state = {
            "mean": 0.0,
            "variance": 0.0,
            "n": 0,
            "last_warning": None,
            "drift_detected": False,
        }
        self.adwin_windows = {}
        self.drift_history = []
        self.warning_history = []
    
    def add_sample(self, score: float) -> Tuple[bool, bool]:
        warning, drift = self._ddm_check(score)
        
        if warning:
            self.warning_history.append(self.ddm_state["n"])
        
        if drift:
            self.drift_history.append(self.ddm_state["n"])
            self.ddm_state["drift_detected"] = True
        
        return warning, drift
    
    def _ddm_check(self, score: float) -> Tuple[bool, bool]:
        n = self.ddm_state["n"]
        mean = self.ddm_state["mean"]
        variance = self.ddm_state["variance"]
        
        if n < 1:
            self.ddm_state["mean"] = score
            self.ddm_state["variance"] = 0.0
            self.ddm_state["n"] = 1
            return False, False
        
        new_n = n + 1
        new_mean = mean + (score - mean) / new_n
        
        if new_n > 1:
            new_variance = variance + (score - mean) ** 2 - (score - new_mean) ** 2 - variance / new_n
            new_variance = max(0.0, new_variance)
        else:
            new_variance = 0.0
        
        self.ddm_state["mean"] = new_mean
        self.ddm_state["variance"] = new_variance
        self.ddm_state["n"] = new_n
        
        if new_n < self.DDM_MIN_SAMPLES:
            return False, False
        
        std = math.sqrt(new_variance / new_n) if new_n > 1 else 1.0
        std = max(std, 1e-6)
        
        drift_level = abs(new_mean - 0.5) / std
        
        if drift_level > self.DDM_DRIFT_THRESHOLD:
            return True, True
        elif drift_level > self.DDM_WARNING_THRESHOLD:
            return True, False
        
        return False, False
    
    def reset(self):
        self.ddm_state = {
            "mean": 0.0,
            "variance": 0.0,
            "n": 0,
            "last_warning": None,
            "drift_detected": False,
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "n_samples": self.ddm_state["n"],
            "mean": self.ddm_state["mean"],
            "drift_detected": self.ddm_state["drift_detected"],
            "num_drifts": len(self.drift_history),
            "num_warnings": len(self.warning_history),
        }


class AgentDriftDetector:
    def __init__(self):
        self.agent_detectors: Dict[str, DriftDetector] = {}
        self.level_agent_scores: Dict[int, Dict[str, List[float]]] = {}
        self.current_level = 0
    
    def register_agent(self, agent_name: str):
        if agent_name not in self.agent_detectors:
            self.agent_detectors[agent_name] = DriftDetector()
    
    def add_agent_score(self, agent_name: str, score: float):
        if agent_name not in self.agent_detectors:
            self.register_agent(agent_name)
        
        self.agent_detectors[agent_name].add_sample(score)
    
    def add_level_scores(self, level: int, agent_scores: Dict[str, List[float]]):
        self.level_agent_scores[level] = agent_scores
        self.current_level = level
    
    def detect_agent_drift(self, agent_name: str) -> Dict[str, Any]:
        if agent_name not in self.agent_detectors:
            return {"drift": False, "warning": False}
        
        detector = self.agent_detectors[agent_name]
        status = detector.get_status()
        
        return {
            "drift": status["drift_detected"],
            "warning": len(status.get("warning_history", [])) > 0 and not status["drift_detected"],
            "num_drifts": status["num_drifts"],
            "current_mean": status["mean"],
        }
    
    def detect_all_drift(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        for agent_name in self.agent_detectors:
            results[agent_name] = self.detect_agent_drift(agent_name)
        return results
    
    def compare_levels(self, level_a: int, level_b: int) -> Dict[str, float]:
        if level_a not in self.level_agent_scores or level_b not in self.level_agent_scores:
            return {}
        
        comparison = {}
        agents_a = self.level_agent_scores[level_a]
        agents_b = self.level_agent_scores[level_b]
        
        for agent_name in agents_a:
            if agent_name in agents_b:
                scores_a_raw = agents_a[agent_name]
                scores_b_raw = agents_b[agent_name]
                
                if isinstance(scores_a_raw, dict):
                    scores_a = list(scores_a_raw.values())
                else:
                    scores_a = scores_a_raw
                    
                if isinstance(scores_b_raw, dict):
                    scores_b = list(scores_b_raw.values())
                else:
                    scores_b = scores_b_raw
                
                mean_a = sum(scores_a) / len(scores_a) if scores_a else 0
                mean_b = sum(scores_b) / len(scores_b) if scores_b else 0
                
                comparison[agent_name] = abs(mean_b - mean_a)
        
        return comparison
    
    def get_agents_needing_retrain(self, threshold: float = 0.2) -> List[str]:
        agents_to_retrain = []
        
        if self.current_level < 1:
            return agents_to_retrain
        
        prev_level = self.current_level - 1
        comparisons = self.compare_levels(prev_level, self.current_level)
        
        for agent_name, drift_magnitude in comparisons.items():
            if drift_magnitude > threshold:
                agents_to_retrain.append(agent_name)
        
        return agents_to_retrain
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "current_level": self.current_level,
            "num_agents": len(self.agent_detectors),
            "level_history": list(self.level_agent_scores.keys()),
            "all_drift_status": self.detect_all_drift(),
        }


class ADWINDetector:
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = deque(maxlen=1000)
        self.total = 0.0
        self.variance = 0.0
        self.n = 0
    
    def add(self, value: float) -> bool:
        self.window.append(value)
        self.n += 1
        self.total += value
        
        return self._detect_drift()
    
    def _detect_drift(self) -> bool:
        if len(self.window) < 50:
            return False
        
        n = len(self.window)
        
        for i in range(n // 2, n):
            window_left = list(self.window)[:i]
            window_right = list(self.window)[i:]
            
            n1 = len(window_left)
            n2 = len(window_right)
            
            if n1 < 10 or n2 < 10:
                continue
            
            mean1 = sum(window_left) / n1
            mean2 = sum(window_right) / n2
            
            epsilon_cut = self._compute_epsilon_cut(n1, n2)
            
            if abs(mean1 - mean2) > epsilon_cut:
                return True
        
        return False
    
    def _compute_epsilon_cut(self, n1: int, n2: int) -> float:
        m = 1.0 / (1.0 / n1 + 1.0 / n2)
        return math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / self.delta))
    
    def reset(self):
        self.window.clear()
        self.total = 0.0
        self.variance = 0.0
        self.n = 0
    
    @property
    def mean(self) -> float:
        if not self.window:
            return 0.0
        return self.total / len(self.window)
