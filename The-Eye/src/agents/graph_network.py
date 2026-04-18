import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import networkx as nx
from typing import List, Dict, Any, Set
from collections import defaultdict

from src.data.feature_store import FeatureStore


class GraphNetworkAgent:
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.graph = None
        self.node_metrics = {}
        self._build_graph()
        self._compute_metrics()
    
    def _build_graph(self):
        self.graph = nx.DiGraph()
        
        for txn in self.feature_store.transactions:
            sender = txn["sender_id"]
            recipient = txn["recipient_id"]
            amount = txn["amount"]
            
            if self.graph.has_edge(sender, recipient):
                self.graph[sender][recipient]["weight"] += amount
                self.graph[sender][recipient]["count"] += 1
            else:
                self.graph.add_edge(sender, recipient, weight=amount, count=1)
    
    def _compute_metrics(self):
        if len(self.graph.nodes) == 0:
            return
        
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
            for node, score in pagerank.items():
                if node not in self.node_metrics:
                    self.node_metrics[node] = {}
                self.node_metrics[node]["pagerank"] = score
        except:
            pass
        
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        for node in self.graph.nodes:
            if node not in self.node_metrics:
                self.node_metrics[node] = {}
            self.node_metrics[node]["in_degree"] = in_degrees.get(node, 0)
            self.node_metrics[node]["out_degree"] = out_degrees.get(node, 0)
        
        try:
            betweenness = nx.betweenness_centrality(self.graph)
            for node, score in betweenness.items():
                if node not in self.node_metrics:
                    self.node_metrics[node] = {}
                self.node_metrics[node]["betweenness"] = score
        except:
            pass
        
        try:
            cycles = list(nx.simple_cycles(self.graph))
            self.cycle_nodes = set()
            for cycle in cycles:
                for node in cycle:
                    self.cycle_nodes.add(node)
        except:
            self.cycle_nodes = set()
    
    def _detect_rings(self, sender: str, recipient: str) -> bool:
        if not self.graph.has_node(recipient):
            return False
        
        try:
            if recipient in self.cycle_nodes:
                return True
            
            predecessors = set(self.graph.predecessors(recipient))
            successors = set(self.graph.successors(recipient))
            
            if sender in successors and len(predecessors) > 2:
                return True
        except:
            pass
        
        return False
    
    def _detect_money_mule_pattern(self, sender: str, recipient: str) -> float:
        if sender not in self.node_metrics or recipient not in self.node_metrics:
            return 0.0
        
        sender_out = self.node_metrics[sender]["out_degree"]
        sender_in = self.node_metrics[sender]["in_degree"]
        recipient_out = self.node_metrics[recipient]["out_degree"]
        
        if sender_out > 5 and sender_in == 0:
            return 0.8
        
        if sender_out > 3 and recipient_out > 3:
            return 0.6
        
        return 0.0
    
    def score(self, transaction: Dict[str, Any]) -> float:
        sender = transaction["sender_id"]
        recipient = transaction["recipient_id"]
        
        scores = []
        
        if self._detect_rings(sender, recipient):
            scores.append(0.7)
        
        mule_score = self._detect_money_mule_pattern(sender, recipient)
        if mule_score > 0:
            scores.append(mule_score)
        
        if sender in self.node_metrics:
            pagerank = self.node_metrics[sender].get("pagerank", 0)
            if pagerank > 0.1:
                scores.append(0.3 * pagerank / 0.1)
        
        if recipient in self.node_metrics:
            out_deg = self.node_metrics[recipient].get("out_degree", 0)
            if out_deg > 10:
                scores.append(0.4)
            elif out_deg > 5:
                scores.append(0.2)
        
        if sender not in self.node_metrics:
            scores.append(0.3)
        
        if recipient == "":
            scores.append(0.5)
        
        return min(1.0, max(scores) if scores else 0.0)
    
    def score_all(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            txn["transaction_id"]: self.score(txn)
            for txn in transactions
        }
