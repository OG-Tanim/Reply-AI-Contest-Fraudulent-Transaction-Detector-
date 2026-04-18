import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import re
from typing import List, Dict, Any, Optional, Set

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from src.data.loader import DataLoader
from src.data.feature_store import FeatureStore


load_dotenv()


class CommunicationsAgent:
    PHISHING_KEYWORDS = [
        "urgent", "verify", "account", "suspended", "click here",
        "confirm identity", "password", "security", "alert",
        "unusual activity", "immediately", "action required",
        "paypa1", "amaz0n", "bit.ly", "http://", "suspicious login"
    ]
    
    def __init__(self, data: Dict[str, Any], high_risk_threshold: float = 0.5):
        self.sms = data["sms"]
        self.emails = data["emails"]
        self.users = data["users"]
        self.high_risk_threshold = high_risk_threshold
        
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="gpt-4o-preview",
            temperature=0.1,
        )
        
        self._build_user_communication_map()
    
    def _build_user_communication_map(self):
        self.user_sms = {}
        self.user_emails = {}
        
        self.sms_by_keyword = []
        self.email_by_keyword = []
        
        for sms in self.sms:
            text = sms.get("sms", "")
            self._extract_sms_user(sms)
        
        for email in self.emails:
            text = email.get("mail", "")
            self._check_email_phishing(email)
    
    def _extract_sms_user(self, sms: Dict) -> Optional[str]:
        text = sms.get("sms", "")
        
        names = []
        for user in self.users.values():
            first = user.get("first_name", "")
            last = user.get("last_name", "")
            if first:
                names.append(first.lower())
            if last:
                names.append(last.lower())
        
        for name in names:
            if name in text.lower():
                return name
        
        return None
    
    def _check_email_phishing(self, email: Dict) -> bool:
        text = email.get("mail", "")
        text_lower = text.lower()
        
        phishing_count = 0
        for keyword in self.PHISHING_KEYWORDS:
            if keyword.lower() in text_lower:
                phishing_count += 1
        
        if phishing_count >= 2:
            self.email_by_keyword.append(email)
            return True
        
        suspicious_patterns = [
            r'paypa1',
            r'amaz0n',
            r'bit\.ly/\w+',
            r'http://(?!www\.)',
            r'verify.*account',
            r'confirm.*identity',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text_lower):
                self.email_by_keyword.append(email)
                return True
        
        return False
    
    def _classify_with_llm(self, text: str) -> float:
        prompt = f"""Analyze this communication for fraud/phishing indicators.

Communication:
{text[:2000]}

Respond with ONLY a number between 0.0 and 1.0 representing the fraud probability:
- 0.0 = definitely legitimate
- 0.5 = uncertain
- 1.0 = definitely phishing/fraud

Consider:
- Urgency language ("urgent", "immediately", "action required")
- Suspicious links or domains
- Requests for personal information
- Suspicious sender addresses
- Poor grammar or formatting
- Threat language ("account suspended", "verify or lose access")

Output only the number:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            match = re.search(r'0?\.\d+', response_text)
            if match:
                return float(match.group())
        except Exception as e:
            print(f"LLM error: {e}")
        
        return 0.5
    
    def get_communication_risk(self, user_name: str, sender_iban: Optional[str] = None) -> float:
        risk_scores = []
        
        user_lower = user_name.lower()
        
        for sms in self.sms:
            text = sms.get("sms", "")
            if user_lower in text.lower():
                score = self._classify_with_llm(text)
                risk_scores.append(score)
        
        for email in self.emails:
            text = email.get("mail", "")
            if user_lower in text.lower():
                score = self._classify_with_llm(text)
                risk_scores.append(score)
        
        for email in self.email_by_keyword:
            text = email.get("mail", "")
            score = self._classify_with_llm(text)
            risk_scores.append(score * 0.8)
        
        if not risk_scores:
            return 0.0
        
        return max(risk_scores)
    
    def score(self, transaction: Dict[str, Any], prefilter_score: float = 0.0) -> float:
        if prefilter_score < self.high_risk_threshold:
            return 0.0
        
        sender_id = transaction.get("sender_id", "")
        
        user_name = ""
        for user in self.users.values():
            if user.get("first_name"):
                user_name = user.get("first_name", "")
                break
        
        comm_risk = self.get_communication_risk(user_name)
        
        return comm_risk
    
    def score_high_risk_transactions(
        self,
        transactions: List[Dict[str, Any]],
        prefilter_scores: Dict[str, float]
    ) -> Dict[str, float]:
        results = {}
        
        high_risk_txns = [
            (txn_id, score) for txn_id, score in prefilter_scores.items()
            if score >= self.high_risk_threshold
        ]
        
        for txn in transactions:
            txn_id = txn["transaction_id"]
            if txn_id not in prefilter_scores:
                results[txn_id] = 0.0
                continue
            
            pre_score = prefilter_scores[txn_id]
            
            if pre_score < self.high_risk_threshold:
                results[txn_id] = 0.0
            else:
                results[txn_id] = self.score(txn, pre_score)
        
        return results
    
    def score_all(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        return {txn["transaction_id"]: 0.0 for txn in transactions}
