import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class DataLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        
    def load_transactions(self) -> List[Dict[str, Any]]:
        transactions = []
        with open(self.data_dir / "transactions.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                transactions.append({
                    "transaction_id": row["transaction_id"],
                    "sender_id": row["sender_id"],
                    "recipient_id": row["recipient_id"],
                    "transaction_type": row["transaction_type"],
                    "amount": float(row["amount"]) if row["amount"] else 0.0,
                    "location": row["location"] if row["location"] else None,
                    "payment_method": row["payment_method"] if row["payment_method"] else None,
                    "sender_iban": row["sender_iban"] if row["sender_iban"] else None,
                    "recipient_iban": row["recipient_iban"] if row["recipient_iban"] else None,
                    "balance_after": float(row["balance_after"]) if row["balance_after"] else 0.0,
                    "description": row["description"] if row["description"] else None,
                    "timestamp": datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
                })
        return transactions
    
    def load_users(self) -> Dict[str, Dict[str, Any]]:
        users = {}
        with open(self.data_dir / "users.json", "r") as f:
            user_list = json.load(f)
            for user in user_list:
                users[user["iban"]] = {
                    "first_name": user["first_name"],
                    "last_name": user["last_name"],
                    "birth_year": user["birth_year"],
                    "salary": user["salary"],
                    "job": user["job"],
                    "iban": user["iban"],
                    "residence": user["residence"],
                    "description": user.get("description", ""),
                }
        return users
    
    def load_locations(self) -> List[Dict[str, Any]]:
        locations = []
        with open(self.data_dir / "locations.json", "r") as f:
            loc_list = json.load(f)
            for loc in loc_list:
                locations.append({
                    "biotag": loc["biotag"],
                    "timestamp": datetime.fromisoformat(loc["timestamp"]) if loc["timestamp"] else None,
                    "lat": float(loc["lat"]) if loc["lat"] else 0.0,
                    "lng": float(loc["lng"]) if loc["lng"] else 0.0,
                    "city": loc.get("city"),
                })
        return locations
    
    def load_sms(self) -> List[Dict[str, str]]:
        with open(self.data_dir / "sms.json", "r") as f:
            return json.load(f)
    
    def load_emails(self) -> List[Dict[str, str]]:
        with open(self.data_dir / "mails.json", "r") as f:
            return json.load(f)
    
    def load_all(self) -> Dict[str, Any]:
        return {
            "transactions": self.load_transactions(),
            "users": self.load_users(),
            "locations": self.load_locations(),
            "sms": self.load_sms(),
            "emails": self.load_emails(),
        }
    
    def create_iban_to_user_map(self, users: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        iban_map = {}
        for iban, user in users.items():
            iban_map[iban] = user
        return iban_map
    
    def create_biotag_to_user_map(self, users: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        biotag_map = {}
        return biotag_map
