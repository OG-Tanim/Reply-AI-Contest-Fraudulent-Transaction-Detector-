import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    TRAINING_DATA_DIR,
    EVALUATION_DATA_DIR,
    OUTPUT_FILE,
    AGENT_WEIGHTS,
    MIN_FRAUD_DETECTION_RATE,
    NUM_LEVELS,
    LEVEL_CONFIGS,
    DRIFT_CONFIG,
    MEMORY_CONFIG,
    MEMORY_BANK_FILE,
)
from src.data.loader import DataLoader
from src.data.feature_store import FeatureStore
from src.agents.transaction_anomaly import TransactionAnomalyAgent
from src.agents.behavioral_profiler import BehavioralProfilerAgent
from src.agents.geospatial import GeospatialAgent
from src.agents.graph_network import GraphNetworkAgent
from src.agents.temporal import TemporalAgent
from src.agents.communications import CommunicationsAgent
from src.orchestrator.meta_orchestrator import MetaOrchestrator
from src.output.generator import OutputGenerator
from src.adaptation.drift_detector import AgentDriftDetector
from src.adaptation.memory_bank import MemoryBank
from src.adaptation.adaptation_engine import AdaptationEngine, AgentRetrainer


load_dotenv()


langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)


def generate_session_id():
    team = os.getenv("TEAM_NAME", "reply-mirror").replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def load_and_process_data(data_dir: Path):
    print(f"Loading data from {data_dir}...")
    loader = DataLoader(data_dir)
    data = loader.load_all()
    
    print(f"  Transactions: {len(data['transactions'])}")
    print(f"  Users: {len(data['users'])}")
    print(f"  Locations: {len(data['locations'])}")
    print(f"  SMS: {len(data['sms'])}")
    print(f"  Emails: {len(data['emails'])}")
    
    return data


def train_agents_on_training_data(data: dict, memory_bank: MemoryBank = None):
    print("\nTraining agents on training data...")
    
    feature_store = FeatureStore(data)
    
    print("  Training Transaction Anomaly Agent...")
    txn_agent = TransactionAnomalyAgent(feature_store)
    
    print("  Training Behavioral Profiler Agent...")
    behavior_agent = BehavioralProfilerAgent(feature_store)
    
    print("  Training Geospatial Agent...")
    geo_agent = GeospatialAgent(feature_store)
    
    print("  Training Graph Network Agent...")
    graph_agent = GraphNetworkAgent(feature_store)
    
    print("  Training Temporal Agent...")
    temporal_agent = TemporalAgent(feature_store)
    
    print("  Training Communications Agent...")
    comm_agent = CommunicationsAgent(data)
    
    return {
        "feature_store": feature_store,
        "txn_agent": txn_agent,
        "behavior_agent": behavior_agent,
        "geo_agent": geo_agent,
        "graph_agent": graph_agent,
        "temporal_agent": temporal_agent,
        "comm_agent": comm_agent,
    }


def score_transactions(transactions: list, trained_agents: dict):
    print("\nScoring evaluation transactions...")
    
    feature_store = trained_agents["feature_store"]
    
    print("  Running Transaction Anomaly Agent...")
    txn_scores = trained_agents["txn_agent"].score_all(transactions)
    
    print("  Running Behavioral Profiler Agent...")
    behavior_scores = trained_agents["behavior_agent"].score_all(transactions)
    
    print("  Running Geospatial Agent...")
    geo_scores = trained_agents["geo_agent"].score_all(transactions)
    
    print("  Running Graph Network Agent...")
    graph_scores = trained_agents["graph_agent"].score_all(transactions)
    
    print("  Running Temporal Agent...")
    temporal_scores = trained_agents["temporal_agent"].score_all(transactions)
    
    print("  Running Communications Agent (high-risk only)...")
    comm_scores = trained_agents["comm_agent"].score_all(transactions)
    
    agent_scores = {
        "transaction_anomaly": txn_scores,
        "behavioral_profiler": behavior_scores,
        "geospatial": geo_scores,
        "graph_network": graph_scores,
        "temporal": temporal_scores,
        "communications": comm_scores,
    }
    
    return agent_scores


@observe()
def run_single_level(
    session_id: str,
    level: int,
    data_dir: Path,
    trained_agents: dict = None,
    adaptation_engine: AdaptationEngine = None,
    level_config: dict = None
):
    print(f"\n{'='*60}")
    print(f"LEVEL {level} - Session: {session_id}")
    print(f"{'='*60}")
    
    data = load_and_process_data(data_dir)
    transactions = data["transactions"]
    
    if trained_agents is None or adaptation_engine is None:
        trained_agents = train_agents_on_training_data(data, adaptation_engine.memory_bank if adaptation_engine else None)
    else:
        print("\nUsing pre-trained agents (with potential retraining)")
    
    agent_scores = score_transactions(transactions, trained_agents)
    
    weights = level_config.get("agent_weights", AGENT_WEIGHTS) if level_config else AGENT_WEIGHTS
    min_fraud_rate = level_config.get("min_fraud_rate", MIN_FRAUD_DETECTION_RATE) if level_config else MIN_FRAUD_DETECTION_RATE
    
    print("\nAggregating scores with Meta-Orchestrator...")
    orchestrator = MetaOrchestrator(agent_scores, weights)
    
    combined_scores = orchestrator.compute_combined_scores()
    
    print("\nCalibrating threshold...")
    threshold = orchestrator.calibrate_threshold(
        combined_scores,
        min_fraud_rate=min_fraud_rate
    )
    print(f"  Recommended threshold: {threshold:.3f}")
    
    fraud_transactions = orchestrator.classify(threshold=threshold)
    
    print(f"\nFlagged {len(fraud_transactions)} transactions as potentially fraudulent")
    
    result = {
        "level": level,
        "fraud_transactions": fraud_transactions,
        "combined_scores": combined_scores,
        "agent_scores": agent_scores,
        "trained_agents": trained_agents,
    }
    
    return result, data, orchestrator


def run_five_level_loop(session_id: str, training_data_dir: Path, eval_data_dir: Path):
    print(f"\n{'#'*60}")
    print(f"# REPLY MIRROR - 5 LEVEL FRAUD DETECTION LOOP")
    print(f"# Session: {session_id}")
    print(f"{'#'*60}")
    
    drift_detector = AgentDriftDetector()
    memory_bank = MemoryBank(
        similarity_threshold=MEMORY_CONFIG["similarity_threshold"],
        max_patterns_per_level=MEMORY_CONFIG["max_patterns_per_level"]
    )
    retrainer = AgentRetrainer()
    adaptation_engine = AdaptationEngine(
        drift_detector=drift_detector,
        memory_bank=memory_bank,
        retrainer=retrainer
    )
    
    if MEMORY_BANK_FILE.exists():
        try:
            memory_bank.load(MEMORY_BANK_FILE)
            print(f"\nLoaded existing memory bank with {len(memory_bank.patterns)} patterns")
        except Exception as e:
            print(f"Could not load memory bank: {e}")
    
    trained_agents = None
    
    all_results = []
    
    for level in range(1, NUM_LEVELS + 1):
        level_config = LEVEL_CONFIGS.get(level, LEVEL_CONFIGS[1])
        print(f"\n{'='*60}")
        print(f"PROCESSING LEVEL {level}: {level_config['name']}")
        print(f"{'='*60}")
        
        if level == 1:
            result, data, orchestrator = run_single_level(
                session_id, level, training_data_dir, trained_agents, adaptation_engine, level_config
            )
        else:
            result, data, orchestrator = run_single_level(
                session_id, level, training_data_dir, trained_agents, adaptation_engine, level_config
            )
        
        all_results.append(result)
        
        if level < NUM_LEVELS:
            print(f"\n{'='*60}")
            print(f"ADAPTATION PHASE - After Level {level}")
            print(f"{'='*60}")
            
            adaptation_result = adaptation_engine.process_level(
                level=level,
                data=data,
                agents=result["trained_agents"],
                agent_scores=result["agent_scores"],
                fraud_transactions=result["fraud_transactions"],
                combined_scores=result["combined_scores"]
            )
            
            trained_agents = adaptation_result.get("retrained_agents", result["trained_agents"])
            if not trained_agents:
                trained_agents = result["trained_agents"]
            
            print(f"\n  Drift detected: {adaptation_result['drift_detected']}")
            print(f"  Agents retrained: {adaptation_result['agents_to_retrain']}")
            print(f"  New patterns stored: {adaptation_result['new_patterns_added']}")
            for adaptation in adaptation_result.get("adaptations_applied", []):
                print(f"    - {adaptation}")
            
            try:
                memory_bank.save(MEMORY_BANK_FILE)
                print(f"\n  Memory bank saved to {MEMORY_BANK_FILE}")
            except Exception as e:
                print(f"  Could not save memory bank: {e}")
    
    print(f"\n{'#'*60}")
    print(f"# FINAL EVALUATION - Level {NUM_LEVELS}")
    print(f"{'#'*60}")
    
    final_level_config = LEVEL_CONFIGS[NUM_LEVELS]
    
    if trained_agents:
        print("\nRetraining agents on training data before final evaluation...")
        trained_agents = train_agents_on_training_data(
            load_and_process_data(training_data_dir),
            memory_bank
        )
    
    final_result, _, _ = run_single_level(
        session_id, NUM_LEVELS, eval_data_dir, trained_agents, adaptation_engine, final_level_config
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    
    output_gen = OutputGenerator(
        fraud_transactions=final_result["fraud_transactions"],
        combined_scores=final_result["combined_scores"],
        agent_scores=final_result["agent_scores"]
    )
    
    output_path = output_gen.write_output()
    print(f"\nOutput written to: {output_path}")
    
    output_gen.print_summary()
    
    print(f"\n{'='*60}")
    print(f"LEVEL SUMMARY")
    print(f"{'='*60}")
    for result in all_results:
        print(f"  Level {result['level']}: {len(result['fraud_transactions'])} fraud transactions flagged")
    
    print(f"\n  Total fraud patterns in memory bank: {len(memory_bank.patterns)}")
    print(f"  Memory bank saved to: {MEMORY_BANK_FILE}")
    
    print(f"\n{'='*60}")
    print(f"ADAPTATION STATUS")
    print(f"{'='*60}")
    print(f"  Drift detector status: {drift_detector.get_status()}")
    print(f"  Memory bank status: {memory_bank.get_status()}")
    print(f"  Retrainer status: {retrainer.get_status()}")
    
    return final_result


@observe()
def run_fraud_detection(session_id: str, data_dir: Path, level: int = 1):
    level_config = LEVEL_CONFIGS.get(level, LEVEL_CONFIGS[1])
    result, data, orchestrator = run_single_level(
        session_id, level, data_dir, None, None, level_config
    )
    
    output_gen = OutputGenerator(
        fraud_transactions=result["fraud_transactions"],
        combined_scores=result["combined_scores"],
        agent_scores=result["agent_scores"]
    )
    
    output_path = output_gen.write_output()
    print(f"\nOutput written to: {output_path}")
    
    output_gen.print_summary()
    
    return result


def main():
    session_id = generate_session_id()
    
    mode = None
    for arg in sys.argv:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1]
        if arg.startswith("--level="):
            try:
                level = int(arg.split("=")[1])
            except:
                level = 1
    
    if "--train" in sys.argv:
        mode = "train"
    elif "--train-full" in sys.argv:
        mode = "train_full"
    elif "--full" in sys.argv:
        mode = "full"
    elif "--level" in sys.argv:
        mode = "single"
    else:
        mode = "eval"
    
    try:
        if mode == "full":
            print("Running FULL 5-level loop with adaptation...")
            run_five_level_loop(session_id, TRAINING_DATA_DIR, EVALUATION_DATA_DIR)
        elif mode == "train_full":
            print("Running 5-level loop on TRAINING data (no final evaluation)...")
            run_five_level_loop(session_id, TRAINING_DATA_DIR, TRAINING_DATA_DIR)
        elif mode == "train":
            print("Running on TRAINING data (Level 1)...")
            run_fraud_detection(session_id, TRAINING_DATA_DIR, level=1)
        elif mode == "single":
            print(f"Running single level {level}...")
            run_fraud_detection(session_id, EVALUATION_DATA_DIR, level=level)
        else:
            print("Running on EVALUATION data (Level 1)...")
            run_fraud_detection(session_id, EVALUATION_DATA_DIR, level=1)
        
        langfuse_client.flush()
        
        print(f"\n{session_id}")
        print(f"Traces sent to Langfuse dashboard (may take a few minutes to appear)")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
