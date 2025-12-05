import json
import time
import logging
import tqdm
import os
from src.elasticsearch_client import ElasticsearchClient
from src.simulation_client import SimulationClient
from src.logger import setup_logger, log_interaction
from dotenv import load_dotenv

from src.features import FeatureExtractor
from src.ranker import PersonalizedRanker
from src.mf_ranker import MFRanker
from src.bpr_ranker import BPRRanker
from src.linucb_ranker import LinUCBRanker

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = "data/articles.jsonl"
XGBOOST_MODEL_FILE = "models/ranker.model"
MF_MODEL_FILE = "models/mf_model.pkl"
BPR_MODEL_FILE = "models/bpr_model.pkl"
LINUCB_MODEL_FILE = "models/linucb_model.pkl"

# Configuration
NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS", 500)) 
RANKER_TYPE = os.getenv("RANKER_TYPE", "xgboost") 

def load_articles(file_path):
    """Generates articles from the JSONL file."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    articles = []
    with open(file_path, 'r') as f:
        for line in f:
            articles.append(json.loads(line))
    return articles

import hashlib

class ExperimentManager:
    def __init__(self, salt="ire_project"):
        self.salt = salt

    def get_group(self, user_id):
        """Deterministically assigns user to a group based on hash."""
        hash_input = f"{self.salt}_{user_id}".encode('utf-8')
        hash_val = int(hashlib.sha256(hash_input).hexdigest(), 16)
        return "treatment" if hash_val % 2 == 0 else "control"

def main():
    es_client = ElasticsearchClient()
    sim_client = SimulationClient()
    interaction_logger = setup_logger()
    experiment_manager = ExperimentManager()

    # 1. Setup & Indexing
    logger.info("Checking Elasticsearch index...")
    es_client.create_index()
    
    # Load articles for feature extraction
    logger.info("Loading articles into memory...")
    articles = load_articles(DATA_FILE)
    if not articles:
        return

    # Check if we need to index
    try:
        count = int(es_client.es.count(index=es_client.index_name)['count'])
        if count == 0:
            logger.info(f"Index is empty. Loading articles from {DATA_FILE}...")
            es_client.index_documents(articles)
        else:
            logger.info(f"Index already contains {count} documents.")
    except Exception as e:
        logger.error(f"Error checking/indexing documents: {e}")
        return

    # 2. Initialize Ranker
    feature_extractor = FeatureExtractor()
    feature_extractor.load_article_cache(articles)
    
    ranker = None
    if RANKER_TYPE == "xgboost":
        logger.info("Initializing XGBoost Ranker...")
        ranker = PersonalizedRanker(feature_extractor, model_path=XGBOOST_MODEL_FILE)
    elif RANKER_TYPE == "mf":
        logger.info("Initializing Matrix Factorization Ranker...")
        ranker = MFRanker(model_path=MF_MODEL_FILE)
    elif RANKER_TYPE == "bpr":
        logger.info("Initializing BPR Ranker...")
        ranker = BPRRanker(model_path=BPR_MODEL_FILE)
    elif RANKER_TYPE == "linucb":
        logger.info("Initializing LinUCB Ranker (Online Learning)...")
        ranker = LinUCBRanker(model_path=LINUCB_MODEL_FILE, alpha=0.5)
        ranker.load_article_cache(articles) # LinUCB needs direct article access
    elif RANKER_TYPE == "baseline":
        logger.info("Running in Baseline mode (No Personalized Ranker)...")
        ranker = None
    else:
        logger.error(f"Unknown RANKER_TYPE: {RANKER_TYPE}")
        return

    # 3. Interaction Loop
    logger.info(f"Starting A/B Experiment loop with {RANKER_TYPE}...")
    for i in tqdm.tqdm(range(NUM_ITERATIONS), desc="Processing Queries"):
        # Fetch Query
        query_data = sim_client.get_query()
        if not query_data:
            logger.warning("Failed to get query. Retrying...")
            time.sleep(1)
            continue
        
        user_id = query_data['user_id']
        query_id = query_data['query_id']
        query_text = query_data['query_text']
        
        # Assign Group
        group = experiment_manager.get_group(user_id)
        
        # logger.info(f"Iteration {i+1}: Query='{query_text}' (User: {user_id}, Group: {group})")

        # Get Candidates (Elasticsearch Default)
        # Fetch more candidates for re-ranking (e.g., 100) to increase recall
        candidate_ids = es_client.search(query_text, size=100)
        
        if group == "treatment" and ranker:
            # Re-rank with exploration (epsilon=0.1)
            # Note: LinUCB handles exploration internally via UCB, so epsilon might be redundant or additive.
            # We'll pass it anyway, ranker can ignore it.
            ranked_ids = ranker.rank(user_id, query_text, candidate_ids, epsilon=0.1)
            final_ranking = ranked_ids[:10]
        elif RANKER_TYPE == "baseline":
             # Baseline Mode (Data Collection):
             # Pure Exploration: Shuffle the top 100 candidates to find ANY relevant items.
             # This helps break the "cold start" by gathering diverse clicks.
             import random
             shuffled_candidates = list(candidate_ids)
             random.shuffle(shuffled_candidates)
             final_ranking = shuffled_candidates[:10]
        else:
            # Control Group (during experiment): Just take top 10 from ES (BM25)
            final_ranking = candidate_ids[:10]
        
        # Send to Simulation
        response = sim_client.post_ranklist(user_id, query_id, final_ranking)
        if not response:
            logger.warning("Failed to get actions. Skipping...")
            continue
            
        actions = response['actions']
        
        # Log Interaction
        interaction = {
            "user_id": user_id,
            "query_id": query_id,
            "query_text": query_text,
            "ranked_article_ids": final_ranking,
            "actions": actions,
            "experiment_group": group,
            "ranker_type": RANKER_TYPE
        }
        log_interaction(interaction_logger, interaction)
        
        # Update User Profile (Always update, regardless of group, to keep history fresh)
        # Only needed for XGBoost feature extraction, but good to keep consistent
        for j, aid in enumerate(final_ranking):
            article = feature_extractor.article_cache.get(aid)
            if not article: continue
            
            article_actions = actions[j]
            is_click = "Click" in article_actions
            
            if is_click:
                feature_extractor.update_user_profile(user_id, article, "Click")
                
            # ONLINE LEARNING UPDATE (LinUCB)
            if RANKER_TYPE == "linucb" and group == "treatment":
                # Reward: 1 for Click, 0 for No Click
                reward = 1.0 if is_click else 0.0
                # Only update for the displayed items
                ranker.update(user_id, aid, reward)
        
        # Save LinUCB model periodically (every 100 iterations)
        if RANKER_TYPE == "linucb" and (i + 1) % 100 == 0:
            ranker.save_model()
        
        # Optional: Sleep
        # time.sleep(0.1)

    # Final save
    if RANKER_TYPE == "linucb" and ranker:
        ranker.save_model()

    logger.info("Experiment data collection complete.")

if __name__ == "__main__":
    # Wait for ES to be ready (simple retry mechanism could be added here)
    time.sleep(5) 
    main()
