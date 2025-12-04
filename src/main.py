import json
import time
import logging
import os
from src.elasticsearch_client import ElasticsearchClient
from src.simulation_client import SimulationClient
from src.logger import setup_logger, log_interaction

from src.features import FeatureExtractor
from src.ranker import PersonalizedRanker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = "data/articles.jsonl"
MODEL_FILE = "models/ranker.model"
NUM_ITERATIONS = 500 # Number of queries to process

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
    
    ranker = PersonalizedRanker(feature_extractor, model_path=MODEL_FILE)

    # 3. Interaction Loop
    logger.info("Starting A/B Experiment loop...")
    for i in range(NUM_ITERATIONS):
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
        
        logger.info(f"Iteration {i+1}: Query='{query_text}' (User: {user_id}, Group: {group})")

        # Get Candidates (Elasticsearch Default)
        # Fetch more candidates for re-ranking (e.g., 50)
        candidate_ids = es_client.search(query_text, size=50)
        
        if group == "treatment":
            # Re-rank with exploration (epsilon=0.1)
            ranked_ids = ranker.rank(user_id, query_text, candidate_ids, epsilon=0.1)
            final_ranking = ranked_ids[:10]
        else:
            # Baseline: Just take top 10 from ES
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
            "experiment_group": group
        }
        log_interaction(interaction_logger, interaction)
        
        # Update User Profile (Always update, regardless of group, to keep history fresh)
        for j, aid in enumerate(final_ranking):
            article = feature_extractor.article_cache.get(aid)
            if not article: continue
            
            article_actions = actions[j]
            if "Click" in article_actions:
                feature_extractor.update_user_profile(user_id, article, "Click")
        
        # Optional: Sleep
        time.sleep(0.1)

    logger.info("Experiment data collection complete.")

if __name__ == "__main__":
    # Wait for ES to be ready (simple retry mechanism could be added here)
    time.sleep(5) 
    main()
