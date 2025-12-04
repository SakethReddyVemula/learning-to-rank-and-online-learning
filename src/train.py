import json
import logging
import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from src.features import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = "data/articles.jsonl"
LOG_FILE = "interaction_logs.jsonl"
MODEL_FILE = "models/ranker.model"

def load_articles(file_path):
    articles = []
    with open(file_path, 'r') as f:
        for line in f:
            articles.append(json.loads(line))
    return articles

def load_logs(file_path):
    logs = []
    with open(file_path, 'r') as f:
        for line in f:
            logs.append(json.loads(line))
    return logs

def train():
    logger.info("Loading data...")
    if not os.path.exists(LOG_FILE):
        logger.error(f"Log file {LOG_FILE} not found. Run baseline first.")
        return

    articles = load_articles(DATA_FILE)
    logs = load_logs(LOG_FILE)
    
    logger.info(f"Loaded {len(articles)} articles and {len(logs)} interaction logs.")

    # Initialize Feature Extractor
    feature_extractor = FeatureExtractor()
    feature_extractor.load_article_cache(articles)

    # Replay history to build user profiles and generate training data
    X = []
    y = []
    qids = [] # Query IDs for ranking (groups)
    
    logger.info("Processing logs and extracting features...")
    for i, log in enumerate(logs):
        user_id = log['user_id']
        query_text = log['query_text']
        ranked_ids = log['ranked_article_ids']
        actions = log['actions']
        
        # We need to update user profile *after* processing this interaction for training?
        # Or should we use the profile *at that point in time*?
        # Correct way: Use profile built from *previous* interactions.
        
        # For each article in the ranked list
        for j, aid in enumerate(ranked_ids):
            article = feature_extractor.article_cache.get(aid)
            if not article:
                continue
            
            # Extract features (using current profile state)
            features = feature_extractor.get_features(user_id, article, query_text)
            X.append(features)
            
            # Determine label
            # Action is a list of strings/dicts e.g. ["Click", {"Dwell": ...}] or []
            article_actions = actions[j]
            label = 1 if "Click" in article_actions else 0
            y.append(label)
            
            # Group ID (just the index of the query in the logs is sufficient for grouping)
            qids.append(i)

        # Update profile for *next* time
        for j, aid in enumerate(ranked_ids):
             article = feature_extractor.article_cache.get(aid)
             if not article: continue
             
             article_actions = actions[j]
             if "Click" in article_actions:
                 feature_extractor.update_user_profile(user_id, article, "Click")

    if not X:
        logger.error("No training data generated.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(X)
    labels = pd.Series(y)
    groups = pd.Series(qids)
    
    logger.info(f"Training data shape: {df.shape}")
    logger.info(f"Positive samples: {labels.sum()}")

    # Train/Validation Split (Group-aware)
    # Simple split by query index
    unique_groups = groups.unique()
    train_groups, val_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)
    
    train_mask = groups.isin(train_groups)
    val_mask = groups.isin(val_groups)
    
    X_train, y_train = df[train_mask], labels[train_mask]
    X_val, y_val = df[val_mask], labels[val_mask]
    
    # XGBoost Ranker
    # We need to sort data by group for XGBoost ranking
    # Actually, simpler to use standard classifier for pointwise or just use group info
    # Let's use Pointwise classification (Binary Logistic) for simplicity first, 
    # as it's often robust enough and easier to implement than pairwise for small data.
    # If we want pairwise, we need to use 'rank:pairwise' and provide group info.
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    
    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    
    model.save_model(MODEL_FILE)
    logger.info(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train()
