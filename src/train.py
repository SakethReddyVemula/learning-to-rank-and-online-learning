import json
import logging
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import argparse
import pickle
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from src.features import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTICLES_FILE = "data/articles.jsonl"
LOGS_FILE = "interaction_logs.jsonl"
XGBOOST_MODEL_FILE = "models/ranker.model"
MF_MODEL_FILE = "models/mf_model.pkl"

def load_data():
    logger.info("Loading data...")
    if not os.path.exists(LOGS_FILE):
        logger.error(f"Log file {LOGS_FILE} not found. Run baseline first.")
        return None, None

    articles = []
    with open(ARTICLES_FILE, 'r') as f:
        for line in f:
            articles.append(json.loads(line))
            
    logs = []
    with open(LOGS_FILE, 'r') as f:
        for line in f:
            logs.append(json.loads(line))
            
    logger.info(f"Loaded {len(articles)} articles and {len(logs)} interaction logs.")
    return articles, logs

def train_xgboost(articles, logs):
    logger.info("Training XGBoost model...")
    # Initialize Feature Extractor
    feature_extractor = FeatureExtractor()
    feature_extractor.load_article_cache(articles)

    # Prepare Training Data
    logger.info("Processing logs and extracting features...")
    X = []
    y = []

    for log in logs:
        user_id = log['user_id']
        query_text = log['query_text']
        ranked_ids = log['ranked_article_ids']
        actions = log['actions']
        
        for i, aid in enumerate(ranked_ids):
            article = feature_extractor.article_cache.get(aid)
            if not article: continue
            
            # Label
            label = 0
            if i < len(actions) and "Click" in actions[i]:
                label = 1
            
            # Extract features
            feats = feature_extractor.get_features(user_id, article, query_text)
            X.append(feats)
            y.append(label)
            
            # Update profile if clicked
            if label == 1:
                feature_extractor.update_user_profile(user_id, article, "Click")

    df = pd.DataFrame(X)
    y = np.array(y)
    
    logger.info(f"Training data shape: {df.shape}")
    logger.info(f"Positive samples: {sum(y)}")

    # Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=42)

    # Train XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # Save Model
    if not os.path.exists("models"):
        os.makedirs("models")
        
    model.save_model(XGBOOST_MODEL_FILE)
    logger.info(f"XGBoost model saved to {XGBOOST_MODEL_FILE}")

def train_mf(articles, logs):
    logger.info("Training Matrix Factorization model...")
    
    # 1. Build User-Item Matrix
    user_map = {} # user_id -> index
    item_map = {} # article_id -> index
    
    data = []
    rows = []
    cols = []
    
    u_counter = 0
    i_counter = 0
    
    for log in logs:
        user_id = log['user_id']
        if user_id not in user_map:
            user_map[user_id] = u_counter
            u_counter += 1
            
        u_idx = user_map[user_id]
        
        ranked_ids = log['ranked_article_ids']
        actions = log['actions']
        
        for i, aid in enumerate(ranked_ids):
            if aid not in item_map:
                item_map[aid] = i_counter
                i_counter += 1
            
            i_idx = item_map[aid]
            
            # Interaction Value: 1 for Click
            if i < len(actions) and "Click" in actions[i]:
                rows.append(u_idx)
                cols.append(i_idx)
                data.append(1.0) # Click
    
    if not data:
        logger.error("No clicks found for MF training.")
        return

    # Create Sparse Matrix
    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(u_counter, i_counter))
    
    logger.info(f"Interaction Matrix: {interaction_matrix.shape} with {interaction_matrix.nnz} entries.")
    
    # 2. SVD
    n_components = 20
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(interaction_matrix)
    item_factors = svd.components_.T
    
    logger.info(f"User Factors: {user_factors.shape}, Item Factors: {item_factors.shape}")
    
    # 3. Save Model
    model_data = {
        'user_map': user_map,
        'item_map': item_map,
        'user_factors': user_factors,
        'item_factors': item_factors
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    with open(MF_MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)
        
    logger.info(f"MF Model saved to {MF_MODEL_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="xgboost", choices=["xgboost", "mf"], help="Model type to train")
    args = parser.parse_args()
    
    articles, logs = load_data()
    
    if articles and logs:
        if args.model_type == "xgboost":
            train_xgboost(articles, logs)
        elif args.model_type == "mf":
            train_mf(articles, logs)
