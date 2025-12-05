import json
import logging
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import argparse
import pickle
import numpy as np
import random
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from src.features import FeatureExtractor
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
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

def calculate_relevance(actions, use_extended=False):
    """Calculates relevance score based on actions."""
    if not use_extended:
        return 1.0 if "Click" in actions else 0.0
    
    score = 0.0
    if "Click" in actions:
        score += 1.0
    if "Like" in actions:
        score += 2.0
    if "Share" in actions:
        score += 3.0
    if "Bookmark" in actions:
        score += 3.0
    return score

def train_xgboost(articles, logs, use_extended=False):
    logger.info(f"Training XGBoost model (Extended Actions: {use_extended})...")
    feature_extractor = FeatureExtractor()
    feature_extractor.load_article_cache(articles)

    logger.info("Processing logs and extracting features...")
    X = []
    y = []
    weights = []

    for log in tqdm(logs, desc="Extracting Features"):
        user_id = log['user_id']
        query_text = log['query_text']
        ranked_ids = log['ranked_article_ids']
        actions = log['actions']
        
        for i, aid in enumerate(ranked_ids):
            article = feature_extractor.article_cache.get(aid)
            if not article: continue
            
            # Determine label and weight
            if i < len(actions):
                act = actions[i]
                relevance = calculate_relevance(act, use_extended)
            else:
                relevance = 0.0
            
            label = 1 if relevance > 0 else 0
            weight = relevance if relevance > 0 else 1.0 # Default weight 1 for negatives
            
            feats = feature_extractor.get_features(user_id, article, query_text)
            X.append(feats)
            y.append(label)
            weights.append(weight)
            
            if label == 1:
                feature_extractor.update_user_profile(user_id, article, "Click")

    df = pd.DataFrame(X)
    y = np.array(y)
    weights = np.array(weights)
    
    logger.info(f"Training data shape: {df.shape}")
    logger.info(f"Positive samples: {sum(y)}")

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(df, y, weights, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=900,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        reg_alpha=0.5,
        reg_lambda=1.2,
        gamma=1.0,
        tree_method="hist",
    )

    model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val], verbose=True)

    if not os.path.exists("models"):
        os.makedirs("models")
        
    model.save_model(XGBOOST_MODEL_FILE)
    logger.info(f"XGBoost model saved to {XGBOOST_MODEL_FILE}")

def train_mf(articles, logs, use_extended=False):
    logger.info(f"Training Matrix Factorization model (Extended Actions: {use_extended})...")
    
    user_map = {}
    item_map = {}
    
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
            
            if i < len(actions):
                relevance = calculate_relevance(actions[i], use_extended)
                if relevance > 0:
                    rows.append(u_idx)
                    cols.append(i_idx)
                    data.append(relevance) # Use relevance score as rating
    
    if not data:
        logger.error("No interactions found for MF training.")
        return

    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(u_counter, i_counter))
    
    logger.info(f"Interaction Matrix: {interaction_matrix.shape} with {interaction_matrix.nnz} entries.")
    
    n_components = 20
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(interaction_matrix)
    item_factors = svd.components_.T
    
    logger.info(f"User Factors: {user_factors.shape}, Item Factors: {item_factors.shape}")
    
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

BPR_MODEL_FILE = "models/bpr_model.pkl"

def train_bpr(articles, logs, use_extended=False):
    logger.info(f"Training BPR-MF model (Extended Actions: {use_extended})...")
    
    user_map = {} 
    item_map = {} 
    
    positives = set()
    
    u_counter = 0
    i_counter = 0
    
    all_items = []
    
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
                all_items.append(item_map[aid])
            
            i_idx = item_map[aid]
            
            if i < len(actions):
                relevance = calculate_relevance(actions[i], use_extended)
                if relevance > 0:
                    positives.add((u_idx, i_idx))

    if not positives:
        logger.error("No interactions found for BPR training.")
        return

    n_users = u_counter
    n_items = i_counter
    logger.info(f"Users: {n_users}, Items: {n_items}, Interactions: {len(positives)}")

    n_factors = 20
    learning_rate = 0.01
    reg = 0.01
    epochs = 20
    
    user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
    item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
    
    positives_list = list(positives)
    
    for epoch in range(epochs):
        random.shuffle(positives_list)
        loss = 0
        
        for u, i in tqdm(positives_list, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            j = random.choice(all_items)
            while (u, j) in positives:
                j = random.choice(all_items)
            
            x_ui = np.dot(user_factors[u], item_factors[i])
            x_uj = np.dot(user_factors[u], item_factors[j])
            x_uij = x_ui - x_uj
            
            sigmoid = 1 / (1 + np.exp(-x_uij))
            
            coeff = 1 - sigmoid
            
            d_u = coeff * (item_factors[i] - item_factors[j]) - reg * user_factors[u]
            user_factors[u] += learning_rate * d_u
            
            d_i = coeff * user_factors[u] - reg * item_factors[i]
            item_factors[i] += learning_rate * d_i
            
            d_j = coeff * (-user_factors[u]) - reg * item_factors[j]
            item_factors[j] += learning_rate * d_j
            
            loss += -np.log(sigmoid)
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    model_data = {
        'user_map': user_map,
        'item_map': item_map,
        'user_factors': user_factors,
        'item_factors': item_factors
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    with open(BPR_MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)
        
    logger.info(f"BPR Model saved to {BPR_MODEL_FILE}")

from lightfm import LightFM

FM_MODEL_FILE = "models/fm_model.pkl"

def train_fm(articles, logs, use_extended=False):
    logger.info(f"Training Factorization Machines (LightFM) (Extended Actions: {use_extended})...")
    
    user_map = {}
    item_map = {}
    topic_map = {}
    
    all_topics = set()
    for art in articles:
        for t in art.get("topics", []):
            all_topics.add(t)
    
    topic_list = sorted(list(all_topics))
    for i, t in enumerate(topic_list):
        topic_map[t] = i
        
    n_topics = len(topic_list)
    
    for art in articles:
        aid = art.get("id", art.get("uuid"))
        if aid not in item_map:
            item_map[aid] = len(item_map)
            
    interactions_list = [] # (u_idx, i_idx, weight)
    
    for log in tqdm(logs, desc="Processing Logs for FM"):
        user_id = log['user_id']
        if user_id not in user_map:
            user_map[user_id] = len(user_map)
        u_idx = user_map[user_id]
        
        ranked_ids = log['ranked_article_ids']
        actions = log['actions']
        
        for i, aid in enumerate(ranked_ids):
            if aid in item_map:
                i_idx = item_map[aid]
                if i < len(actions):
                    relevance = calculate_relevance(actions[i], use_extended)
                    if relevance > 0:
                        interactions_list.append((u_idx, i_idx, relevance))
    
    n_users = len(user_map)
    n_items = len(item_map)
    
    data = [x[2] for x in interactions_list]
    rows = [x[0] for x in interactions_list]
    cols = [x[1] for x in interactions_list]
    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    
    logger.info(f"Interaction Matrix: {n_users} Users x {n_items} Items, {len(interactions_list)} interactions.")
    
    
    n_features = n_items + n_topics
    feat_rows = []
    feat_cols = []
    feat_data = []
    
    for aid, i_idx in item_map.items():
        feat_rows.append(i_idx)
        feat_cols.append(i_idx)
        feat_data.append(1.0)
        

        
    for art in articles:
        aid = art.get("id", art.get("uuid"))
        if aid in item_map:
            i_idx = item_map[aid]
            for t in art.get("topics", []):
                if t in topic_map:
                    t_idx = topic_map[t]
                    feat_rows.append(i_idx)
                    feat_cols.append(n_items + t_idx)
                    feat_data.append(1.0)
                    
    item_features = csr_matrix((feat_data, (feat_rows, feat_cols)), shape=(n_items, n_features))
    
    model = LightFM(loss='warp', no_components=20, learning_rate=0.05)
    model.fit(interaction_matrix, item_features=item_features, epochs=30, num_threads=2, sample_weight=interaction_matrix.tocoo())
    
    model_data = {
        'model': model,
        'user_map': user_map,
        'item_map': item_map,
        'topic_map': topic_map,
        'item_features': item_features
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    with open(FM_MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)
        
    logger.info(f"FM Model saved to {FM_MODEL_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="xgboost", choices=["xgboost", "mf", "bpr", "fm"], help="Model type to train")
    parser.add_argument("--use_extended_actions", action="store_true", help="Use extended actions (Like, Share, Bookmark) for training")
    args = parser.parse_args()
    
    articles, logs = load_data()
    
    if articles and logs:
        if args.model_type == "xgboost":
            train_xgboost(articles, logs, args.use_extended_actions)
        elif args.model_type == "mf":
            train_mf(articles, logs, args.use_extended_actions)
        elif args.model_type == "bpr":
            train_bpr(articles, logs, args.use_extended_actions)
        elif args.model_type == "fm":
            train_fm(articles, logs, args.use_extended_actions)
