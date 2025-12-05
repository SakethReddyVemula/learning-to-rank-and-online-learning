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

# Configure logging
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

def train_xgboost(articles, logs):
    logger.info("Training XGBoost model...")
    # Initialize Feature Extractor
    feature_extractor = FeatureExtractor()
    feature_extractor.load_article_cache(articles)

    # Prepare Training Data
    logger.info("Processing logs and extracting features...")
    X = []
    y = []

    for log in tqdm(logs, desc="Extracting Features"):
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
        n_estimators=900,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        man_child_weight=3,
        reg_alpha=0.5,
        reg_lambda=1.2,
        gamma=1.0,
        tree_method="hist",
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

BPR_MODEL_FILE = "models/bpr_model.pkl"

def train_bpr(articles, logs):
    logger.info("Training BPR-MF model...")
    
    # 1. Build User-Item Interactions & Mappings
    user_map = {} 
    item_map = {} 
    
    # Set of (u_idx, i_idx) for positive interactions
    positives = set()
    
    u_counter = 0
    i_counter = 0
    
    # Also keep track of all items for negative sampling
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
            
            # If clicked, add to positives
            if i < len(actions) and "Click" in actions[i]:
                positives.add((u_idx, i_idx))

    if not positives:
        logger.error("No clicks found for BPR training.")
        return

    n_users = u_counter
    n_items = i_counter
    logger.info(f"Users: {n_users}, Items: {n_items}, Interactions: {len(positives)}")

    # 2. Initialize Factors
    n_factors = 20
    learning_rate = 0.01
    reg = 0.01
    epochs = 20
    
    user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
    item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
    
    positives_list = list(positives)
    
    # 3. SGD Training
    for epoch in range(epochs):
        random.shuffle(positives_list)
        loss = 0
        
        for u, i in tqdm(positives_list, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Sample negative j
            j = random.choice(all_items)
            while (u, j) in positives:
                j = random.choice(all_items)
            
            # Predict scores
            x_ui = np.dot(user_factors[u], item_factors[i])
            x_uj = np.dot(user_factors[u], item_factors[j])
            x_uij = x_ui - x_uj
            
            # Sigmoid
            sigmoid = 1 / (1 + np.exp(-x_uij))
            
            # Update gradients
            # d(ln sigma)/d(theta) = (1 - sigma) * d(x_uij)/d(theta)
            coeff = 1 - sigmoid
            
            # User update
            d_u = coeff * (item_factors[i] - item_factors[j]) + reg * user_factors[u]
            user_factors[u] += learning_rate * d_u
            
            # Item i update
            d_i = coeff * user_factors[u] + reg * item_factors[i]
            item_factors[i] += learning_rate * d_i
            
            # Item j update
            d_j = coeff * (-user_factors[u]) + reg * item_factors[j]
            item_factors[j] += learning_rate * d_j
            
            loss += -np.log(sigmoid)
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    # 4. Save Model
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

def train_fm(articles, logs):
    logger.info("Training Factorization Machines (LightFM)...")
    
    # 1. Mappings
    user_map = {}
    item_map = {}
    topic_map = {}
    
    # Get all topics first
    all_topics = set()
    for art in articles:
        for t in art.get("topics", []):
            all_topics.add(t)
    
    topic_list = sorted(list(all_topics))
    for i, t in enumerate(topic_list):
        topic_map[t] = i
        
    n_topics = len(topic_list)
    
    # 2. Build Interactions
    # We need to map all users and items involved in interactions
    # AND all items in articles (to build feature matrix for all)
    
    # Map all items from articles first (to ensure feature matrix covers them)
    for art in articles:
        aid = art.get("id", art.get("uuid"))
        if aid not in item_map:
            item_map[aid] = len(item_map)
            
    # Map users from logs
    interactions_list = [] # (u_idx, i_idx)
    
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
                if i < len(actions) and "Click" in actions[i]:
                    interactions_list.append((u_idx, i_idx))
    
    n_users = len(user_map)
    n_items = len(item_map)
    
    # Create Interaction Matrix
    # LightFM expects (n_users, n_items)
    data = np.ones(len(interactions_list))
    rows = [x[0] for x in interactions_list]
    cols = [x[1] for x in interactions_list]
    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    
    logger.info(f"Interaction Matrix: {n_users} Users x {n_items} Items, {len(interactions_list)} interactions.")
    
    # 3. Build Item Features
    # Shape: (n_items, n_features)
    # Features = Identity (n_items) + Topics (n_topics)
    # LightFM by default adds identity if no features passed.
    # If we pass features, we usually want to include identity explicitly or rely on content only.
    # Hybrid = Identity + Content.
    
    # Let's define features as just Topics for now? 
    # No, Hybrid is best. So we want a matrix of shape (n_items, n_items + n_topics).
    # But that's huge. LightFM allows supplying a feature matrix where rows=items.
    # If we want identity, we can use `build_item_features` from lightfm.data or just construct sparse matrix.
    # Let's construct: (n_items, n_topics) and let LightFM add identity?
    # Actually, LightFM `fit` has `item_features`. If supplied, it uses it.
    # If we want hybrid, we usually construct a matrix where we have identity features AND topic features.
    # But for simplicity and memory, let's try using JUST topic features first?
    # No, that's pure content-based. We want Hybrid.
    # LightFM documentation says: "If you want to use both [id and metadata], you should append an identity matrix to your feature matrix."
    # But creating a (n_items, n_items) identity matrix is memory heavy if dense, but fine if sparse.
    
    # Let's try constructing a sparse matrix of shape (n_items, n_topics).
    # And we will NOT add identity manually, but we will see if LightFM performs well.
    # Wait, if we don't add identity, it can't learn per-item biases/factors effectively if features are shared.
    # So we SHOULD add identity.
    # Feature dimension = n_topics.
    # But we want per-item latent factors too.
    # Let's use `identity` features implicitly? No, LightFM doesn't do that if features are passed.
    
    # Okay, let's build (n_items, n_topics) matrix.
    # And we will rely on the fact that if topics are unique enough it helps.
    # BUT to be truly hybrid, we need identity.
    # Let's construct a LIL matrix of shape (n_items, n_items + n_topics).
    # This might be too big for 12k items? 12k x 12k is 144M entries (sparse though).
    # 12k rows. Each row has 1 (identity) + ~2 (topics) = 3 entries.
    # Total entries = 36k. Very sparse. Totally fine.
    
    n_features = n_items + n_topics
    feat_rows = []
    feat_cols = []
    feat_data = []
    
    for aid, i_idx in item_map.items():
        # Identity feature
        feat_rows.append(i_idx)
        feat_cols.append(i_idx)
        feat_data.append(1.0)
        
        # Topic features
        # We need to find the article object.
        # We iterated articles before.
        # Let's do a lookup or iterate again.
        pass # Optimization: do this in the first loop?
        
    # Re-iterate articles to fill features
    for art in articles:
        aid = art.get("id", art.get("uuid"))
        if aid in item_map:
            i_idx = item_map[aid]
            for t in art.get("topics", []):
                if t in topic_map:
                    t_idx = topic_map[t]
                    # Feature index = n_items + t_idx
                    feat_rows.append(i_idx)
                    feat_cols.append(n_items + t_idx)
                    feat_data.append(1.0)
                    
    item_features = csr_matrix((feat_data, (feat_rows, feat_cols)), shape=(n_items, n_features))
    
    # 4. Train
    model = LightFM(loss='warp', no_components=20, learning_rate=0.05)
    model.fit(interaction_matrix, item_features=item_features, epochs=30, num_threads=2)
    
    # 5. Save
    model_data = {
        'model': model,
        'user_map': user_map,
        'item_map': item_map,
        'topic_map': topic_map,
        'item_features': item_features # Save this for ranking!
    }
    
    with open(FM_MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)
        
    logger.info(f"FM Model saved to {FM_MODEL_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="xgboost", choices=["xgboost", "mf", "bpr", "fm"], help="Model type to train")
    args = parser.parse_args()
    
    articles, logs = load_data()
    
    if articles and logs:
        if args.model_type == "xgboost":
            train_xgboost(articles, logs)
        elif args.model_type == "mf":
            train_mf(articles, logs)
        elif args.model_type == "bpr":
            train_bpr(articles, logs)
        elif args.model_type == "fm":
            train_fm(articles, logs)
