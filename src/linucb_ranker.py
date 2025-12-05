import numpy as np
import pickle
import logging
import os

class LinUCBRanker:
    def __init__(self, model_path=None, alpha=1.0):
        self.logger = logging.getLogger(__name__)
        self.alpha = alpha
        self.model_path = model_path
        
        self.topics = [
            "education", "environment", "health", "labor", 
            "lifestyle and leisure", "religion and belief", 
            "science and technology", "sport"
        ]
        self.topic_map = {t: i for i, t in enumerate(self.topics)}
        self.n_features = len(self.topics)
        
        self.user_models = {}
        
        self.article_cache = {}

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_article_cache(self, articles):
        """Pre-loads articles for feature extraction."""
        for article in articles:
            aid = article.get("id", article.get("uuid"))
            self.article_cache[aid] = article

    def get_article_features(self, article_id):
        """Returns feature vector for an article (One-hot encoding of topics)."""
        article = self.article_cache.get(article_id)
        if not article:
            return np.zeros(self.n_features)
        
        features = np.zeros(self.n_features)
        topics = article.get("topics", [])
        for t in topics:
            if t in self.topic_map:
                features[self.topic_map[t]] = 1.0
        
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features

    def _init_user_model(self, user_id):
        """Initializes A and b for a new user."""
        if user_id not in self.user_models:
            self.user_models[user_id] = {
                'A': np.identity(self.n_features),
                'b': np.zeros(self.n_features)
            }

    def rank(self, user_id, query_text, candidate_ids, epsilon=None):
        """
        Ranks candidates using LinUCB scores.
        epsilon is ignored here as LinUCB has its own exploration (alpha).
        """
        self._init_user_model(user_id)
        
        model = self.user_models[user_id]
        A_inv = np.linalg.inv(model['A'])
        theta = np.dot(A_inv, model['b'])
        
        scores = []
        valid_candidates = []
        
        for aid in candidate_ids:
            x = self.get_article_features(aid)
            
            mean = np.dot(x, theta)
            var = np.dot(np.dot(x, A_inv), x)
            std_dev = np.sqrt(var)
            
            ucb_score = mean + self.alpha * std_dev
            
            scores.append(ucb_score)
            valid_candidates.append(aid)
            
        ranked_pairs = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
        return [aid for aid, score in ranked_pairs]

    def update(self, user_id, article_id, reward):
        """Updates the user model based on reward (Click=1, No Click=0)."""
        self._init_user_model(user_id)
        
        x = self.get_article_features(article_id)
        
        self.user_models[user_id]['A'] += np.outer(x, x)
        
        self.user_models[user_id]['b'] += reward * x
        

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.user_models, f)
            self.logger.info(f"LinUCB model saved to {path}")

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                self.user_models = pickle.load(f)
            self.logger.info(f"Loaded LinUCB model from {path}")
        except Exception as e:
            self.logger.error(f"Error loading LinUCB model: {e}")
