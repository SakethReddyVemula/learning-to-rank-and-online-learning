import numpy as np
import pickle
import logging
import os
from lightfm import LightFM
from scipy.sparse import csr_matrix, lil_matrix

class FMRanker:
    def __init__(self, model_path=None):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.topic_map = {}
        self.article_cache = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        """Loads the LightFM model and mappings."""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.user_map = data['user_map']
            self.item_map = data['item_map']
            self.topic_map = data['topic_map']
            self.item_features = data.get('item_features')
            
            self.logger.info(f"Loaded FM model from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading FM model: {e}")
            self.model = None

    def load_article_cache(self, articles):
        """Pre-loads articles for feature extraction."""
        for article in articles:
            aid = article.get("id", article.get("uuid"))
            self.article_cache[aid] = article

    def _get_item_features(self, candidate_ids):
        """Constructs the item feature matrix for the given candidates."""
        n_candidates = len(candidate_ids)
        n_train_items = len(self.item_map)
        n_train_topics = len(self.topic_map)
        n_total_features = n_train_items + n_train_topics
        
        rows = []
        cols = []
        data = []
        
        for i, aid in enumerate(candidate_ids):
            if aid in self.item_map:
                item_idx = self.item_map[aid]
                rows.append(i)
                cols.append(item_idx)
                data.append(1.0)
            
            article = self.article_cache.get(aid)
            if article:
                topics = article.get("topics", [])
                for topic in topics:
                    if topic in self.topic_map:
                        topic_idx = self.topic_map[topic]
                        col_idx = n_train_items + topic_idx
                        rows.append(i)
                        cols.append(col_idx)
                        data.append(1.0)
        
        return csr_matrix((data, (rows, cols)), shape=(n_candidates, n_total_features))

    def rank(self, user_id, query_text, candidate_ids, epsilon=None):
        """Ranks candidates using LightFM scores."""
        if not self.model:
            return candidate_ids

        u_idx = self.user_map.get(user_id)
        if u_idx is None:
            return candidate_ids

        item_indices = []
        valid_candidates = []
        
        for aid in candidate_ids:
            if aid in self.item_map:
                item_indices.append(self.item_map[aid])
                valid_candidates.append(aid)
            else:
                pass
        
        if not item_indices:
            return candidate_ids

        item_features = getattr(self, 'item_features', None)
        
        scores = self.model.predict(u_idx, np.array(item_indices), item_features=item_features)
        
        ranked_pairs = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
        
        ranked_ids = [aid for aid, score in ranked_pairs]
        for aid in candidate_ids:
            if aid not in ranked_ids:
                ranked_ids.append(aid)
                
        return ranked_ids
