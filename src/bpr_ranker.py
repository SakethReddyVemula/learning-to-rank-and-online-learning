import pickle
import numpy as np
import logging
import random

class BPRRanker:
    def __init__(self, model_path=None):
        self.logger = logging.getLogger(__name__)
        self.model_data = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Loads the BPR model (Latent factors and mappings)."""
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            self.logger.info(f"Loaded BPR model from {model_path}")
            
            self.user_map = self.model_data['user_map']
            self.item_map = self.model_data['item_map']
            self.user_factors = self.model_data['user_factors']
            self.item_factors = self.model_data['item_factors']
            
        except Exception as e:
            self.logger.error(f"Error loading BPR model: {e}")
            self.model_data = None

    def rank(self, user_id, query_text, candidate_ids, epsilon=0.1):
        """Re-ranks candidates using dot product of latent factors."""
        if not candidate_ids:
            return []

        if random.random() < epsilon:
            self.logger.info(f"Exploration triggered for user {user_id}")
            shuffled = list(candidate_ids)
            random.shuffle(shuffled)
            return shuffled

        if not self.model_data:
            return candidate_ids

        scores = []
        valid_candidates = []
        
        if user_id not in self.user_map:
            return candidate_ids
            
        u_idx = self.user_map[user_id]
        u_vec = self.user_factors[u_idx]

        for aid in candidate_ids:
            if aid in self.item_map:
                i_idx = self.item_map[aid]
                i_vec = self.item_factors[i_idx]
                score = np.dot(u_vec, i_vec)
            else:
                score = 0.0
            
            scores.append(score)
            valid_candidates.append(aid)

        ranked_pairs = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
        return [aid for aid, score in ranked_pairs]
