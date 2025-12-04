import pickle
import numpy as np
import logging
import random

class MFRanker:
    def __init__(self, model_path=None):
        self.logger = logging.getLogger(__name__)
        self.model_data = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Loads the MF model (SVD components and mappings)."""
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            self.logger.info(f"Loaded MF model from {model_path}")
            
            # Unpack data
            self.user_map = self.model_data['user_map']
            self.item_map = self.model_data['item_map']
            self.user_factors = self.model_data['user_factors']
            self.item_factors = self.model_data['item_factors']
            
        except Exception as e:
            self.logger.error(f"Error loading MF model: {e}")
            self.model_data = None

    def rank(self, user_id, query_text, candidate_ids, epsilon=0.1):
        """Re-ranks candidates using dot product of latent factors."""
        if not candidate_ids:
            return []

        # Epsilon-Greedy Exploration
        if random.random() < epsilon:
            self.logger.info(f"Exploration triggered for user {user_id}")
            # Create a copy to shuffle
            shuffled = list(candidate_ids)
            random.shuffle(shuffled)
            return shuffled

        if not self.model_data:
            # Fallback if no model loaded
            return candidate_ids

        scores = []
        valid_candidates = []
        
        # Check if user is known
        if user_id not in self.user_map:
            # Cold start user: Return baseline order (or random)
            return candidate_ids
            
        u_idx = self.user_map[user_id]
        u_vec = self.user_factors[u_idx]

        for aid in candidate_ids:
            if aid in self.item_map:
                i_idx = self.item_map[aid]
                i_vec = self.item_factors[i_idx]
                score = np.dot(u_vec, i_vec)
            else:
                # Cold start item: Assign neutral/low score
                score = 0.0
            
            scores.append(score)
            valid_candidates.append(aid)

        # Sort by score descending
        ranked_pairs = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
        return [aid for aid, score in ranked_pairs]
