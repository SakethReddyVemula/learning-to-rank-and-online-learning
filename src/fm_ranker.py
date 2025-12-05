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
            self.item_features = data.get('item_features') # Load features
            
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
        # LightFM expects a CSR matrix of shape (n_items, n_features)
        # However, for prediction, we usually pass item_ids and let it look up if we trained with features.
        # But here we might encounter new items or we just want to pass features for the specific candidates.
        # Wait, LightFM's predict takes `item_features` argument which must match the training dimension.
        # The training dimension is (n_total_items, n_total_features).
        # If we trained with item features (identity + topics), we need to provide that.
        
        # Actually, standard LightFM usage with features:
        # Train: fit(interactions, item_features=F) where F is (n_items, n_features)
        # Predict: predict(user_id, item_ids, item_features=F)
        # So we need the FULL feature matrix F available or at least the rows corresponding to item_ids?
        # No, we need the full F if we pass it, or we need to construct it such that it aligns.
        
        # Simpler approach:
        # We save the `item_features` matrix during training and load it here?
        # Or we reconstruct it? Reconstructing is safer if we have the mappings.
        # But `predict` expects `item_features` to be of shape (n_items, n_features).
        # If we pass `item_ids` as indices into this matrix, it works.
        
        # So, we need:
        # 1. The full item feature matrix used during training (or an updated one).
        # 2. Map candidate_ids to their internal indices.
        
        # Let's assume we load the feature matrix or reconstruct it on the fly?
        # Reconstructing on the fly for *all* items is expensive.
        # Let's try to save the feature matrix in the model file for simplicity.
        pass

    def rank(self, user_id, query_text, candidate_ids, epsilon=None):
        """Ranks candidates using LightFM scores."""
        if not self.model:
            return candidate_ids

        # Map User
        u_idx = self.user_map.get(user_id)
        if u_idx is None:
            # New user: LightFM can handle this if we had user features, but we only have ID.
            # Fallback to baseline or random?
            # Or just return as is.
            return candidate_ids

        # Map Items
        item_indices = []
        valid_candidates = []
        
        for aid in candidate_ids:
            if aid in self.item_map:
                item_indices.append(self.item_map[aid])
                valid_candidates.append(aid)
            else:
                # Unknown item. If we had pure content features we could handle it,
                # but LightFM with ID+Features usually expects known ID for the ID part.
                # We'll skip or append with low score.
                pass
        
        if not item_indices:
            return candidate_ids

        # Predict
        # We need the item_features matrix if we trained with it.
        # Let's assume we saved it in `self.item_features` (loaded from pickle).
        item_features = getattr(self, 'item_features', None)
        
        scores = self.model.predict(u_idx, np.array(item_indices), item_features=item_features)
        
        # Sort
        ranked_pairs = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
        
        # Merge back any skipped candidates (at the end)
        ranked_ids = [aid for aid, score in ranked_pairs]
        for aid in candidate_ids:
            if aid not in ranked_ids:
                ranked_ids.append(aid)
                
        return ranked_ids
