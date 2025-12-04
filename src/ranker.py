import logging
import xgboost as xgb
import numpy as np
import pandas as pd

import random

class PersonalizedRanker:
    def __init__(self, feature_extractor, model_path=None):
        self.feature_extractor = feature_extractor
        self.logger = logging.getLogger(__name__)
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Loads a trained XGBoost model."""
        try:
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            self.logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")

    def rank(self, user_id, query_text, candidate_ids, epsilon=0.1):
        """Re-ranks candidate articles for the user with epsilon-greedy exploration."""
        if not candidate_ids:
            return []

        # 1. Extract features for all candidates
        features_list = []
        valid_candidates = []
        
        for aid in candidate_ids:
            article = self.feature_extractor.article_cache.get(aid)
            if not article:
                continue
            
            feats = self.feature_extractor.get_features(user_id, article, query_text)
            features_list.append(feats)
            valid_candidates.append(aid)

        if not valid_candidates:
            return []

        # Epsilon-Greedy Exploration
        if random.random() < epsilon:
            self.logger.info(f"Exploration triggered for user {user_id}")
            random.shuffle(valid_candidates)
            return valid_candidates

        # 2. Predict scores
        if self.model:
            # Convert to DMatrix for XGBoost
            df = pd.DataFrame(features_list)
            # Ensure column order matches training
            # For now, we rely on pandas column order, but in prod we should enforce it
            dtest = xgb.DMatrix(df)
            scores = self.model.predict(dtest)
        else:
            # Fallback: Sort by topic match score if no model
            scores = [f["topic_match_score"] for f in features_list]

        # 3. Sort candidates by score
        # Zip candidates with scores, sort descending
        ranked_pairs = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)
        
        return [aid for aid, score in ranked_pairs]
