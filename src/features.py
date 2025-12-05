import numpy as np
from collections import defaultdict

class FeatureExtractor:
    def __init__(self):
        self.user_profiles = defaultdict(lambda: defaultdict(int))
        self.article_cache = {}

    def update_user_profile(self, user_id, article_data, action_type="Click"):
        """Updates user profile based on interaction."""
        if action_type == "Click":
            topics = article_data.get("topics", [])
            for topic in topics:
                self.user_profiles[user_id][topic] += 1

    def get_features(self, user_id, article_data, query_text):
        """Extracts features for a (user, article) pair."""
        features = {}
        
        user_profile = self.user_profiles[user_id]
        total_clicks = sum(user_profile.values())
        
        article_topics = article_data.get("topics", [])
        
        topic_match_score = 0
        if total_clicks > 0:
            for topic in article_topics:
                topic_match_score += user_profile.get(topic, 0) / total_clicks
        
        features["topic_match_score"] = topic_match_score
        features["num_topics"] = len(article_topics)
        
        query_tokens = set(query_text.lower().split())
        title_tokens = set(article_data.get("title", "").lower().split())
        features["title_overlap"] = len(query_tokens.intersection(title_tokens))
        
        return features

    def load_article_cache(self, articles):
        """Pre-loads articles into memory for fast feature extraction."""
        for article in articles:
            aid = article.get("id", article.get("uuid"))
            self.article_cache[aid] = article
