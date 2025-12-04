import requests
import logging

class SimulationClient:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    def get_query(self):
        """Fetches a random user and query from the simulation."""
        try:
            response = requests.get(f"{self.base_url}/query")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching query: {e}")
            return None

    def post_ranklist(self, user_id, query_id, ranked_article_ids):
        """Sends the ranked list to the simulation and gets user actions."""
        payload = {
            "user_id": user_id,
            "query_id": query_id,
            "ranked_article_ids": ranked_article_ids
        }
        try:
            response = requests.post(f"{self.base_url}/ranklist", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error posting ranklist: {e}")
            return None
