import logging
from elasticsearch import Elasticsearch, helpers

class ElasticsearchClient:
    def __init__(self, hosts=["http://localhost:9200"], index_name="articles"):
        self.es = Elasticsearch(hosts)
        self.index_name = index_name
        self.logger = logging.getLogger(__name__)

    def create_index(self):
        """Creates the index with default mapping if it doesn't exist."""
        try:
            if self.es.indices.exists(index=self.index_name, ignore=[400, 404]):
                self.logger.info(f"Index {self.index_name} already exists.")
                return

            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "text": {"type": "text"},
                        "topics": {"type": "keyword"}
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping, ignore=[400])
            self.logger.info(f"Created index: {self.index_name}")
        except Exception as e:
            self.logger.error(f"Error creating index: {e}")

    def index_documents(self, documents):
        """Bulk indexes documents."""
        actions = [
            {
                "_index": self.index_name,
                "_id": doc.get("id", doc.get("uuid")),
                "_source": doc
            }
            for doc in documents
        ]
        success, failed = helpers.bulk(self.es, actions, stats_only=True)
        self.logger.info(f"Indexed {success} documents. Failed: {failed}")

    def search(self, query_text, size=10):
        """Basic Elasticsearch Default search."""
        try:
            body = {
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["text", "topics"]
                    }
                },
                "size": size
            }
            response = self.es.search(index=self.index_name, body=body)
            hits = response['hits']['hits']
            return [hit['_id'] for hit in hits]
        except Exception as e:
            self.logger.error(f"Error searching: {e}")
            return []
