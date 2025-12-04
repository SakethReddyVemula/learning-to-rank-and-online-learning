import logging
import json
import os

def setup_logger(log_file="interaction_logs.jsonl"):
    """Sets up a logger that writes JSON objects to a file."""
    logger = logging.getLogger("interaction_logger")
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

def log_interaction(logger, interaction_data):
    """Logs a single interaction dictionary as a JSON line."""
    logger.info(json.dumps(interaction_data))
