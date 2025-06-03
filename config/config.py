import os
import json
from rag.config_schema import RulebookConfig


def load_config_from_file(path="config/keys.json") -> RulebookConfig:
    with open(path) as f:
        keys = json.load(f)
    return RulebookConfig(
        openai_api_key=keys["OPENAI_API_KEY"],
        pinecone_api_key=keys.get("PINECONE_API_KEY"),
        pinecone_env=keys.get("PINECONE_ENV"),
        pinecone_index_name=keys.get("PINECONE_INDEX_NAME"),
        langchain_api_key=keys.get("LANGCHAIN_API_KEY"),
        # langchain_tracing=bool(keys.get("LANGCHAIN_TRACING", False)),
        # mlflow_tracking=bool(keys.get("MLFLOW_TRACKING", False)),
        mlflow_uri=keys.get("MLFLOW_TRACKING_URI"),
        mlflow_experiment=keys.get("MLFLOW_EXPERIMENT"),
    )
