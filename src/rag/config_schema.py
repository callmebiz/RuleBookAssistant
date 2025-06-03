from dataclasses import dataclass
from typing import Optional


@dataclass
class RulebookConfig:
    openai_api_key: str
    pinecone_api_key: Optional[str] = None
    pinecone_env: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    langchain_api_key: Optional[str] = None
    langchain_tracing: bool = False
    mlflow_tracking: bool = False
    mlflow_uri: Optional[str] = None
    mlflow_experiment: Optional[str] = None
