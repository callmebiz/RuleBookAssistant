import argparse
import os
from config.config import load_environment
from rag.indexing import index_pdfs
from rag.config_schema import RulebookConfig


# Load environment variables and API keys
keys = load_environment()

# Initialize config with API keys and settings
config = RulebookConfig(
    openai_api_key=keys["OPENAI_API_KEY"],
    pinecone_api_key=keys.get("PINECONE_API_KEY"),
    pinecone_env=keys.get("PINECONE_ENV"),
    pinecone_index_name=keys.get("PINECONE_INDEX_NAME"),
    langchain_api_key=keys.get("LANGCHAIN_API_KEY"),
    langchain_tracing=bool(keys.get("LANGCHAIN_TRACING", False)),
    mlflow_tracking=bool(keys.get("MLFLOW_TRACKING", False)),
    mlflow_uri=keys.get("MLFLOW_TRACKING_URI"),
    mlflow_experiment=keys.get("MLFLOW_EXPERIMENT"),
)

parser = argparse.ArgumentParser(
    description="Index rulebook PDFs for a =specific game.")
parser.add_argument(
    "--game", required=True, help="Name of the game (e.g., 'monopoly', 'dnd')")
parser.add_argument(
    "--target", choices=["pinecone", "chroma"], default="pinecone",
    help="Vectorstore target")
parser.add_argument(
    "--batch_size", required=False, default=200,
    help="Batch size for indexing (default: 200)")

args = parser.parse_args()

game = args.game
target = args.target
namespace = game
raw_path = os.path.join("data", "raw", game)
persist_path = os.path.join("data", "vectorstore", game)

if not os.path.isdir(raw_path):
    raise FileNotFoundError(f"No folder found at: {raw_path}")

print(f"\nIndexing PDFs from '{raw_path}' using {target.capitalize()}...")

index_pdfs(
    raw_dir=raw_path,
    persist_dir=persist_path,
    use_pinecone=(target == "pinecone"),
    namespace=namespace,
    config=config,
    batch_size=int(args.batch_size),
)
