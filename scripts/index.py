import argparse
import os
from config.config import load_environment
from rag.indexing import index_pdfs

load_environment()

parser = argparse.ArgumentParser(description="Index rulebook PDFs for a specific game.")
parser.add_argument("--game", required=True, help="Name of the game (e.g., 'monopoly', 'dungeons_and_dragons')")
parser.add_argument("--target", choices=["pinecone", "chroma"], default="pinecone", help="Vectorstore target")

args = parser.parse_args()

game = args.game
target = args.target
namespace = game
raw_path = os.path.join("data", "raw", game)
persist_path = os.path.join("data", "vectorstore", game)

if not os.path.isdir(raw_path):
    raise FileNotFoundError(f"No folder found at: {raw_path}")

print(f"\n📚 Indexing PDFs from '{raw_path}' using {target.capitalize()}...")

index_pdfs(
    raw_dir=raw_path,
    persist_dir=persist_path,
    use_pinecone=(target == "pinecone"),
    namespace=namespace
)
