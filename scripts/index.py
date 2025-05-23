from config.config import load_environment
load_environment()

from rag.indexing import index_pdfs
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index rulebooks to a vectorstore.")
    parser.add_argument("--target", choices=["chroma", "pinecone"], default="chroma", help="Which vectorstore to use")
    parser.add_argument("--raw_dir", default="data/raw", help="Directory containing PDF files")
    parser.add_argument("--persist_dir", default="data/vectorstore", help="Chroma vectorstore directory (ignored for Pinecone)")
    parser.add_argument("--namespace", default="dnd", help="Pinecone namespace to index into")

    args = parser.parse_args()
    use_pinecone = args.target == "pinecone"

    print(f"\n📚 Indexing PDFs from '{args.raw_dir}' using {'Pinecone' if use_pinecone else 'Chroma'}...\n")
    index_pdfs(
        raw_dir=args.raw_dir,
        persist_dir=args.persist_dir if not use_pinecone else None,
        use_pinecone=use_pinecone,
        namespace=args.namespace
    )
