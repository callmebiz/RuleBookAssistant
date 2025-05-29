import argparse
import json
from config.config import load_environment
load_environment()

from langchain_openai import ChatOpenAI
from rag.indexing import load_vectorstore
from rag.query_translation import QueryTranslator
from rag.query_construction import get_prompt
from langsmith import traceable
from rag.tracing import (
    traced_translate,
    traced_retrieve,
    traced_construct_prompt,
    traced_generate,
)

import mlflow
from rag.ml_tracking import (
    start_experiment,
    log_pipeline_params,
    log_pipeline_metrics,
    log_artifacts
)


@traceable(name="RAG End-to-End")
def run_pipeline(question: str, strategy: str, use_pinecone: bool, namespace: str) -> str:
    start_experiment()

    with mlflow.start_run():
        model = "gpt-4o-mini"
        llm = ChatOpenAI(model=model)
        translator = QueryTranslator(llm, strategy=strategy)
        prompt = get_prompt()
        vectorstore = load_vectorstore(namespace=namespace, persist_dir="data/vectorstore", use_pinecone=use_pinecone)
        retriever = vectorstore.as_retriever()

        log_pipeline_params({
            "question": question,
            "strategy": strategy,
            "vectorstore": "pinecone" if use_pinecone else "chroma",
            "namespace": namespace,
            "llm_model": model
        })

        queries = traced_translate(translator, question)
        if isinstance(queries, str):
            queries = [queries]

        docs = traced_retrieve(retriever, queries)
        context = traced_construct_prompt(docs)
        response = traced_generate(prompt, llm, context, question)

        log_pipeline_metrics({
            "num_queries": len(queries),
            "num_docs": len(docs),
            "response_length": len(response)
        })

        log_artifacts(response, context)

        return response

def get_supported_games():
    with open('config/supported_games.json', 'r') as file:
        supported_games = json.load(file)
        supported_games = [game["abbr"] for _, game in supported_games.items()]
    return supported_games        

def create_parser(supported_games):
    parser = argparse.ArgumentParser(description="Run a RAG pipeline for a given game.")

    parser.add_argument(
        "-q", "--question",
        required=True,
        help="The user question to answer."
    )

    parser.add_argument(
        "-s", "--strategy",
        default="passthrough",
        choices=["passthrough", "multi_query", "rag_fusion", "hyde", "step_back", "decompose"],
        help="Query translation strategy to use."
    )

    parser.add_argument(
        "-t", "--target",
        default="pinecone",
        choices=["pinecone", "chroma"],
        help="Vectorstore backend to use."
    )

    parser.add_argument(
        "-n", "--namespace",
        required=True,
        choices=supported_games,
        help="The namespace (game name) for document retrieval.  See `config/supported_games.json` for supported game/namespace names."
    )
    
    return parser


if __name__ == "__main__":
    supported_games = get_supported_games()
    parser = create_parser(supported_games)
    args = parser.parse_args()

    answer = run_pipeline(
        question=args.question,
        strategy=args.strategy,
        use_pinecone=(args.target == "pinecone"),
        namespace=args.namespace
    )
    print(answer)
