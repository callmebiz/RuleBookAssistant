"""
main.py - CLI entry point for RuleBook Assistant

This script runs the end-to-end RAG pipeline for answering questions about tabletop game rulebooks.
It supports multiple query strategies and vector backends, and logs metrics/artifacts for each run.
"""
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
from rag.langchain_callback import UsageTrackingCallback

import mlflow
from rag.ml_tracking import (
    start_experiment,
    log_pipeline_params,
    log_pipeline_metrics,
    log_artifacts
)


@traceable(name="RAG End-to-End")
def run_pipeline(question: str, strategy: str, use_pinecone: bool, namespace: str, supported_games: dict) -> str:
    """
    Runs the full RAG pipeline: query translation, retrieval, prompt construction, and response generation.
    Logs metrics and artifacts for each run.

    Args:
        question (str): The user's question.
        strategy (str): Query translation strategy.
        use_pinecone (bool): Whether to use Pinecone or Chroma as the vectorstore backend.
        namespace (str): The namespace (game abbreviation) for document retrieval.
        supported_games (dict): Mapping of supported games and their abbreviations.

    Returns:
        str: The generated answer from the LLM.
    """
    # Get the full name of the game being queried
    game = "N/A"
    for _game, details in supported_games.items():
        if details['abbr'] == namespace:
            game = _game
            break
    # Add game context to the question for LLM, but instruct not to mention the game name
    game_context = f"\n[The question is pertaining to the game '{game}'. Do not mention the game name.]\n"
    
    start_experiment()
    with mlflow.start_run():
        callback = UsageTrackingCallback()
        model = "gpt-4o-mini"
        # Initialize LLM with callback for usage tracking
        llm = ChatOpenAI(model=model, temperature=0, callbacks=[callback])
        translator = QueryTranslator(llm, strategy=strategy)
        prompt = get_prompt()
        # Load the appropriate vectorstore (Pinecone or Chroma)
        vectorstore = load_vectorstore(namespace=namespace, persist_dir="data/vectorstore", use_pinecone=use_pinecone)
        retriever = vectorstore.as_retriever()

        # Log pipeline parameters to MLflow
        log_pipeline_params({
            "question": question,
            "strategy": strategy,
            "vectorstore": "pinecone" if use_pinecone else "chroma",
            "namespace": namespace,
            "llm_model": model
        })

        # Translate the question according to the chosen strategy
        queries = traced_translate(translator, question + game_context)
        if isinstance(queries, str):
            queries = [queries]

        # Retrieve relevant documents
        docs = traced_retrieve(retriever, queries)
        # Construct the context for the prompt
        context = traced_construct_prompt(docs)
        # Generate the final response
        response = traced_generate(prompt, llm, context, question)

        # Log metrics to MLflow
        log_pipeline_metrics({
            "num_queries": len(queries),
            "num_docs": len(docs),
            "response_length": len(response),
            "prompt_tokens": callback.prompt_tokens,
            "completion_tokens": callback.completion_tokens,
            "total_tokens": callback.total_tokens,
            "llm_calls": callback.calls
        })

        # Log artifacts (response, context, token details)
        log_artifacts(response, context, callback.completion_token_details)

        return response       


def create_parser(supported_games):
    """
    Creates an argument parser for the CLI.

    Args:
        supported_games (list): List of supported game abbreviations.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
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


def get_supported_games():
    """
    Loads the supported games and their abbreviations from config/supported_games.json.

    Returns:
        dict: Mapping of game names to their details (including abbreviation).
    """
    with open('config/supported_games.json', 'r') as file:
        supported_games = json.load(file)
    return supported_games 


if __name__ == "__main__":
    # Load supported games from config
    supported_games = get_supported_games()
    # Create CLI parser with supported game abbreviations
    parser = create_parser([game["abbr"] for _, game in supported_games.items()])
    args = parser.parse_args()

    # Run the pipeline and print the answer
    answer = run_pipeline(
        question=args.question,
        strategy=args.strategy,
        use_pinecone=(args.target == "pinecone"),
        namespace=args.namespace,
        supported_games=supported_games
    )
    print(answer)
