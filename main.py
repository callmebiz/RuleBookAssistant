from config.config import load_environment
load_environment()

from langchain_openai import ChatOpenAI
from src.rag.indexing import load_vectorstore
from src.rag.query_translation import QueryTranslator
from src.rag.query_construction import get_prompt
from langsmith import traceable


@traceable(name="RAG End-to-End")
def run_pipeline(question: str, strategy: str = "original") -> str:
    from src.rag.tracing import (
        traced_translate,
        traced_retrieve,
        traced_construct_prompt,
        traced_generate,
    )

    # Initialize components
    llm = ChatOpenAI(model="gpt-4o-mini")
    translator = QueryTranslator(llm, strategy=strategy)
    prompt = get_prompt()
    vectorstore = load_vectorstore("data/vectorstore")
    retriever = vectorstore.as_retriever()

    # Translate query
    queries = traced_translate(translator, question)
    if isinstance(queries, str):
        queries = [queries]

    # Retrieve documents
    docs = traced_retrieve(retriever, queries)

    # Construct context from retrieved docs
    context = traced_construct_prompt(docs)

    # Generate final response
    response = traced_generate(prompt, llm, context, question)
    return response


if __name__ == "__main__":
    sample_question = "What is the process for building a balanced encounter for a party of 5th-level characters?"
    strategy = "rag_fusion"  # Options: original, multi_query, rag_fusion, hyde, step_back, decompose

    answer = run_pipeline(sample_question, strategy=strategy)
    print("\n--- FINAL RESPONSE ---\n")
    print(answer)
