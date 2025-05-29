# src/rag/tracing.py

from langsmith import traceable
from src.rag.retrieval import retrieve_documents
from src.rag.query_construction import format_doc
from src.rag.generation import generate_response


@traceable(name="Translate Query")
def traced_translate(translator, query):
    return translator.translate(query)


@traceable(name="Retrieve Documents")
def traced_retrieve(retriever, queries, top_k=4):
    return retrieve_documents(retriever, queries, top_k)


@traceable(name="Construct Prompt")
def traced_construct_prompt(docs):
    context = "\n\n".join(format_doc(doc) for doc in docs)
    return context


@traceable(name="Generate Response")
def traced_generate(prompt, llm, context, question):
    return generate_response(prompt, llm, context, question)
