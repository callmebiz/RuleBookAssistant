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

@traceable(name="RAG End-to-End")
def run_pipeline(question: str, strategy: str, use_pinecone: bool, namespace: str) -> str:

    # Init components
    llm = ChatOpenAI(model="gpt-4o-mini")
    translator = QueryTranslator(llm, strategy=strategy)
    prompt = get_prompt()
    vectorstore = load_vectorstore("data/vectorstore", use_pinecone=use_pinecone, namespace=namespace)
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
    question = "What are good plot hooks for a beginner campaign?"
    
    strategy = "multi_query"       # Options: original, multi_query, rag_fusion, hyde, step_back, decompose
    use_pinecone = True            # Toggle vectorstore backend. True = Pinecone, False = Chroma.
    namespace = "dnd"              # Game-specific document group

    answer = run_pipeline(
        question=question,
        strategy=strategy,
        use_pinecone=use_pinecone,
        namespace=namespace
    )
    print("\n--- FINAL RESPONSE ---\n")
    print(answer)
