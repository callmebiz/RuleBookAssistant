from typing import List
from langchain_core.documents import Document


def retrieve_documents(retriever, queries: List[str], top_k: int = 4) -> List[Document]:
    """
    Supports single or multi-query retrieval using retriever.invoke() or retriever.batch().
    Deduplicates and limits to top_k.
    """
    if len(queries) == 1:
        docs = retriever.invoke(queries[0])
    else:
        all_results = retriever.batch(queries)
        docs = [doc for result in all_results for doc in result]

    # Optional deduplication by content
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs[:top_k]
