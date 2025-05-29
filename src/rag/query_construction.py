from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant that answers users questions about game mechanics and details, "
            "using only the provided context. Do not make direct reference to 'the context' unless citing. "
            "Cite references (document name, chapters, pages, quotes, etc.)."
            )
        ),
        ("user", "Question: {question}\n\nContext:\n{context}")
    ])

def format_doc(doc):
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "unknown")
    return f"[{source}, page {page}]\n{doc.page_content.strip()}"
