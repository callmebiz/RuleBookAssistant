from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use only the provided context to answer the question. Cite page numbers if possible."),
        ("user", "Question: {question}\n\nContext:\n{context}")
    ])


def format_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)
