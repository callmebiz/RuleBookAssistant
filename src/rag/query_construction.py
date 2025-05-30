from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import os


PROMPT = """
You are a helpful assistant that answers questions about game mechanics and rules using only the provided context.

• Never refer to "the context" directly in your answer.
• If citing sources, include the document filename (e.g., "rules.pdf") — never show full file paths or directories.
• When possible, cite specific page numbers, sections, or quotes from the source.
• If the context does not contain relevant information, clearly say so rather than guessing or making up answers.
• Your goal is to be accurate, concise, and grounded strictly in the provided material.
"""


def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", (PROMPT)
        ),
        ("user", "Question: {question}\n\nContext:\n{context}")
    ])

def format_doc(doc):
    source = doc.metadata.get("source", "unknown")
    source = os.path.basename(source) if is_path(source) else source
    page = doc.metadata.get("page", "unknown")
    return f"[{source}, page {page}]\n{doc.page_content.strip()}"

def is_path(s):
    path = Path(s)
    return path.suffix != "" or len(path.parts) > 1
