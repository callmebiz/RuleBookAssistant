from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def generate_response(prompt, llm, context: str, question: str) -> str:
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})
