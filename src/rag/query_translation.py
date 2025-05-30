# src/rag/query_translation.py

from typing import List, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json


class QueryTranslator:
    def __init__(self, llm: BaseLanguageModel, strategy: str = "passthrough"):
        self.llm = llm
        self.strategy = strategy.lower()
        self.output_parser = StrOutputParser()
        self.prefix = "You are an expert helper assistant on the rules and mechanics of tabletop games. "
        self.postfix = " write this postfix: \"Cite references (e.g., document name, chapters, pages, quotes, etc.)\", at the end of the line."
    
    def translate(self, query: str) -> Union[str, List[str]]:
        if self.strategy == "passthrough":
            return query
        elif self.strategy == "multi_query":
            return self._multi_query(query)
        elif self.strategy == "rag_fusion":
            return self._multi_query(query)  # Retrieval fusion will use these
        elif self.strategy == "hyde":
            return self._hyde_query(query)
        elif self.strategy == "step_back":
            return self._step_back_query(query)
        elif self.strategy == "decompose":
            return self._decompose_query(query)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _multi_query(self, query: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prefix + "Reformulate user question(s) for better document retrieval."),
            ("user", "Original question: {question}\nGenerate 4 rephrasings, line by line. For each rephrasing," + self.postfix)
        ])
        chain = prompt | self.llm | self.output_parser
        result = chain.invoke({"question": query})
        return [q.strip("- ").strip() for q in result.split("\n") if q.strip()]
    
    def _hyde_query(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prefix + "Given a question, write a hypothetical answer."),
            ("user", "{question}")
        ])
        chain = prompt | self.llm | self.output_parser
        llm_answer = chain.invoke({"question": query})
        return llm_answer.strip()
    
    def _step_back_query(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prefix + "Generalize the question(s)."),
            ("user", "Specific question: {question}\nWhat is a more general version?" + self.postfix)
        ])
        chain = prompt | self.llm | self.output_parser
        broader_question = chain.invoke({"question": query})
        return broader_question.strip()
    
    def _decompose_query(self, query: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prefix + "Break down complex question(s) into sub-questions."),
            ("user", "Complex question: {question}\nList 3 sub-questions, one line at a time. For each sub-questions," + self.postfix)
        ])
        chain = prompt | self.llm | self.output_parser
        result = chain.invoke({"question": query})
        return [q.strip("- ").strip() for q in result.split("\n") if q.strip()]
