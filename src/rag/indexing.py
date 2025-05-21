from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import os
import re


def clean_text(text):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if len(line.strip()) > 1]
    text = '\n'.join(filtered_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


def index_pdfs(raw_dir: str, persist_dir: str):
    loader = DirectoryLoader(raw_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_dir
    )
    return vectorstore


def load_vectorstore(persist_dir: str):
    return Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_dir
    )
