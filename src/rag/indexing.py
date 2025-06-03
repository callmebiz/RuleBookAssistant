import os
import re
import json
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from rag.config_schema import RulebookConfig

from config.config import load_config_from_file


def clean_text(text):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if len(line.strip()) > 1]
    text = '\n'.join(filtered_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


def upload_in_batches(
    documents,
    embedding,
    index_name: str,
    namespace: str,
    batch_size: int = 200
):
    total = len(documents)
    print(f"Uploading {total} documents to Pinecone "
          f"in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = documents[i:i+batch_size]
        print(f"\t→ Uploading batch {i // batch_size + 1} "
              f"({len(batch)} docs)...")
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embedding,
            index_name=index_name,
            text_key="text",
            namespace=namespace
        )


def index_pdfs(raw_dir: str, namespace: str, config: RulebookConfig,
               batch_size: int = 200, persist_dir: Optional[str] = None,
               use_pinecone: bool = False):
    loader = DirectoryLoader(raw_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                              chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()

    if use_pinecone:
        if not config.pinecone_index_name:
            raise ValueError(
                "pinecone_index_name is required when use_pinecone=True")
        upload_in_batches(
            documents=chunks,
            embedding=embeddings,
            index_name=config.pinecone_index_name,
            namespace=namespace,
            batch_size=batch_size
        )
        print("All batches uploaded successfully.")
        return

    if not persist_dir:
        raise ValueError("persist_dir is required when use_pinecone=False")

    print(f"Indexing to local Chroma at '{persist_dir}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.path.join(persist_dir, namespace)
    )
    print("Chroma indexing complete.")
    return vectorstore


def load_vectorstore(namespace: str, persist_dir: Optional[str] = None,
                     use_pinecone: bool = False):
    config = load_config_from_file()
    embeddings = OpenAIEmbeddings(api_key=config.openai_api_key)

    if use_pinecone:
        return PineconeVectorStore(
            index_name=config.pinecone_index_name,
            text_key="text",
            namespace=namespace,
            embedding=embeddings,
            pinecone_api_key=config.pinecone_api_key
        )

    if not persist_dir:
        raise ValueError("persist_dir is required when use_pinecone=False")

    return Chroma(
        persist_directory=os.path.join(persist_dir, namespace),
        embedding_function=embeddings
    )
