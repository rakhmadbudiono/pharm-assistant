import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, VECTOR_STORE_DIR

VECTOR_STORE_PATH = VECTOR_STORE_DIR / "faiss_index"


class VectorStoreManager:
    def __init__(self):
        self.vector_store = None

    def load_index(self, embeddings):
        if VECTOR_STORE_PATH.exists():
            self.vector_store = FAISS.load_local(
                str(VECTOR_STORE_PATH), embeddings, allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None

    def save_index(self):
        self.vector_store.save_local(str(VECTOR_STORE_PATH))

    def create_index(self, documents, embeddings):
        self.vector_store = FAISS.from_documents(documents, embeddings)

    def process_documents(self, uploaded_files, embeddings):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        all_chunks = []

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.name).suffix
            ) as tmp:
                tmp.write(file.getbuffer())
                temp_path = Path(tmp.name)

            try:
                loader = (
                    PyPDFLoader(str(temp_path))
                    if file.name.endswith(".pdf")
                    else TextLoader(str(temp_path))
                )
                docs = loader.load()
                chunks = text_splitter.split_documents(docs)
                for chunk in chunks:
                    chunk.metadata["source"] = file.name
                all_chunks.extend(chunks)
            finally:
                temp_path.unlink()

        if self.vector_store is None:
            self.create_index(all_chunks, embeddings)
        else:
            self.vector_store.add_documents(all_chunks)

        self.save_index()
        st.success(f"Processed {len(uploaded_files)} file(s)")
