import streamlit as st
from dotenv import load_dotenv

from src.chatbot.engine import create_rag_chain
from src.config import (
    EMBEDDING_MODEL_PROVIDER,
    MODEL_PROVIDER,
)
from src.knowledge_base.vectorstore import VectorStoreManager
from src.models.gemini_model import gemini_model
from src.models.hf_model import hf_model
from src.models.openai_model import openai_model

load_dotenv()


def get_model():
    if MODEL_PROVIDER == "openai":
        return openai_model
    if MODEL_PROVIDER == "hugging-face":
        return hf_model
    return gemini_model


def get_model_embedding():
    if EMBEDDING_MODEL_PROVIDER == "openai":
        return openai_model.get_embeddings()
    if EMBEDDING_MODEL_PROVIDER == "gemini":
        return gemini_model.get_embeddings()
    return hf_model.get_embeddings()


def get_embeddings():
    if "embeddings" not in st.session_state or st.session_state.embeddings is None:
        if model.is_configured():
            st.session_state.embeddings = get_model_embedding()
    return st.session_state.embeddings


if "messages" not in st.session_state:
    st.session_state.messages = []

model = get_model()
vector_store_manager = VectorStoreManager()

if model.is_configured():
    embeddings = get_embeddings()
    if embeddings and vector_store_manager.vector_store is None:
        vector_store_manager.load_index(embeddings)


st.title("RAG Chatbot")

with st.sidebar:
    st.header("Current Knowledge Base")

    if vector_store_manager.vector_store is not None:
        doc_metadata = vector_store_manager.vector_store.docstore._dict.values()
        sources = sorted(
            list(set([doc.metadata.get("source", "Unknown") for doc in doc_metadata]))
        )

        if sources:
            for source in sources:
                st.info(f"📄 {source}")
        else:
            st.write("No documents in index.")
    else:
        st.write("Knowledge base is empty.")

    st.divider()

    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Process"):
        if not model.is_configured():
            api_key_name = (
                "GOOGLE_API_KEY" if MODEL_PROVIDER == "gemini" else "OPENAI_API_KEY"
            )
            st.error(f"Please set {api_key_name} in .env file")
        else:
            with st.spinner("Processing..."):
                embeddings = get_embeddings()
                vector_store_manager.process_documents(uploaded_files, embeddings)

if not model.is_configured():
    api_key_name = "GOOGLE_API_KEY" if MODEL_PROVIDER == "gemini" else "OPENAI_API_KEY"
    st.warning(f"Please set {api_key_name} in .env file")
elif not vector_store_manager.vector_store:
    st.info("Upload documents to start chatting")
else:
    if "chain" not in st.session_state:
        st.session_state.chain = create_rag_chain(
            model, vector_store_manager.vector_store
        )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain({"question": prompt})
                answer = result["answer"]
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
