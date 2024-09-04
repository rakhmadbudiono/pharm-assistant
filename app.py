__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def load_and_split_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

def initialize_vectorstore(splits, embedding_model, api_key):
    embedding_function = OpenAIEmbeddings(model=embedding_model, openai_api_key=api_key)
    return Chroma.from_documents(documents=splits, embedding=embedding_function)

def create_prompt_template():
    system_prompt = (
        "Kamu adalah asisten apoteker untuk tugas tanya-jawab. "
        "Gunakan bagian dari konteks yang diambil untuk menjawab "
        "pertanyaan. Jika kamu tidak tahu jawabannya, katakan bahwa kamu "
        "tidak tahu. Gunakan maksimal tiga kalimat dan jawab secara singkat."
        "\n\n"
        "{context}"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

def initialize_chains(retriever, llm):
    prompt = create_prompt_template()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

def main():
    file_path = "farmakologi.pdf"
    splits = load_and_split_document(file_path)
    
    vectorstore = initialize_vectorstore(splits, "text-embedding-3-small", os.getenv("OPENAI_SECRET_KEY"))
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_SECRET_KEY"))
    rag_chain = initialize_chains(retriever, llm)
    
    st.title("Pharmacist Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_input := st.chat_input("Pertanyaan:"):
        st.session_state.messages.append({"role": "human", "content": prompt_input})
        with st.chat_message("human"):
            st.markdown(prompt_input)

        res = vectorstore.similarity_search_with_score(prompt_input, k=1)
        context = "\n\n---\n\n".join([doc.page_content for doc, _score in res])
        response = rag_chain.invoke({"input": prompt_input, "context": context})

        with st.chat_message("system"):
            st.markdown(response["answer"])

        st.session_state.messages.append({"role": "system", "content": response["answer"]})

if __name__ == "__main__":
    main()
