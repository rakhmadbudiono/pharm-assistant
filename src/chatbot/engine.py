from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.prompts import PromptTemplate

from src.config import RETRIEVER_K, TEMPERATURE


def create_rag_chain(model, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    llm = model.get_llm(TEMPERATURE)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    system_template = """You are a helpful pharmaceutical assistant.
    If you need to diagnose, ask user relevant question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate(
        template=system_template, input_variables=["context", "question"]
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
