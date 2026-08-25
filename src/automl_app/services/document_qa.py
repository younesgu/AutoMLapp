"""Document question-answering helpers."""

import os

from dotenv import load_dotenv

load_dotenv()


def answer_document(file_path: str, query: str) -> str:
    """Build a temporary vector store from a PDF and answer a question."""
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is not configured. Add it to your environment."
        )

    from langchain import HuggingFaceHub
    from langchain.chains import RetrievalQA
    from langchain.document_loaders import PyPDFLoader
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import SKLearnVectorStore

    pages = PyPDFLoader(file_path).load_and_split()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=64
    ).split_documents(pages)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_store = SKLearnVectorStore.from_documents(chunks, embedding=embeddings)
    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.1, "max_length": 512},
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    )
    return qa.invoke({"query": query})["result"]
