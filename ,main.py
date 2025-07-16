import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retrieval
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


#set up streanlit
st.title("Coversational RAG with PDF Uploads and Chat History")
st.write("Upload PDF and chat with the content")

##Input the Groq API Key
api_key = st.text_input("Enter your Groq API Key here: ", type = "password")


if api_key:
    llm = ChatGroq(groq_api_key = api_key, model_name ="Gemma2-9b-It")

    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store[]

    uploaded_files = st.file_uploader("Choose a pdf file", type = "pdf", accept_multiple_files=False)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            ##store in local folder
            temppdf = f"./temp.pdf"

            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFDirectoryLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, 
                                                           chunk_overlap = 100)
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever()

contextualize_q_system_prompt = (
    "Given a "
)