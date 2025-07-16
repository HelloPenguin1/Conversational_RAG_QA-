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
from langchain_core.chat_history import BaseChatMessageHistory

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
        st.session_state.store={}

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

        #################################################################################################################

        reformulation_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood"
            "without the chat histor. DO NOT answer the question"
            "just reformulate it if needed and otherwise return it as is"
        )

        reformulation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", reformulation_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retrieval(llm, retriever, reformulation_prompt)
        
        ###################################################################################################################

        ##Answer question prompt
        answer_system_prompt = (
            "You are an assistant for question answering tasks"
            "Please answer the question as accurately and informatively"
            "possible using the following retrieved context. In the minimum (unless specified by the question)"
            "give a three to four line answer to the question."
            "\n\n {context}"
        )

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", answer_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
        rag_pipeline = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        ###################################################################################################################

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]


        conversational_rag_chain = RunnableWithMessageHistory(
            rag_pipeline, 
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        ###################################################################################################################
        
        user_input = st.text_input("Please enter you question: ")

        if user_input:
            session_history= get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {"input": user_input},

                config = {
                    "configurable":{"session_id":session_id}
                }
            )

            st.write(st.session_state.store)
            st.success("Assistant:", response['answer'])
            st.write("Chat History: ", session_history.messages)

else:
    st.warning("Please enter your GROQ API Key")


        
        



