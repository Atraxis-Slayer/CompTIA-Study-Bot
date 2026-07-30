import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="CompTIA A+ Study Coach", layout="wide")

# --- 1. SECURE API KEY ---
# On local: set environment variable or use .env
# On Streamlit Cloud: Use "Settings > Secrets"
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("Please provide an OpenAI API Key to continue.")
    st.stop()

# --- 2. CONFIGURATION ---
PDF_PATH = "CompTIA_A_Plus_Objectives.pdf"
PERSIST_DIR = "chroma_db"

@st.cache_resource
def build_vectorstore():
    if not os.path.exists(PDF_PATH):
        st.error(f"Missing {PDF_PATH}! Please upload it to the directory.")
        st.stop()
        
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    
    return Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
        persist_directory=PERSIST_DIR
    )

vectorstore = build_vectorstore()

# --- 3. CONVERSATIONAL CHAIN ---
@st.cache_resource
def build_study_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # This handles "condensing" the chat history into a single question for the search
    contextualize_q_system_prompt = "Given a chat history and the latest user question, formulate a standalone question."
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Actual Study Tutor Prompt
    qa_system_prompt = """You are an expert CompTIA A+ Lead Instructor. 
    Use the following context to guide your answer, but use your broad IT knowledge to explain concepts in detail.
    
    Rules:
    1. Explain technical terms simply (analogy-based).
    2. If a question is about troubleshooting, mention the 'CompTIA 6-Step Troubleshooting Process'.
    3. End your answer with a 'Quick Test Question' based on your explanation.

    Context:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

study_chain = build_study_chain()

# --- 4. UI AND CHAT HISTORY ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🛡️ CompTIA A+ (220-1201/1202) Study Bot")

# Display Chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Port Numbers, RAID, Malware removal..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = study_chain.invoke({
            "input": prompt, 
            "chat_history": st.session_state.chat_history
        })
        answer = response["answer"]
        st.markdown(answer)
        
        with st.expander("See Study Sources"):
            for doc in response["context"]:
                st.write(f"Page {doc.metadata['page']}: {doc.page_content[:200]}...")

    # Update History
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
