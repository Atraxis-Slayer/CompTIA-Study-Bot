import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# These are the imports that usually cause the ModuleNotFoundError if 'langchain' is missing
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- PAGE CONFIG ---
st.set_page_config(page_title="CompTIA A+ Study Bot", layout="wide")

# --- 1. SECURE API KEY ---
# Looks for secret in Streamlit Cloud settings or local .env
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("API Key not found. Please add OPENAI_API_KEY to your Streamlit Secrets.")
    st.stop()

# --- 2. DATA LOADING & VECTOR DB ---
PDF_FILES = [
    "comptia-a-220-1201-exam-objectives-(2-0).pdf",
    "comptia-a-220-1202-exam-objectives-(2-0).pdf"
]
PERSIST_DIR = "chroma_db"

@st.cache_resource
def build_vectorstore():
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    for pdf_path in PDF_FILES:
        if not os.path.exists(pdf_path):
            st.warning(f"File not found: {pdf_path}. Ensure it is uploaded to GitHub.")
            continue
            
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split and add source metadata
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source_file"] = pdf_path
        
        all_chunks.extend(chunks)

    if not all_chunks:
        st.error("No PDFs loaded. Please check your file names in the GitHub repository.")
        st.stop()
        
    return Chroma.from_documents(
        documents=all_chunks,
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
        persist_directory=PERSIST_DIR
    )

# Initialize database
vectorstore = build_vectorstore()

# --- 3. CONVERSATIONAL AI LOGIC ---
@st.cache_resource
def build_study_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 1. Contextualize question (handles follow-up questions)
    context_q_system_prompt = "Given a chat history and the latest user question, formulate a standalone question."
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    # 2. Main Tutor Prompt
    qa_system_prompt = """You are a CompTIA A+ Lead Instructor.
    Use the context provided to answer questions about the 220-1201 and 220-1202 exams.
    
    - If the context is from '1201', it's Core 1 (Hardware/Network).
    - If the context is from '1202', it's Core 2 (Software/Security).
    - Be technical but explain concepts simply.
    - End with a one-sentence practice challenge.

    Context:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # 3. Combine into final chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

study_chain = build_study_chain()

# --- 4. STREAMLIT UI ---
st.title("🛡️ CompTIA A+ Expert Tutor")
st.caption("Covers Core 1 (220-1201) and Core 2 (220-1202)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What is the laser printing process?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Consulting objectives..."):
            response = study_chain.invoke({
                "input": prompt, 
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            st.markdown(answer)
            
            # Show Sources
            with st.expander("View Reference Material"):
                for doc in response["context"]:
                    st.write(f"**From:** {doc.metadata.get('source_file')}")
                    st.caption(doc.page_content[:300] + "...")

    # Save History
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
