import streamlit as st
import os
from dotenv import load_dotenv
from src.ingestion import load_and_split_pdf
from src.vector_db import store_chunks_in_chroma
from src.retrieval import get_retriever
from src.generator import create_rag_chain

load_dotenv()

st.title("Nexus RAG Assistant 🚀")

# 1. Initialize session state agar wo pehle se nahi hai
if "chain" not in st.session_state:
    st.session_state.chain = None

pdf_path = "data/llm_handbook.pdf"

if st.button("Initialize Bot"):
    with st.spinner("Processing PDF..."):
        # Yahan tumhara logic
        chunks = load_and_split_pdf(pdf_path)
        vdb = store_chunks_in_chroma(chunks)
        retriever = get_retriever(vdb)
        
        # Chain ko session_state mein store karo
        st.session_state.chain = create_rag_chain(retriever)
        st.success("Bot is ready!")

# 2. Check karo ki chain exist karti hai ya nahi
query = st.text_input("Ask a question:")
if query:
    if st.session_state.chain is not None:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(query)
            st.write(response)
    else:
        st.warning("Please click 'Initialize Bot' first!")