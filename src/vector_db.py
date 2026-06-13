from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def store_chunks_in_chroma(chunks):
    # 1. Embedding Model download karo
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Vector Store banao aur data load karo
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db" # Ye data ko folder mein save karega
    )
    return vector_db