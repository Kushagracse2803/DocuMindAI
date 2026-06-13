# src/retrieval.py

def get_retriever(vector_db):
    # ChromaDB se retriever banana
    retriever = vector_db.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3} 
    )
    return retriever