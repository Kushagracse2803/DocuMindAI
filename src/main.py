import os
from dotenv import load_dotenv
from ingestion import load_and_split_pdf
from vector_db import store_chunks_in_chroma
from generator import create_rag_chain

# Load environment variables (API Keys)
load_dotenv()

def main():
    # 1. Path to your PDF file (Make sure you have a 'data' folder)
    file_path = "data/your_document.pdf"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print("--- Phase 1: Ingesting Data ---")
    chunks = load_and_split_pdf(file_path)
    print(f"Created {len(chunks)} chunks.")

    print("\n--- Phase 2: Vectorizing & Storing ---")
    vector_db = store_chunks_in_chroma(chunks)
    
    # Define Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    print("\n--- Phase 3: Setting up RAG Chain ---")
    rag_chain = create_rag_chain(retriever)

    # 4. User Interaction Loop
    print("\n--- Nexus RAG is ready! Type 'exit' to quit. ---")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == 'exit':
            break
        
        # Invoke the chain
        response = rag_chain.invoke(query)
        print(f"\nAI Answer: {response}")

if __name__ == "__main__":
    main()