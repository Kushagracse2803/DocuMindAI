from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path):
    # 1. PDF load karna
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Splitter define karna
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Har chunk mein 1000 characters honge
        chunk_overlap=200   # 200 characters ka overlap (context maintain karne ke liye)
    )
    
    # 3. Documents ko chunks mein convert karna
    chunks = text_splitter.split_documents(documents)
    return chunks