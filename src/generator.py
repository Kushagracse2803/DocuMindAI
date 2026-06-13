from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Groq model initialize karna
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

def create_rag_chain(retriever):
    # Prompt Template (Ye batata hai ki bot ko behave kaise karna hai)
    template = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Keep the answer concise.
    
    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
    
    # LCEL Chain (Ye poore process ko connect karta hai)
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain