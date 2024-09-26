import streamlit as st
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv #to hide the api keys also, .env ko then .gitignore me add kra taaki git mei push naa ho jaaye
import os

load_dotenv()

# Pinecone and LLM configuration
pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Add your Pinecone API key 
index_name = "test3"  # Pinecone index name
google_api_key = os.getenv("GOOGLE_API_KEY")  # Add your Google Generative AI key

# Load Pinecone vector store
def load_vector_store(): 
    # Create Pinecone instance
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    
    # Generate embeddings using HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store

# Initialize LLM (Google Generative AI)
def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.0-pro",
        google_api_key=google_api_key,
        temperature=0.01,
    )
    return llm 

# Create a custom RAG (Retrieve and Generate) chain
def create_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6}, threshold=0.7)
    
    # Define prompt for RAG
    prompt_template = """You are an expert document expalainer, and excel in explaining the the overall detailed summary of any question asked by the UPSC aspirants. Brief the important points based on the question, while keeping the details you find associated with the question, give the user a brief but detailed explanation of the answer.
    
    {context}
    
    Question: {question}
    
    Answer:"""
    
    custom_rag_prompt = PromptTemplate.from_template(prompt_template)
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Format retrieved documents
def format_docs(docs):
    return ''.join([doc.page_content for doc in docs]) #  Format the retrieved documents into a single string, page_content meta data eliminate kr deta hai

# Streamlit interface
st.title("Chat with Your Database")

# Load Pinecone vector store and LLM
vector_store = load_vector_store()
llm = load_llm()
rag_chain = create_rag_chain(vector_store, llm)

# Input for the user's query
user_input = st.text_input("Ask a question:")

# Process the user's query
if st.button("Submit"):
    if user_input:
        response = rag_chain.invoke(user_input)
        st.write("Answer:", response)
    else:
        st.write("Please enter a question.")
