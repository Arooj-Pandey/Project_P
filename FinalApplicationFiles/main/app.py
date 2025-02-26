import streamlit as st
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from FinalApplicationFiles.embeddings.embeddings import initialize_embeddings
from FinalApplicationFiles.VectorDB.pinecone_setup import PineconeVectorDatabase
from FinalApplicationFiles.LLMs.llm import initialize_llm, create_rag_chain

# Load environment variables
load_dotenv()

# Fetch API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Pinecone API key
gemini_api_key = os.getenv("GOOGLE_API_KEY")  # Google Generative AI key
PCVD = PineconeVectorDatabase(pinecone_api_key, "pib", initialize_embeddings())
PCVD.setup_pinecone()
# Initialize embeddings and Pinecone only once (to avoid reloading on every interaction)
@st.cache_resource
def initialize_system():
    embeddings = initialize_embeddings()
    #vector_store = PCVD.setup_pinecone()
    retriever = PCVD.retrieve_documents()
    llm = initialize_llm(gemini_api_key)
    rag_chain = create_rag_chain(retriever, llm)
    return rag_chain, retriever

def main():
    st.title("Ask Questions using RAG (Retrieval-Augmented Generation)")

    # Initialize the system (embeddings, retriever, LLM, RAG chain)
    rag_chain, chunk = initialize_system()

    # User input for questions
    question = st.text_input("Enter your question:")

    # If the 'Ask' button is pressed, generate an answer
    if st.button("Ask"):
        if question:
            # Use the RAG chain to generate an answer
            with st.spinner("Generating answer..."):
                answer = rag_chain.invoke(question)
                
                st.success("Answer generated!")

                # Display the question and answer using markdown
                st.markdown(f"### Question: \n{question}")
                st.markdown(f"### Answer: \n{answer}")
                st.markdown(f"### Answer: \n{chunk.invoke(question)}")

        else:
            st.warning("Please enter a question.")

# Streamlit app execution
if __name__ == "__main__":
    main()
