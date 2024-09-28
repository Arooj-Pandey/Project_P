# Import necessary libraries
from dotenv import load_dotenv
import os
from embeddings import initialize_embeddings
from pinecone_setup import setup_pinecone, retrieve_documents
from llm import initialize_llm, create_rag_chain

# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Pinecone API key
gemini_api_key = os.getenv("GEMINI_API_KEY")  # Google Generative AI key

def main():
    # Initialize embeddings model
    embeddings = initialize_embeddings()

    # Setup Pinecone connection and index
    index_name = "pib"
    vector_store = setup_pinecone(pinecone_api_key, index_name, embeddings)

    # Retrieve documents from the vector store (no vector addition)
    retriever = retrieve_documents(vector_store)

    # Initialize the Google Generative AI model
    llm = initialize_llm(gemini_api_key)

    # Create the RAG chain (Retrieval-Augmented Generation)
    rag_chain = create_rag_chain(retriever, llm)

    # Loop to ask multiple questions
    while True:
        question = input("Enter your question (or type 'exit' to stop): ")

        if question.lower() == 'exit':
            print("Exiting the Q&A session.")
            break
        
        # Use the chain to generate an answer
        ans = rag_chain.invoke(question)
    

        # Print the generated answer
        print(f"Answer: {ans}")

        print("\n")  # Blank line between questions

if __name__ == "__main__":
    main()