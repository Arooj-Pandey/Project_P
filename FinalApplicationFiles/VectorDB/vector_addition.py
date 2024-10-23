from dotenv import load_dotenv
import os
from VectorDB.pinecone_setup import PineconeVectorDatabase
from Project_P.FinalApplicationFiles.embeddings.embeddings import initialize_embeddings

"""
This function takes in a list of Chunks and adds them to the Pinecone index. If the index does not exist, it creates a new Pinecone index.
"""

# Load the Pinecone API key from the .env file
load_dotenv()


api_key = os.getenv("PINECONE_API_KEY")
embeddings = initialize_embeddings()
name = "pib"

pineconedb = PineconeVectorDatabase(api_key=api_key, index_name=name, embeddings=embeddings)

pineconedb._setup_pinecone()

def pinecone_add_documents(chunks):
    """
    Adds the provided chunks to the Pinecone index.
    """
    pineconedb.add_documents(chunks)
    print("Documents added to Pinecone index.")