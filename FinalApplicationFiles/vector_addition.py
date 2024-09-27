import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from embeddings import initialize_embeddings
from chunker import chunker

load_dotenv()
pinecone_api_key = os.getenv(PINECONE_API_KEY) # Use your API key
pc = Pinecone(api_key=pinecone_api_key)


def vector_addition(initialize_embeddings, index_name, chunker):
    index_name = index_name
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=initialize_embeddings)
    vector_store.add_documents(chunker)

    index = pc.Index(index_name)