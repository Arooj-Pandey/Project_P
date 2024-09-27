from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

def setup_pinecone(api_key, index_name, embeddings):
    """
    Sets up Pinecone by connecting with the provided API key, and initializes the index.
    Returns the initialized Pinecone vector store without adding vectors.
    """
    pc = Pinecone(api_key=api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        # Create a new index if it doesn't exist
        return "Index not exists"
    
    # Retrieve the created index
    index = pc.Index(index_name)
    # Initialize vector store without adding vectors
    return PineconeVectorStore(index=index, embedding=embeddings)

def retrieve_documents(vector_store):
    """
    Converts the vector store into a retriever with specific search criteria.
    Returns the retriever object.
    """
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.7},  # Search top 4 results with a threshold
    )
