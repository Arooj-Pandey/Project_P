
# from langchain_pinecone import PineconeVectorStore

# from pinecone import Pinecone, ServerlessSpec

# class PineconeVectorDatabase:
#     def __init__(self, api_key, index_name, embeddings):
#         """
#         Initializes the PineconeVectorDatabase with the provided API key, index name, and embeddings.
#         Connects to Pinecone and initializes the specified index.
#         """
#         self.api_key = api_key
#         self.index_name = index_name
#         self.embeddings = embeddings
#         self.vector_store = None
#         self._setup_pinecone()

#     def _setup_pinecone(self):
#         """
#         Connects to Pinecone using the provided API key and initializes the specified index.
#         If the index does not exist, it returns a message indicating so.
#         Otherwise, it initializes the Pinecone vector store without adding vectors.
#         """
#         pc = Pinecone(api_key=self.api_key)
#         existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

#         if self.index_name not in existing_indexes:
#             # Create a new index if it doesn't exist
#             print("Index not exists")
            
        
#             pc.create_index(
#                 name=self.index_name,
#                 dimension=384,
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud="aws", region="us-east-1")
#             )

#         # Retrieve the created index
#         index = pc.Index(self.index_name)
#         # Initialize vector store without adding vectors
#         self.vector_store = PineconeVectorStore(index=index, embedding=self.embeddings)


#     def add_documents(self, docs):
#         """
#         Adds the provided documents to the vector store.
#         """
#         if not self.vector_store:
#             raise ValueError("Vector store is not initialized.")
#         self.vector_store.add_documents(docs)



#     def retrieve_documents(self):
#         try:
#             return self.vector_store.as_retriever()  # Example: Pinecone retriever
#         except Exception as e:
#             print(f"Error in retrieving documents: {e}")
#             return None


from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()
class PineconeVectorDatabase:
    def __init__(self, api_key, index_name, embeddings):
        """
        Initializes the PineconeVectorDatabase with the provided API key, index name, and embeddings.
        Connects to Pinecone and initializes the specified index.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.index_name = index_name
        self.embeddings = embeddings
        self.vector_store = None  # Initialize as None
        self.index = None  # Initialize as None
        self.setup_pinecone()

    def setup_pinecone(self):
        """
        Connects to Pinecone using the provided API key and initializes the specified index.
        If the index does not exist, it creates one. Otherwise, it connects to the existing index.
        """
    # Use the instance variables directly, no need for local variables
        pc = Pinecone(api_key=self.api_key)
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if self.index_name not in existing_indexes:
        # Create a new index if it doesn't exist
            print("Index does not exist. Creating index.")
            pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

    # Retrieve or create the index
        self.index = pc.Index(self.index_name)

    # Initialize vector store
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
    
        return self.vector_store  # Return for convenience

    def add_documents(self, docs):
        """
        Adds the provided documents to the vector store.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
        self.vector_store.add_documents(docs)

    def retrieve_documents(self):
        """
        Returns a retriever instance from the vector store.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
        try:
            return self.vector_store.as_retriever()  # Example: Pinecone retriever
        except Exception as e:
            print(f"Error in retrieving documents: {e}")
            return None
