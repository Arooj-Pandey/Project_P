from langchain_huggingface import HuggingFaceEmbeddings

def initialize_embeddings():
    """
    Initializes and returns the HuggingFace embeddings model.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")


