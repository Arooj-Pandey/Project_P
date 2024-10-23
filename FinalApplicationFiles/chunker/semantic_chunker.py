from langchain_experimental.text_splitter import SemanticChunker
from embeddings import initialize_embeddings # embeddings file
from chunker import Chunker   

chunk_size = 1000  
chunk_overlap = 200 
recursive = True  
main_file_path =  r'D:\Projects\PIB Chatbot\Project_P\ScraperFiles\newFiles'

chunker_instance = Chunker(main_file_path, chunk_size, chunk_overlap, recursive)

chunker_instance.load_documents()

documents = chunker_instance.get_documents()
    
def semantic_chunker(document):
    text_splitter = SemanticChunker(initialize_embeddings(), breakpoint_threshold_type='standard_deviation')
    chunks = text_splitter.split_documents(documents) # return the chunks, to vector store 
    return chunks