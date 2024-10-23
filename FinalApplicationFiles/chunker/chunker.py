import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.document_loaders import TextLoader
from Project_P.FinalApplicationFiles.embeddings.embeddings import initialize_embeddings


def chunker(Main_file_path, chunk_size = 1000 , chunk_overlap = 200, recursive = True):
    documents = []


    for i in os.listdir(Main_file_path):
        for j in os.listdir(os.path.join(Main_file_path, i)):
            for k in os.listdir(os.path.join(Main_file_path, i , j)):
                try:
                    if k.endswith('.txt'):
                        file_path = os.path.join(Main_file_path, i , j, k)
                        
                        # Create loader and load the document
                        loader = TextLoader(file_path, encoding='utf-8')
                        doc = loader.load()[0]
                        
                        # Add metadata (e.g., filename or custom data)
                       # doc.metadata = {"source": k, "Ministry": j, "Month":i}
                        doc.metadata = {"source": k, "Ministry": j} #removed month from meta data to shorten the length of meta data
                        
                        documents.append(doc)
                except Exception as e:
                    print(k, 'Error' , e)
    if(recursive): #if no splitter preference has been given during the function call, then use default(recursive = True)
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=[
            "\n\n",
            "\n"
        ])

    else: # if a preference has been given, i.e (recursive = False), then use semantic chunker 
        text_splitter = SemanticChunker(initialize_embeddings(), breakpoint_threshold_type='standard_deviation')
    

    chunks = text_splitter.split_documents(documents) # return the chunks, to vector store 
    return chunks
