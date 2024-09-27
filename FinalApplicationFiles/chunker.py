import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader



def chunker(Main_file_path, chunk_size = 1000 , chunk_overlap = 200):
    documents = []


    for i in os.listdir(Main_file_path):
        for j in os.listdir(os.path.join(Main_file_path, i)):
            for k in os.listdir(os.path.join(Main_file_path, i , j)):
                try:
                    if k.endswith(".txt"):
                        file_path = os.path.join(Main_file_path, i , j, k)
                        
                        # Create loader and load the document
                        loader = TextLoader(file_path)
                        doc = loader.load()[0]
                        
                        # Add metadata (e.g., filename or custom data)
                        doc.metadata = {"source": k, "Ministry": j, "Month":i}
                        
                        documents.append(doc)
                except Exception as e:
                    print(k, 'Error' , e)
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
    separators=[
        "\n\n",
        "\n"
    ])
    chunks = text_splitter.split_documents(documents)
    return chunks
