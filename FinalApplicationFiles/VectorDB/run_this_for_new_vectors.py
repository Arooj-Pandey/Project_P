from FinalApplicationFiles.chunker.Chunker_setup_class import Chunker
from FinalApplicationFiles.VectorDB.vector_addition import vector_addition


path = r'D:\Projects\PIB Chatbot\Project_P\ScraperFiles\newFiles'
chunker  =  Chunker(path)


def main():
    path = r'D:\Projects\PIB Chatbot\Project_P\ScraperFiles\newFiles'
    try:
        vector_addition(chunker(path, recursive=False), 'semanticpib')
    except Exception as e:
        print("Not Added", e)    

if __name__ == "__main__":
    main()
