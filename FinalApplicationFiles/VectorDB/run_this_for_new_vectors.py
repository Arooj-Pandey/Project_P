from Project_P.FinalApplicationFiles.chunker.chunker import chunker
from Project_P.FinalApplicationFiles.VectorDB.vector_addition import vector_addition





def main():
    path = r'D:\Projects\PIB Chatbot\Project_P\ScraperFiles\newFiles'
    try:
        vector_addition(chunker(path, recursive=False), 'semanticpib')
    except Exception as e:
        print("Not Added", e)    

if __name__ == "__main__":
    main()
