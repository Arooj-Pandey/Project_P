import os
from langchain.document_loaders import TextLoader  # Assuming you are using langchain's TextLoader

class Chunker:
    def __init__(self, main_file_path, chunk_size=1000, chunk_overlap=200, recursive=True):
        self.main_file_path = main_file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive = recursive
        self.documents = []

    def load_documents(self):
        """
        This method iterates through the directory structure and loads .txt files
        into a list of documents with metadata.
        """
        for i in os.listdir(self.main_file_path):
            for j in os.listdir(os.path.join(self.main_file_path, i)):
                for k in os.listdir(os.path.join(self.main_file_path, i, j)):
                    try:
                        if k.endswith('.txt'):
                            file_path = os.path.join(self.main_file_path, i, j, k)

                            # Create loader and load the document
                            loader = TextLoader(file_path, encoding='utf-8')
                            doc = loader.load()[0]

                            # Add metadata (e.g., filename or custom data)
                            doc.metadata = {"source": k, "Ministry": j}  # Metadata with 'source' and 'Ministry'

                            self.documents.append(doc)
                    except Exception as e:
                        print(f"Error loading {k}: {e}")

    def get_documents(self):
        """
        Returns the list of loaded documents.
        """
        return self.documents
