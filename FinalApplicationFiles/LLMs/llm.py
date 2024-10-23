from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough

def initialize_llm(api_key):
    """
    Initializes and returns the Google Generative AI model with the provided API key.
    """
    return GoogleGenerativeAI(
        model="gemini-1.5-flash",  # Model type
        google_api_key=api_key,
        temperature=0.01  # Controls creativity vs accuracy (low value -> more accuracy)
    )

def create_rag_chain(retriever, llm):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain using a custom prompt.
    Returns the RAG chain.
    """
    # Define the custom prompt for RAG
    template = """You are expert in coveying long stories into well detailed and in well understandable manner, you explain things in such a way that even a high school student is able to understand, so you have to people in understanding these press releases so that people can understand get better insight out of it. You will be provided with the pieces of the press releases you have to answere the question asked with the help of these pieces , if you are unable to answere just reply in a aplogizing tone, do not make up the answere. 
    {context}

    Question: {question}

    Helpful Answer:"""

    # Custom function to format retrieved documents into context string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create prompt template and RAG chain
    custom_rag_prompt = PromptTemplate.from_template(template)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
