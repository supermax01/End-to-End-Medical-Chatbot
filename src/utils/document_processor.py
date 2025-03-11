from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_documents(data_dir):
    """
    Load PDF documents from a directory
    
    Args:
        data_dir (str): Path to the directory containing PDF files
        
    Returns:
        list: List of loaded documents
    """
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        recursive=False,
        show_progress=True
    )
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=20):
    """
    Split documents into chunks
    
    Args:
        documents (list): List of documents
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks 