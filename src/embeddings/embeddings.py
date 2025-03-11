from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize and return a HuggingFaceEmbeddings model
    
    Args:
        model_name (str): Name of the model to use for embeddings
        
    Returns:
        HuggingFaceEmbeddings: The initialized embeddings model
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings 