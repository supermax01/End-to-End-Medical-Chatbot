import os
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()

def init_pinecone():
    """
    Initialize Pinecone client
    
    Returns:
        Pinecone: Initialized Pinecone client
    """
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Check if API key is loaded
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing! Set it in the .env file.")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc

def get_or_create_index(pc, index_name="medical-chatbot", dimension=384):
    """
    Get or create a Pinecone index
    
    Args:
        pc (Pinecone): Pinecone client
        index_name (str): Name of the index
        dimension (int): Dimension of the embeddings
        
    Returns:
        Index: Pinecone index
    """
    # Ensure the index exists before using it
    if index_name not in pc.list_indexes().names():
        from pinecone import ServerlessSpec

        print(f"Index '{index_name}' not found. Creating it now...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # Connect to the index
    index = pc.Index(index_name)
    return index

def store_embeddings(index, text_chunks, embeddings, batch_size=100):
    """
    Store embeddings in Pinecone
    
    Args:
        index: Pinecone index
        text_chunks: List of text chunks
        embeddings: Embeddings model
        batch_size (int): Batch size for processing
        
    Returns:
        None
    """
    # Process in batches
    total_batches = len(text_chunks) // batch_size + (1 if len(text_chunks) % batch_size != 0 else 0)
    print(f"Processing {len(text_chunks)} chunks in {total_batches} batches of size {batch_size}")

    for i in tqdm(range(0, len(text_chunks), batch_size)):
        # Get the current batch
        batch = text_chunks[i:i+batch_size]
        
        # Create vectors for the current batch
        vectors = [
            (str(i+j), embeddings.embed_query(t.page_content)) 
            for j, t in enumerate(batch)
        ]
        
        # Upsert the current batch
        index.upsert(vectors=vectors)

    print(f"Successfully stored all embeddings in Pinecone index: {index.name}")

def query_pinecone(index, query_embedding, top_k=2, include_metadata=True):
    """
    Query Pinecone index
    
    Args:
        index: Pinecone index
        query_embedding: Embedding of the query
        top_k (int): Number of results to return
        include_metadata (bool): Whether to include metadata
        
    Returns:
        dict: Query results
    """
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=include_metadata
    )
    return results 