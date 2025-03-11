from langchain_core.prompts import PromptTemplate

def get_qa_prompt():
    """
    Get the prompt template for QA
    
    Returns:
        PromptTemplate: The prompt template
    """
    template = """
    You are a knowledgeable medical assistant providing accurate information based on medical literature. 
    Your task is to answer the user's question using ONLY the provided context information.

    Guidelines:
    - Base your answer EXCLUSIVELY on the information in the context provided below
    - If the context doesn't contain enough information to answer the question fully, acknowledge the limitations
    - If you don't know the answer based on the context, simply say "I don't have enough information to answer this question"
    - Be concise but thorough in your explanations
    - Use medical terminology appropriately but explain complex terms
    - Format your answer in clear, complete sentences with proper paragraphs
    - Use bullet points only when listing multiple items
    - Ensure your response has a logical flow and is easy to read
    - Do NOT include any information that is not supported by the context
    - Do NOT make up or infer information beyond what is explicitly stated in the context
    - Do NOT reference the context directly in your answer (e.g., don't say "According to the context...")

    Question: {question}

    Context:
    {context}

    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "context"]
    )
    
    return prompt

def answer_question(question, index, embeddings, text_chunks, llm, k=3):
    """
    Answer a question using the retrieval-based QA system
    
    Args:
        question (str): The question to answer
        index: Pinecone index
        embeddings: Embeddings model
        text_chunks: List of text chunks
        llm: Language model
        k (int): Number of documents to retrieve (increased from 2 to 3 for better context)
        
    Returns:
        dict: Result containing the answer and source documents
    """
    # Convert query to embedding
    query_embedding = embeddings.embed_query(question)
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    
    # Get the documents
    documents = []
    for match in results['matches']:
        doc_id = int(match['id'])
        if 0 <= doc_id < len(text_chunks):
            documents.append(text_chunks[doc_id])
    
    # Create context from documents
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Create prompt
    prompt = get_qa_prompt().format(
        question=question,
        context=context
    )
    
    # Get answer from LLM
    answer = llm.invoke(prompt)
    
    return {
        "result": answer,
        "source_documents": documents
    } 