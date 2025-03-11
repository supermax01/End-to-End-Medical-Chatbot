# Data Directory

This directory is where you should place any PDF files you want the chatbot to learn from. The system will automatically extract text from these files, process it, and use it as the knowledge base for answering questions.

## How PDFs Are Used

1. **Text Extraction**: When you run the system, it will extract all text from the PDFs in this directory.
2. **Chunking**: The text is split into smaller chunks (about 500 characters each).
3. **Embedding**: Each chunk is converted into a vector embedding.
4. **Storage**: These embeddings are stored in Pinecone for fast retrieval.
5. **Retrieval**: When you ask a question, the most relevant chunks are retrieved.
6. **Answer Generation**: The LLM uses these chunks to generate an answer to your question.

## What PDFs to Add

For best results, add PDFs containing factual medical information such as:

- Medical textbooks
- Medical encyclopedias
- Medical journals and research papers
- Drug reference guides
- Disease and condition guides

## Important Notes

1. **Content Determines Answers**: The chatbot can only answer questions based on information contained in the PDFs you add here. If information about a specific medical condition is not in your PDFs, the chatbot won't be able to provide accurate answers about it.

2. **Quality Matters**: The quality of the chatbot's responses depends directly on the quality and accuracy of the documents placed here.

3. **Copyright Considerations**: Ensure you have the appropriate rights to use any documents you place in this directory.

4. **Processing Time**: Large documents or a large number of documents will increase the initial processing time when you first run the system.

5. **Storage Requirements**: The system will generate embeddings for all text in these documents, which will be stored in Pinecone. Make sure your Pinecone plan has sufficient storage capacity.

## Examples

- A medical encyclopedia in PDF format
- A pharmacology textbook in PDF format
- A collection of medical journal articles in PDF format
- Medical guidelines or protocols in PDF format

## Customizing Processing

If you need to adjust how PDFs are processed (chunk size, overlap, etc.), you can modify the settings in `src/utils/document_processor.py`. 