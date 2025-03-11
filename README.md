# MediRAG: Local RAG System : Get Accurate medical answers from your trusted documents locally

Created by [supermax01](https://github.com/supermax01)

A specialized medical question-answering system that leverages large language models (LLMs) and retrieval-augmented generation (RAG) to provide accurate medical information based on trusted medical literature. This system runs entirely on your local machine, giving you full control over your data and privacy.

![Screenshot 1](Screenshot 2025-03-11 at 12.21.22 AM.png)
![Screenshot 2](Screenshot 2025-03-11 at 12.22.01 AM.png)
![Screenshot 3](Screenshot 2025-03-11 at 12.22.18 AM.png)

## Project Overview

This project creates an end-to-end local RAG (Retrieval-Augmented Generation) medical chatbot that:

1. **Extracts knowledge** from medical PDFs and documents
2. **Processes and chunks** the text into manageable segments
3. **Generates embeddings** using sentence transformers
4. **Stores vectors** in Pinecone for efficient semantic search
5. **Retrieves relevant context** when a medical question is asked
6. **Generates accurate answers** using Ollama's LLMs with the retrieved context

The system is designed to provide factual, context-based responses to medical queries while citing the sources of information, making it suitable for educational purposes and preliminary medical information lookup.

## What is RAG (Retrieval-Augmented Generation)?

RAG combines the power of large language models with information retrieval systems to generate more accurate, factual, and contextually relevant responses:

1. **Retrieval**: When a question is asked, the system searches through a knowledge base (in this case, your medical PDFs) to find the most relevant information.

2. **Augmentation**: The retrieved information is added to the prompt sent to the language model.

3. **Generation**: The language model generates a response based on both its pre-trained knowledge and the specific information retrieved from your documents.

### Benefits of Local RAG

- **Accuracy**: Responses are grounded in specific documents you provide, reducing hallucinations
- **Privacy**: Your medical documents and queries never leave your computer
- **Customization**: You control exactly what knowledge the system has access to
- **Transparency**: The system shows you the sources it used to generate each answer
- **Cost-effective**: No need for expensive API calls to cloud-based LLMs

## How It Works

1. **Document Ingestion**: The system reads and processes any PDF files placed in the `data/` directory. These can be medical textbooks, research papers, or any text-based medical information.

2. **Knowledge Base Creation**: The content is split into chunks and converted into vector embeddings, which are stored in Pinecone.

3. **Question Processing**: When you ask a question, the system:
   - Converts your question into an embedding
   - Searches Pinecone for the most relevant text chunks
   - Retrieves these chunks to use as context

4. **Answer Generation**: The system uses Ollama to run a local LLM (like llama3.2 or phi4-mini) that:
   - Receives your question and the retrieved context
   - Generates an answer based only on the provided context
   - Cites the sources used to create the response

## Key Features

- **Fully Local Processing**: All components run on your machine, with no data sent to external APIs
- **Document Processing**: Automatically extracts and processes text from medical PDFs
- **Vector Search**: Uses semantic search to find the most relevant information for each query
- **Context-Aware Responses**: Generates answers based only on the retrieved medical literature
- **Source Attribution**: Provides the sources of information used to generate each answer
- **Modular Architecture**: Easily extensible with new data sources or models
- **Interactive Web Interface**: User-friendly Streamlit interface for asking questions

## Technical Stack

- **LLM**: Ollama (with models like llama3.2, phi4-mini) - runs locally on your machine
- **Embeddings**: HuggingFace Sentence Transformers - processed locally
- **Vector Database**: Pinecone - for efficient similarity search
- **Document Processing**: LangChain document loaders and text splitters
- **Web Interface**: Streamlit - runs locally in your browser
- **Language**: Python 3.9+

## Project Structure

```
End-to-End-Medical-Chatbot/
├── data/                  # Directory for medical PDF files
│   └── README.md          # Instructions for adding medical PDFs
├── src/                   # Source code
│   ├── embeddings/        # Embeddings generation module
│   ├── llm/               # LLM integration with Ollama
│   ├── retrieval/         # Pinecone vector search module
│   ├── utils/             # Document processing and QA utilities
│   ├── app.py             # Streamlit web application
│   └── check_setup.py     # Setup verification script
├── .env                   # Environment variables (API keys)
├── .gitignore             # Git ignore file
├── requirements.txt       # Dependencies
├── LICENSE                # License file
└── README.md              # This file
```

## Complete Setup Guide

Follow these steps to set up and run the medical chatbot on your device:

### 1. Clone the Repository

```bash
git clone https://github.com/supermax01/End-to-End-Medical-Chatbot.git
cd End-to-End-Medical-Chatbot
```

### 2. Set Up Python Environment

It's recommended to use a virtual environment:

```bash
# For conda (recommended)
conda create -n mchatbot python=3.9
conda activate mchatbot

# OR for venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Pinecone

1. Create a free account at [Pinecone](https://www.pinecone.io/)
2. Create a new project and get your API key
3. Create a `.env` file in the project root with your API key:

```
PINECONE_API_KEY=your_pinecone_api_key
```

### 5. Install and Set Up Ollama

1. Download and install Ollama from [ollama.ai](https://ollama.ai) for your operating system
2. Start the Ollama service:
   - On macOS: Ollama will start automatically after installation
   - On Linux: Run `ollama serve` in a terminal
   - On Windows: Ollama will start automatically after installation
3. Pull the models you want to use:

```bash
ollama pull llama3.2  # Recommended model
# OR
ollama pull phi4-mini  # Alternative model
```

4. Verify Ollama is running and models are available:

```bash
ollama list
```

### 6. Add Medical PDF Files

1. Place your medical PDF files in the `data/` directory
   - The app will create this directory automatically if it doesn't exist
   - You can use medical textbooks, research papers, or any PDF with medical information
2. Make sure you have the appropriate rights to use these documents

### 7. Verify Your Setup

Run the setup check script to verify that everything is properly configured:

```bash
python src/check_setup.py
```

This script will:
- Check if Python version is compatible
- Verify all required dependencies are installed
- Check if environment variables are set correctly
- Verify Ollama is installed and running
- Check if the data directory exists and contains PDF files

If any issues are found, the script will provide guidance on how to fix them.

### 8. Run the Application

```bash
streamlit run src/app.py
```

The application will:
1. Automatically open in your default web browser at http://localhost:8501
2. Process all PDF files in your `data/` directory
3. Generate and store embeddings in Pinecone
4. Start the chat interface where you can ask medical questions

If you want to access the app from other devices on your network:

```bash
streamlit run src/app.py --server.address 0.0.0.0
```

Then access it from other devices using your computer's IP address: `http://YOUR_IP_ADDRESS:8501`

### 9. Troubleshooting

If you encounter any issues:

- **Ollama not running**: Make sure Ollama is installed and running
- **Pinecone API key error**: Check your `.env` file has the correct API key
- **No PDF files found**: Add PDF files to the `data/` directory
- **Import errors**: Make sure all dependencies are installed correctly
- **Memory issues**: Try using smaller PDF files or fewer files

## Customizing the System

### Changing the LLM Model

You can modify which Ollama model is used by editing `src/llm/ollama_llm.py`:

```python
# Change the default model and parameters
def get_ollama_llm(
    model_name="llama3.2",  # Change to any model you've pulled in Ollama
    temperature=0.5,        # Adjust for more/less creative responses
    num_predict=256,        # Maximum tokens to generate
    top_k=40,               # Sampling parameter
    top_p=0.9,              # Sampling parameter
    repeat_penalty=1.18     # Penalty for repetition
):
```

### Customizing the Prompt Template

You can modify how questions are formatted for the LLM by editing `src/utils/qa_chain.py`:

```python
def get_qa_prompt():
    template = """
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.

    Question: {question}

    Context: {context}

    Only answer in the context of the provided information. Be concise and accurate.
    """
    # You can customize this prompt template to change how the model responds
```

## Example Questions

- "What are the symptoms of diabetes?"
- "How is pneumonia diagnosed?"
- "What treatments are available for migraines?"
- "What are the side effects of ibuprofen?"

## Limitations

- The chatbot can only answer based on the information in the PDF files you provide
- It is designed for informational purposes only and should not replace professional medical advice
- Response quality depends on the quality and coverage of the source documents
- The system requires Ollama to be installed and running on your machine
- While the LLM and processing run locally, Pinecone is a cloud service that stores your vector embeddings

## Future Improvements

- Integration with medical knowledge graphs
- Support for multi-modal inputs (images, lab results)
- User feedback loop for answer quality improvement
- Expanded medical document corpus
- Option for fully local vector database (like Chroma or FAISS) instead of Pinecone
