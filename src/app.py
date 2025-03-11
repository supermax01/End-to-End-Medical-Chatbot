import streamlit as st
import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv
import html

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("PINECONE_API_KEY"):
    st.error("⚠️ PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# Import our modules
try:
    from embeddings.embeddings import get_embeddings
    from retrieval.pinecone_retriever import init_pinecone, get_or_create_index
    from llm.ollama_llm import get_ollama_llm
    from utils.document_processor import load_pdf_documents, split_documents
    from utils.qa_chain import answer_question
except ImportError as e:
    st.error(f"⚠️ Error importing modules: {str(e)}")
    st.error("Please make sure you've installed all required dependencies with 'pip install -r requirements.txt'")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="wide",  # Using wide layout but with custom container width
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f3ff;
        border-left: 5px solid #2196F3;
    }
    .chat-message.bot {
        background-color: #f0f0f0;
        border-left: 5px solid #4CAF50;
    }
    .chat-message .message-content {
        display: flex;
        flex-direction: column;
        margin-top: 0.5rem;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .source-box {
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 0.3rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .source-title {
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    /* Improve button layout */
    .stButton>button {
        width: 100%;
        text-align: center;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    /* Improve spinner visibility */
    .stSpinner {
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    /* Fix header spacing */
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    /* Improve chat input */
    .stChatInput {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none;
    }
    /* Make header more prominent */
    .header-container {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #4CAF50;
    }
    /* Fix text formatting */
    p {
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your medical assistant. I can answer questions based only on the uploaded medical PDF documents. My knowledge is limited to the information contained in these documents. How can I help you today?"}
    ]

# Initialize session state for components
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.embeddings = None
    st.session_state.index = None
    st.session_state.text_chunks = None
    st.session_state.llm = None
    st.session_state.initialization_error = None

# Header section
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.title("MediRAG")
st.markdown("**Your trusted medical knowledge assistant powered by local documents and AI**")
st.markdown("This chatbot answers medical questions based **only** on the uploaded medical PDF documents. The answers are limited to the information contained in these documents. Ask any medical question to get started.")
st.markdown("<div style='font-size: 0.8rem; margin-top: 0.5rem;'>Created by <a href='https://github.com/supermax01' target='_blank'>supermax01</a> | <a href='https://github.com/supermax01/End-to-End-Medical-Chatbot' target='_blank'>GitHub Repository</a></div>", unsafe_allow_html=True)

# Check if Ollama is running
try:
    import subprocess
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode == 0:
        models = result.stdout.strip().split('\n')[1:]  # Skip header
        if models and any(line.strip() for line in models):
            model_names = [model.split()[0] for model in models if model.strip()]
            st.markdown(f"✅ **Using Ollama with models:** {', '.join(model_names)}")
        else:
            st.warning("⚠️ No Ollama models found. Please pull a model with 'ollama pull llama3.2'")
    else:
        st.error("❌ Ollama is not running. Please start Ollama.")
except Exception as e:
    st.error(f"❌ Error checking Ollama: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Status indicator
if not st.session_state.initialized:
    st.warning("⏳ Please wait for initialization to complete before asking questions.")
else:
    st.success("✅ System is ready! You can ask questions now.")

# Check for data directory and PDF files
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
if not os.path.exists(data_dir):
    st.warning(f"⚠️ Data directory not found at {data_dir}. Creating it now.")
    try:
        os.makedirs(data_dir, exist_ok=True)
        st.success("✅ Data directory created successfully.")
    except Exception as e:
        st.error(f"❌ Error creating data directory: {str(e)}")

pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')] if os.path.exists(data_dir) else []
if not pdf_files:
    st.warning("⚠️ No PDF files found in the data directory. Please add some PDF files to get started.")
    st.markdown(f"""
    ### How to add PDF files:
    1. Navigate to the data directory at: `{data_dir}`
    2. Add your medical PDF files there
    3. Restart the application
    """)
    st.stop()

# Initialize components if not already initialized
if not st.session_state.initialized and pdf_files:
    # Use a more compact spinner with a progress bar
    progress_placeholder = st.empty()
    progress_placeholder.info("Initializing components... This may take a few minutes.")
    
    try:
        # Initialize embeddings
        logger.info("Initializing embeddings...")
        progress_placeholder.info("Initializing embeddings... (1/4)")
        st.session_state.embeddings = get_embeddings()
        
        # Initialize Pinecone
        logger.info("Initializing Pinecone...")
        progress_placeholder.info("Connecting to Pinecone... (2/4)")
        pc = init_pinecone()
        st.session_state.index = get_or_create_index(pc)
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        progress_placeholder.info("Loading language model... (3/4)")
        st.session_state.llm = get_ollama_llm()
        
        # Load and process documents
        logger.info(f"Loading documents from {data_dir}...")
        progress_placeholder.info(f"Processing documents... (4/4)")
        documents = load_pdf_documents(data_dir)
        st.session_state.text_chunks = split_documents(documents)
        
        st.session_state.initialized = True
        logger.info(f"Initialization complete. Loaded {len(st.session_state.text_chunks)} text chunks.")
        progress_placeholder.success(f"✅ Ready! Loaded {len(st.session_state.text_chunks)} text chunks from {len(pdf_files)} PDF files.")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error initializing components: {error_msg}")
        st.session_state.initialization_error = error_msg
        progress_placeholder.error(f"❌ Error initializing components: {error_msg}")
        
        if "Connection refused" in error_msg and "ollama" in error_msg.lower():
            st.error("It looks like Ollama is not running. Please start Ollama and refresh this page.")
        elif "PINECONE_API_KEY" in error_msg:
            st.error("Please make sure your Pinecone API key is set in the .env file.")
        
        st.stop()

# Display initialization error if there is one
if st.session_state.initialization_error:
    st.error(f"❌ Error initializing components: {st.session_state.initialization_error}")
    if "Connection refused" in st.session_state.initialization_error and "ollama" in st.session_state.initialization_error.lower():
        st.error("It looks like Ollama is not running. Please start Ollama and refresh this page.")
    elif "PINECONE_API_KEY" in st.session_state.initialization_error:
        st.error("Please make sure your Pinecone API key is set in the .env file.")
    st.stop()

# Chat interface
st.subheader("Chat")

# Create a container for chat messages with fixed height for scrolling
chat_container = st.container()
with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            # Escape HTML to prevent formatting issues
            content = html.escape(message["content"])
            st.markdown(f'<div class="chat-message user"><div><strong>You</strong></div><div class="message-content">{content}</div></div>', unsafe_allow_html=True)
        else:
            # Escape HTML to prevent formatting issues
            content = html.escape(message["content"])
            sources = message.get("sources", [])
            
            source_html = ""
            if sources:
                source_html = '<div class="source-box"><div class="source-title">Sources:</div><ul>'
                for source in sources:
                    # Escape HTML in sources
                    safe_source = html.escape(source)
                    source_html += f'<li>{safe_source}</li>'
                source_html += '</ul></div>'
            
            st.markdown(f'<div class="chat-message bot"><div><strong>Medical Assistant</strong></div><div class="message-content">{content}</div>{source_html}</div>', unsafe_allow_html=True)

# Chat input
if st.session_state.initialized:
    user_input = st.chat_input("Ask a medical question...")

    # Process user input
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message immediately
        with chat_container:
            # Escape HTML to prevent formatting issues
            content = html.escape(user_input)
            st.markdown(f'<div class="chat-message user"><div><strong>You</strong></div><div class="message-content">{content}</div></div>', unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("Searching medical literature..."):
            try:
                result = answer_question(
                    user_input,
                    st.session_state.index,
                    st.session_state.embeddings,
                    st.session_state.text_chunks,
                    st.session_state.llm
                )
                
                # Format sources
                sources = [doc.page_content[:100] + "..." for doc in result["source_documents"]]
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["result"],
                    "sources": sources
                })
                
                # Display assistant message
                with chat_container:
                    # Escape HTML to prevent formatting issues
                    content = html.escape(result["result"])
                    
                    source_html = ""
                    if sources:
                        source_html = '<div class="source-box"><div class="source-title">Sources:</div><ul>'
                        for source in sources:
                            # Escape HTML in sources
                            safe_source = html.escape(source)
                            source_html += f'<li>{safe_source}</li>'
                        source_html += '</ul></div>'
                    
                    st.markdown(f'<div class="chat-message bot"><div><strong>Medical Assistant</strong></div><div class="message-content">{content}</div>{source_html}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error generating response: {error_msg}")
                st.error(f"❌ Error generating response: {error_msg}")
                
                if "Connection refused" in error_msg and "ollama" in error_msg.lower():
                    st.error("It looks like Ollama is not running. Please start Ollama and refresh this page.")
                
                # Add error message to chat
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"I'm sorry, I encountered an error: {error_msg}"
                })
        
        # Rerun to update the UI
        st.rerun()
else:
    st.info("Initializing components... Please wait.")

# Footer
st.markdown("---")
st.markdown("Powered by Ollama and Pinecone | [GitHub Repository](https://github.com/supermax01/End-to-End-Medical-Chatbot) | Created by [supermax01](https://github.com/supermax01)")

# Run the app with: streamlit run src/app.py 