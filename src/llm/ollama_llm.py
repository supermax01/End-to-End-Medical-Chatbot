from langchain_community.llms import Ollama
import logging

logger = logging.getLogger(__name__)

def get_ollama_llm(
    model_name="llama3.2", 
    temperature=0.3,  # Lower temperature for more factual responses
    num_predict=512,  # Increased token limit for more detailed answers
    top_k=30,         # Reduced for more focused sampling
    top_p=0.85,       # Slightly reduced for more deterministic outputs
    repeat_penalty=1.2, # Slightly increased to reduce repetition
    system_prompt=None # Optional system prompt
):
    """
    Initialize and return an Ollama LLM optimized for medical question answering
    
    Args:
        model_name (str): Name of the Ollama model to use
        temperature (float): Temperature for generation (lower for more factual responses)
        num_predict (int): Maximum number of tokens to generate
        top_k (int): Top-k sampling parameter
        top_p (float): Top-p sampling parameter
        repeat_penalty (float): Penalty for repeating tokens
        system_prompt (str): Optional system prompt to guide the model's behavior
        
    Returns:
        Ollama: The initialized Ollama LLM
    """
    # Default system prompt for medical question answering if none provided
    if system_prompt is None:
        system_prompt = """You are a medical assistant providing accurate information based on medical literature. Your answers should be factual, precise, and based only on verified medical information. Avoid speculation and clearly indicate when information might be incomplete."""
    
    logger.info(f"Initializing Ollama with model: {model_name}")
    
    # Create the Ollama LLM instance
    llm = Ollama(
        model=model_name,
        temperature=temperature,
        num_predict=num_predict,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        system=system_prompt
    )
    
    return llm 