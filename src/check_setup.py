#!/usr/bin/env python3
"""
Setup Check Script for Medical Chatbot

This script checks if all the required components for the Medical Chatbot
are properly set up and working.
"""

import os
import sys
import subprocess
import importlib
import platform
from pathlib import Path
from dotenv import load_dotenv
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored terminal output
colorama.init()

def print_success(message):
    """Print a success message in green"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print an error message in red"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_warning(message):
    """Print a warning message in yellow"""
    print(f"{Fore.YELLOW}! {message}{Style.RESET_ALL}")

def print_info(message):
    """Print an info message in blue"""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")

def print_header(message):
    """Print a header message"""
    print(f"\n{Fore.CYAN}=== {message} ==={Style.RESET_ALL}")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print_success(f"Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor}.{version.micro} is not compatible")
        print_info("This project requires Python 3.9 or higher")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print_header("Checking Dependencies")
    
    required_packages = [
        "langchain",
        "langchain_community",
        "langchain_core",
        "pinecone",
        "sentence_transformers",
        "streamlit",
        "ollama",
        "torch",
        "transformers",
        "pypdf",
        "dotenv",
        "tqdm",
        "pydantic"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print_success(f"{package} is installed")
        except ImportError:
            print_error(f"{package} is not installed")
            all_installed = False
    
    if not all_installed:
        print_info("Install missing dependencies with: pip install -r requirements.txt")
    
    return all_installed

def check_environment_variables():
    """Check if required environment variables are set"""
    print_header("Checking Environment Variables")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for Pinecone API key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key:
        print_success("PINECONE_API_KEY is set")
        return True
    else:
        print_error("PINECONE_API_KEY is not set")
        print_info("Create a .env file in the project root with your Pinecone API key")
        print_info("Example: PINECONE_API_KEY=your_pinecone_api_key")
        return False

def check_ollama():
    """Check if Ollama is installed and running"""
    print_header("Checking Ollama")
    
    # Check if Ollama is installed and running
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print_success("Ollama is installed and running")
            
            # Check available models
            models = result.stdout.strip().split('\n')[1:]  # Skip header
            if models and any(line.strip() for line in models):
                print_success("Ollama models are available:")
                for model in models:
                    if model.strip():
                        print_info(f"  - {model.split()[0]}")
            else:
                print_warning("No Ollama models found")
                print_info("Pull a model with: ollama pull llama3.2")
                return False
        else:
            print_error("Ollama is not running")
            print_info("Start Ollama and try again")
            return False
    except FileNotFoundError:
        print_error("Ollama is not installed or not in PATH")
        print_info("Install Ollama from: https://ollama.ai")
        return False
    except Exception as e:
        print_error(f"Error checking Ollama: {str(e)}")
        return False
    
    return True

def check_data_directory():
    """Check if data directory exists and contains PDF files"""
    print_header("Checking Data Directory")
    
    # Get the path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    # Check if data directory exists
    if os.path.exists(data_dir):
        print_success(f"Data directory exists: {data_dir}")
    else:
        print_warning(f"Data directory does not exist: {data_dir}")
        print_info("Creating data directory...")
        try:
            os.makedirs(data_dir, exist_ok=True)
            print_success("Data directory created successfully")
        except Exception as e:
            print_error(f"Error creating data directory: {str(e)}")
            return False
    
    # Check if data directory contains PDF files
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if pdf_files:
        print_success(f"Found {len(pdf_files)} PDF files in data directory")
        for pdf_file in pdf_files:
            print_info(f"  - {pdf_file}")
    else:
        print_warning("No PDF files found in data directory")
        print_info(f"Add PDF files to: {data_dir}")
        print_info("The application will not work without PDF files")
    
    return True

def main():
    """Main function to run all checks"""
    print_header("Medical Chatbot Setup Check")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Run all checks
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    env_ok = check_environment_variables()
    ollama_ok = check_ollama()
    data_ok = check_data_directory()
    
    # Print summary
    print_header("Summary")
    if python_ok and deps_ok and env_ok and ollama_ok and data_ok:
        print_success("All checks passed! You're ready to run the Medical Chatbot.")
        print_info("Run the application with: streamlit run src/app.py")
    else:
        print_error("Some checks failed. Please fix the issues above before running the application.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 