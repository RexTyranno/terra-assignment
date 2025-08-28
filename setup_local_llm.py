import logging
import requests
import subprocess
import sys
import time
import json
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("setup_local_llm")

def check_ollama_installation() -> bool:
    """Check if Ollama is installed and accessible."""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"Ollama is installed: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    logger.warning("Ollama is not installed or not in PATH")
    return False

def check_ollama_service() -> bool:
    """Check if Ollama service is running."""
    try:
        response = requests.get('http://localhost:11434/api/version', timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            logger.info(f"Ollama service is running (version: {version_info.get('version', 'unknown')})")
            return True
    except requests.exceptions.RequestException:
        pass
    
    logger.error("Ollama service is not running")
    return False

def list_available_models() -> List[str]:
    """List models available in Ollama."""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Failed to list Ollama models")
        pass
    
    return []

def download_model(model_name: str) -> bool:
    """Download a model using Ollama."""
    logger.info(f"Downloading {model_name}... This may take several minutes.")
    try:
        process = subprocess.Popen(['ollama', 'pull', model_name], 
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(f"   {output.strip()}")
        
        return process.poll() == 0
    except FileNotFoundError:
        logger.error("Ollama command not found")
        return False

def test_model_chat(model_name: str) -> bool:
    """Test if a model can generate responses."""
    try:
        payload = {
            "model": model_name,
            "prompt": "Hello! Please respond with just 'Hello traveler!' to test the connection.",
            "stream": False,
            "options": {"temperature": 0.1, "max_tokens": 50}
        }
        
        response = requests.post('http://localhost:11434/api/generate', 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            reply = result.get('response', '').strip()
            logger.info(f"Model {model_name} is working. Test response: {reply[:100]}...")
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error testing model: {e}")
    
    return False

def recommend_models() -> List[tuple]:
    """Recommend models based on use case."""
    return [
        ("llama2:7b", "Good balance of quality and speed", "~4GB RAM"),
        ("llama2:13b", "Higher quality responses", "~8GB RAM"),
        ("mistral:7b", "Fast and efficient", "~4GB RAM"),
        ("codellama:7b", "Good for code-related tasks", "~4GB RAM"),
        ("phi:2.7b", "Very fast, lower quality", "~2GB RAM"),
        ("tinyllama:1.1b", "Extremely fast, basic quality", "~1GB RAM"),
    ]

def main():
    """Main setup function."""
    print("NPC Chat System - Local LLM Setup Helper")
    print("=" * 50)
    
    # Check Ollama installation
    if not check_ollama_installation():
        print("\nTo install Ollama:")
        print("   Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh")
        print("   Windows: Download from https://ollama.com/download")
        print("   Then run: ollama serve")
        return False
    
    # Check if service is running
    if not check_ollama_service():
        print("\nTo start Ollama service:")
        print("   Run: ollama serve")
        print("   Then run this script again")
        return False
    
    # List existing models
    models = list_available_models()
    if models:
        print(f"\Installed models ({len(models)}):")
        for model in models:
            print(f"   • {model}")
    else:
        print("\nNo models installed")
    
    # Show recommendations
    print("\nRecommended models:")
    recommendations = recommend_models()
    for model, description, ram in recommendations:
        status = "Installed" if model in models else "Available"
        print(f"   {status} {model} - {description} ({ram})")
    
    # Interactive model download
    if not models:
        print("\nYou need at least one model to use the NPC Chat System.")
        choice = input("Would you like to download a recommended model? (y/n): ").lower()
        
        if choice in ['y', 'yes']:
            logger.info("\nChoose a model to download:")
            for i, (model, desc, ram) in enumerate(recommendations[:4], 1):
                logger.info(f"   {i}. {model} - {desc} ({ram})")
            
            try:
                print("Enter choice (1-4): ", end="")
                selection = int(input())
                if 1 <= selection <= 4:
                    model_name = recommendations[selection-1][0]
                    if download_model(model_name):
                        logger.info(f"Successfully downloaded {model_name}")
                        
                        # Test the model
                        logger.info(f"\n Testing {model_name}...")
                        if test_model_chat(model_name):
                            logger.info(f"{model_name} is ready to use!")
                        else:
                            logger.warning(f" {model_name} downloaded but test failed")
                    else:
                        logger.error(f"Failed to download {model_name}")
            except (ValueError, KeyboardInterrupt):
                logger.warning("Setup cancelled.")
    
    # Final status
    logger.info("\nSetup Complete!")
    logger.info("\nTo use the NPC Chat System:")
    logger.info("   python npc_chat.py --web")
    logger.info("   or")
    logger.info("   python npc_chat.py --provider local --local-model llama2:7b")
    
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n Unexpected error: {e}", exc_info=True)
        sys.exit(1)