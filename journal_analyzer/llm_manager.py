import os
import logging
import requests
from typing import Optional, Dict, Literal
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from enum import Enum

from .config import (
    DEFAULT_LLM_MODEL,
    OLLAMA_BASE_URL,
    CURRENT_LLM_BACKEND
)

logging.basicConfig(level=logging.INFO)

class LLMBackend(Enum):
    """Enum for supported LLM backends."""
    LAMBDA = "lambda"
    OLLAMA = "ollama"
    
    @classmethod
    def from_string(cls, backend_str: str) -> 'LLMBackend':
        """Convert string to LLMBackend enum, with error handling."""
        try:
            return cls(backend_str.lower())
        except ValueError:
            logging.warning(f"Invalid backend '{backend_str}'. Using Lambda API.")
            return cls.LAMBDA

class LLMManager:
    _instance = None
    
    def __new__(cls, model: str = DEFAULT_LLM_MODEL):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance.model = model
            cls._instance.backend = LLMBackend.from_string(CURRENT_LLM_BACKEND)
            cls._instance.client = cls._instance._create_client() if cls._instance.backend == LLMBackend.LAMBDA else None
        return cls._instance

    def __init__(self, model: str = DEFAULT_LLM_MODEL):
        # __new__ handles initialization
        pass

    def _create_client(self) -> OpenAI:
        """Create and return a Lambda API client."""
        api_key = os.getenv("LAMBDA_API_KEY")
        if not api_key:
            raise ValueError("LAMBDA_API_KEY not found in environment")
            
        return OpenAI(
            api_key=api_key,
            base_url="https://api.lambda.ai/v1"
        )

    def query_llm(self, prompt: str, temperature: float = 0.0) -> Optional[str]:
        """
        Send a query to Lambda's LLM API and return the response.
        
        Args:
            prompt: The prompt to send
            temperature: Controls randomness (0.0 for consistent responses)
            
        Returns:
            The LLM's response as a string, or None if error
        """
        try:

            # Send request to Lambda API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing journal entries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            
            # Log token usage
            if hasattr(response, 'usage'):
                logging.info(f"Token usage: {response.usage.prompt_tokens} prompt + "
                           f"{response.usage.completion_tokens} completion = "
                           f"{response.usage.total_tokens} total")

            return response.choices[0].message.content.strip()
                
        except Exception as e:
            logging.error(f"Error querying Lambda API: {e}")
            return None

# Simplified query function
def query_llm(prompt: str, model: str = None, temperature: float = 0.0) -> Optional[str]:
    """Query LLM using configured backend."""
    
    if CURRENT_LLM_BACKEND == 'lambda':
        # Use Lambda API
        api_key = os.getenv('LAMBDA_API_KEY')
        if not api_key:
            logging.error("LAMBDA_API_KEY not found in environment")
            return None
            
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.lambda.ai/v1"
        )
        
        try:
            response = client.chat.completions.create(
                model=model or os.getenv('LAMBDA_MODEL', 'llama3.3-70b-instruct-fp8'),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing journal entries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Lambda API error: {e}")
            return None
    else:
        # Ollama API call
        try:
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    "model": model or "cogito:70b",
                    "prompt": prompt
                }
            )
            return response.json().get('response')
        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            return None

# Example usage
if __name__ == "__main__":
    test_prompt = "Why is the sky blue?"
    response = query_llm(test_prompt)
    if response:
        print(f"Response: {response}")
    else:
        print("Failed to get response") 