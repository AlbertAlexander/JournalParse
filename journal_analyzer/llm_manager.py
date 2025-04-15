import os
import logging
import requests
import json
from typing import Optional, Dict, Literal
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from enum import Enum

from .config import (
    DEFAULT_LLM_MODEL,
    OLLAMA_BASE_URL,
    CURRENT_LLM_BACKEND,
    CLI_SELECTED_MODEL
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

def get_active_model() -> str:
    """Get the currently active model, preferring CLI selection over backend default."""
    if CLI_SELECTED_MODEL:
        return CLI_SELECTED_MODEL
    return DEFAULT_LLM_MODEL[CURRENT_LLM_BACKEND]

def query_llm(prompt: str) -> str:
    """Query LLM using active model configuration."""
    active_model = get_active_model()
    logging.debug(f"Using {CURRENT_LLM_BACKEND} backend with model: {active_model}")
    
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
                model=active_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing journal entries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
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
                    "model": active_model,
                    "prompt": prompt
                }
            )
            return response.json().get('response')
        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            return None

def parse_llm_json_response(response: str, required_fields: list = None, defaults: dict = None) -> Dict:
    """Parse JSON from LLM response with validation and defaults.
    
    Args:
        response: Raw response string from LLM
        required_fields: List of fields that should be present in response
        defaults: Dictionary of default values for missing fields
        
    Returns:
        Parsed and validated JSON dictionary
    """
    try:
        if not response:
            raise ValueError("Empty response from LLM")
            
        # Extract JSON if it's embedded in other text
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON object found in response")
            
        # Validate and set defaults if specified
        if required_fields and defaults:
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                logging.warning(f"Missing fields in LLM response: {missing_fields}")
                for field in missing_fields:
                    result[field] = defaults.get(field)
                    
        return result
        
    except Exception as e:
        logging.error(f"Error parsing LLM response: {e}")
        raise

# Example usage
if __name__ == "__main__":
    test_prompt = "Why is the sky blue?"
    response = query_llm(test_prompt)
    if response:
        print(f"Response: {response}")
    else:
        print("Failed to get response") 