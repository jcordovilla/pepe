"""
Unified LLM Client

Provides a consistent interface for all LLM calls across the system.
Uses the same Llama model for all operations to ensure consistency.
"""

import asyncio
import json
import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..config.modernized_config import get_modernized_config

logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """
    Unified LLM client that ensures all modules use the same Llama model.
    
    Features:
    - Consistent model usage across all agents
    - Automatic fallback to backup model
    - Retry logic with exponential backoff
    - Caching support
    - Error handling and logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM client with configuration.
        
        Args:
            config: Optional config override, defaults to system config
        """
        self.config = config or get_modernized_config().get("llm", {})
        
        # LLM settings
        self.endpoint = self.config.get("endpoint", "http://localhost:11434/api/generate")
        self.model = self.config.get("model", "llama3.1:8b")
        self.fallback_model = self.config.get("fallback_model", "llama2:latest")
        self.max_tokens = self.config.get("max_tokens", 2048)
        self.temperature = self.config.get("temperature", 0.1)
        self.timeout = self.config.get("timeout", 30)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.timeout = self.timeout
        
        logger.info(f"UnifiedLLMClient initialized with model: {self.model}")
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_fallback: bool = True
    ) -> str:
        """
        Generate text using the configured Llama model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            use_fallback: Whether to use fallback model on failure
            
        Returns:
            Generated text response
        """
        try:
            # Try primary model first
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature
            )
            return response
            
        except Exception as e:
            logger.warning(f"Primary model {self.model} failed: {e}")
            
            if use_fallback and self.fallback_model != self.model:
                try:
                    logger.info(f"Trying fallback model: {self.fallback_model}")
                    response = await self._call_model(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=self.fallback_model,
                        max_tokens=max_tokens or self.max_tokens,
                        temperature=temperature or self.temperature
                    )
                    return response
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback model {self.fallback_model} also failed: {fallback_error}")
                    raise Exception(f"Both primary and fallback models failed. Last error: {fallback_error}")
            else:
                raise e
    
    async def _call_model(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Make the actual API call to the Llama model.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            max_tokens: Max tokens to generate
            temperature: Temperature setting
            
        Returns:
            Model response
        """
        # Build the full prompt
        full_prompt = ""
        if system_prompt:
            full_prompt += f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            full_prompt = prompt
        
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            }
        }
        
        # Make request with retries
        for attempt in range(self.retry_attempts):
            try:
                response = await asyncio.to_thread(
                    self.session.post,
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    raise Exception(f"API returned status {response.status_code}: {response.text}")
                    
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise e
                else:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
    
    async def generate_json(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON response from the model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            
        Returns:
            Parsed JSON response
        """
        # Add JSON formatting instruction to the prompt
        json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."
        
        response = await self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Try to extract JSON from the response
        try:
            # Look for JSON blocks in the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                return json.loads(response)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {response}")
            raise Exception(f"Invalid JSON response from model: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the LLM service is healthy.
        
        Returns:
            Health status information
        """
        try:
            # Test the primary model
            test_response = await self.generate(
                prompt="Hello, this is a health check. Please respond with 'OK'.",
                max_tokens=10,
                temperature=0.0
            )
            
            return {
                "status": "healthy",
                "primary_model": self.model,
                "fallback_model": self.fallback_model,
                "endpoint": self.endpoint,
                "test_response": test_response.strip(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "primary_model": self.model,
                "fallback_model": self.fallback_model,
                "endpoint": self.endpoint,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from the endpoint.
        
        Returns:
            List of available models
        """
        try:
            # Try to get models list from Ollama
            models_endpoint = self.endpoint.replace("/api/generate", "/api/tags")
            response = await asyncio.to_thread(
                self.session.get,
                models_endpoint,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            else:
                logger.warning(f"Could not fetch models list: {response.status_code}")
                return []
                
        except Exception as e:
            logger.warning(f"Error fetching available models: {e}")
            return []


# Global LLM client instance
_llm_client = None

def get_llm_client() -> UnifiedLLMClient:
    """
    Get the global LLM client instance.
    
    Returns:
        Unified LLM client
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = UnifiedLLMClient()
    return _llm_client 