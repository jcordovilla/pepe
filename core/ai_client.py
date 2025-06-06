"""
Unified AI client for PEPE Discord Bot.
Handles all AI operations using local models (Ollama + Sentence Transformers).
"""
import requests
import json
import logging
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np

from .config import get_config

logger = logging.getLogger(__name__)

class AIClient:
    """Unified client for all AI operations using local models."""
    
    def __init__(self):
        self.config = get_config()
        self._embedding_model = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for AI operations."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.config.models.embedding_model}")
            self._embedding_model = SentenceTransformer(self.config.models.embedding_model)
        return self._embedding_model
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate chat completion using Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Convert messages to Ollama format
        prompt = self._format_messages_for_ollama(messages)
        
        payload = {
            "model": self.config.models.chat_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.models.chat_temperature,
                "num_predict": max_tokens or self.config.models.chat_max_tokens,
            }
        }
        
        try:
            response = requests.post(
                f"{self.config.models.ollama_base_url}/api/generate",
                json=payload,
                timeout=self.config.models.ollama_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Failed to get chat completion: {e}")
    
    def _format_messages_for_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Ollama prompt format."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final prompt for assistant response
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def create_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Create embeddings for text(s) using local sentence transformer.
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            logger.debug(f"Created embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise RuntimeError(f"Failed to create embeddings: {e}")
    
    def classify_text(
        self, 
        text: str, 
        categories: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Classify text into one of the provided categories.
        
        Args:
            text: Text to classify
            categories: List of possible categories
            system_prompt: Optional system prompt for classification
            
        Returns:
            The selected category
        """
        default_system = (
            "You are a text classifier. Classify the given text into one of the "
            "provided categories. Respond with only the category name, nothing else."
        )
        
        messages = [
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": f"Categories: {', '.join(categories)}\n\nText to classify: {text}"}
        ]
        
        response = self.chat_completion(messages, temperature=0.0)
        
        # Clean and validate response
        response = response.strip().lower()
        categories_lower = [cat.lower() for cat in categories]
        
        # Find best match
        for i, cat in enumerate(categories_lower):
            if cat in response or response in cat:
                return categories[i]
        
        # Default to first category if no match
        logger.warning(f"Classification failed to match category. Response: {response}")
        return categories[0]
    
    def summarize_text(
        self, 
        text: str, 
        max_length: int = 200,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Summarize the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            system_prompt: Optional system prompt for summarization
            
        Returns:
            Summary text
        """
        default_system = (
            f"You are a helpful assistant that creates concise summaries. "
            f"Summarize the following text in no more than {max_length} characters. "
            f"Focus on the key points and main ideas."
        )
        
        messages = [
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": f"Text to summarize:\n\n{text}"}
        ]
        
        return self.chat_completion(messages, temperature=0.1)
    
    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """
        Extract key terms/keywords from text.
        
        Args:
            text: Text to analyze
            num_keywords: Number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        messages = [
            {
                "role": "system", 
                "content": (
                    f"Extract {num_keywords} key terms or keywords from the given text. "
                    f"Return only the keywords separated by commas, nothing else."
                )
            },
            {"role": "user", "content": text}
        ]
        
        response = self.chat_completion(messages, temperature=0.0)
        
        # Parse keywords from response
        keywords = [kw.strip() for kw in response.split(',')]
        return keywords[:num_keywords]
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of all AI services.
        
        Returns:
            Health status dict
        """
        status = {
            "ollama": False,
            "embeddings": False,
            "overall": False
        }
        
        # Check Ollama
        try:
            response = requests.get(
                f"{self.config.models.ollama_base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                status["ollama"] = True
                logger.info("Ollama service is healthy")
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
        
        # Check embeddings
        try:
            test_embedding = self.create_embeddings("test")
            if test_embedding is not None and len(test_embedding) > 0:
                status["embeddings"] = True
                logger.info("Embedding model is healthy")
        except Exception as e:
            logger.error(f"Embedding health check failed: {e}")
        
        status["overall"] = status["ollama"] and status["embeddings"]
        
        return status


# Global AI client instance
_ai_client: Optional[AIClient] = None

def get_ai_client() -> AIClient:
    """Get the global AI client instance."""
    global _ai_client
    if _ai_client is None:
        _ai_client = AIClient()
    return _ai_client
