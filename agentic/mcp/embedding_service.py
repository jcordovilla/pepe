"""
Embedding Service for MCP Server

Handles embedding generation using OpenAI and sentence-transformers models.
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

import openai
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using various models.
    
    Supports:
    - OpenAI embeddings (text-embedding-3-small)
    - Sentence transformers (msmarco-distilbert-base-v4)
    - Local models via Ollama (future)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Model configuration
        self.default_model = self.config.get("default_model", "text-embedding-3-small")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.sentence_transformer_model = self.config.get("sentence_transformer_model", "msmarco-distilbert-base-v4")
        
        # Initialize models
        self.openai_client = None
        self.sentence_transformer = None
        self._initialize_models()
        
        # Performance tracking
        self.stats = {
            "total_embeddings": 0,
            "openai_embeddings": 0,
            "sentence_transformer_embeddings": 0,
            "errors": 0,
            "cache_hits": 0
        }
        
        logger.info(f"EmbeddingService initialized with default model: {self.default_model}")
    
    def _initialize_models(self):
        """Initialize embedding models based on configuration."""
        try:
            # Initialize OpenAI client if API key is available
            if self.openai_api_key and self.openai_api_key != "test-key-for-testing":
                self.openai_client = openai.AsyncOpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not available, using sentence-transformers only")
            
            # Initialize sentence transformer model
            try:
                self.sentence_transformer = SentenceTransformer(self.sentence_transformer_model)
                logger.info(f"Sentence transformer model loaded: {self.sentence_transformer_model}")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {e}")
                self.sentence_transformer = None
                
        except Exception as e:
            logger.error(f"Error initializing embedding models: {e}")
            raise
    
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Model to use (optional, uses default if not specified)
            
        Returns:
            List of floats representing the embedding
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        model = model or self.default_model
        
        try:
            if model.startswith("text-embedding") and self.openai_client:
                return await self._generate_openai_embedding(text, model)
            else:
                return await self._generate_sentence_transformer_embedding(text, model)
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def batch_embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            model: Model to use (optional, uses default if not specified)
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        model = model or self.default_model
        
        try:
            if model.startswith("text-embedding") and self.openai_client:
                return await self._batch_openai_embedding(texts, model)
            else:
                return await self._batch_sentence_transformer_embedding(texts, model)
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error in batch embedding: {e}")
            raise
    
    async def _generate_openai_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = await self.openai_client.embeddings.create(
                model=model,
                input=text
            )
            
            self.stats["openai_embeddings"] += 1
            self.stats["total_embeddings"] += 1
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    async def _batch_openai_embedding(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings in batch using OpenAI API."""
        try:
            response = await self.openai_client.embeddings.create(
                model=model,
                input=texts
            )
            
            self.stats["openai_embeddings"] += len(texts)
            self.stats["total_embeddings"] += len(texts)
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            raise
    
    async def _generate_sentence_transformer_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding using sentence transformers."""
        if not self.sentence_transformer:
            raise RuntimeError("Sentence transformer model not initialized")
        
        try:
            # Run in thread pool to avoid blocking
            embedding = await asyncio.to_thread(
                self.sentence_transformer.encode,
                text,
                convert_to_numpy=True
            )
            
            self.stats["sentence_transformer_embeddings"] += 1
            self.stats["total_embeddings"] += 1
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Sentence transformer embedding error: {e}")
            raise
    
    async def _batch_sentence_transformer_embedding(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings in batch using sentence transformers."""
        if not self.sentence_transformer:
            raise RuntimeError("Sentence transformer model not initialized")
        
        try:
            # Run in thread pool to avoid blocking
            embeddings = await asyncio.to_thread(
                self.sentence_transformer.encode,
                texts,
                convert_to_numpy=True
            )
            
            self.stats["sentence_transformer_embeddings"] += len(texts)
            self.stats["total_embeddings"] += len(texts)
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Sentence transformer batch embedding error: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        return {
            **self.stats,
            "default_model": self.default_model,
            "openai_available": self.openai_client is not None,
            "sentence_transformer_available": self.sentence_transformer is not None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on embedding service."""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "models": {}
        }
        
        try:
            # Test OpenAI
            if self.openai_client:
                try:
                    test_embedding = await self._generate_openai_embedding("test", "text-embedding-3-small")
                    health["models"]["openai"] = {
                        "status": "healthy",
                        "dimensions": len(test_embedding)
                    }
                except Exception as e:
                    health["models"]["openai"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health["status"] = "degraded"
            
            # Test sentence transformer
            if self.sentence_transformer:
                try:
                    test_embedding = await self._generate_sentence_transformer_embedding("test", "sentence_transformer")
                    health["models"]["sentence_transformer"] = {
                        "status": "healthy",
                        "dimensions": len(test_embedding)
                    }
                except Exception as e:
                    health["models"]["sentence_transformer"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health["status"] = "degraded"
            
            if not health["models"]:
                health["status"] = "unhealthy"
                health["error"] = "No embedding models available"
                
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health 