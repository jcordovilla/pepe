"""
Unified configuration system for PEPE Discord Bot.
Manages all settings, model configurations, and environment variables.
"""
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for AI models."""
    # Language Model (Ollama)
    chat_model: str = "llama2:latest"
    chat_temperature: float = 0.0
    chat_max_tokens: int = 4096
    
    # Embedding Model 
    embedding_model: str = "all-MiniLM-L6-v2"  # Optimized for search/retrieval
    embedding_dimension: int = 384
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 300  # 5 minutes for longer responses


@dataclass
class AppConfig:
    """Main application configuration."""
    # Discord (required)
    discord_token: str
    
    # Models (required)
    models: ModelConfig
    
    # Database
    database_url: str = "sqlite:///data/discord_messages.db"
    
    # Vector Store
    faiss_index_path: str = "index_faiss"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "bot.log"
    
    # Performance
    batch_size: int = 100
    max_search_results: int = 20
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.discord_token:
            raise ValueError("DISCORD_TOKEN environment variable is required")


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    
    # Required environment variables
    discord_token = os.getenv("DISCORD_TOKEN")
    if not discord_token:
        raise ValueError("DISCORD_TOKEN environment variable is not set")
    
    # Optional environment variables with defaults
    config = AppConfig(
        discord_token=discord_token,
        database_url=os.getenv("DATABASE_URL", "sqlite:///data/discord_messages.db"),
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "index_faiss"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "bot.log"),
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "20")),
        models=ModelConfig(
            chat_model=os.getenv("CHAT_MODEL", "llama2:latest"),
            chat_temperature=float(os.getenv("CHAT_TEMPERATURE", "0.0")),
            chat_max_tokens=int(os.getenv("CHAT_MAX_TOKENS", "4096")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "msmarco-distilbert-base-v4"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_timeout=int(os.getenv("OLLAMA_TIMEOUT", "300")),
        )
    )
    
    logger.info(f"Configuration loaded successfully")
    logger.info(f"Chat model: {config.models.chat_model}")
    logger.info(f"Embedding model: {config.models.embedding_model}")
    logger.info(f"Ollama URL: {config.models.ollama_base_url}")
    
    return config


# Global configuration instance
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
