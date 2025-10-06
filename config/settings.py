"""
Configuration management for the agentic RAG system
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration class for the agentic RAG system."""
    
    # API Keys
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "")
    
    # Database Configuration
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./data/chromadb")
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_base")
    
    # Model Configuration
    model_name: str = os.getenv("MODEL_NAME", "models/gemini-2.5-flash")
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))
    top_p: float = float(os.getenv("TOP_P", "0.95"))
    
    # Memory Configuration
    max_memory_messages: int = int(os.getenv("MAX_MEMORY_MESSAGES", "50"))
    context_window: int = int(os.getenv("CONTEXT_WINDOW", "32000"))
    
    # Document Processing Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # RLHF Configuration
    enable_rlhf: bool = os.getenv("ENABLE_RLHF", "true").lower() == "true"
    feedback_collection_rate: float = float(os.getenv("FEEDBACK_COLLECTION_RATE", "0.3"))
    
    # Search Configuration
    duckduckgo_timeout: int = int(os.getenv("DUCKDUCKGO_TIMEOUT", "10"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "./data/logs/app.log")
    
    # Streamlit Configuration
    streamlit_port: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    streamlit_address: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.google_api_key:
            raise ValueError("Google API key is required")
        return True


# Global configuration instance
config = Config()