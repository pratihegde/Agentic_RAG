"""Configuration settings for RAG Agentic System."""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    
    # LLM Configuration
    llm_model: str = "gpt-5-nano"
    embedding_model: str = "text-embedding-3-small"
    
    # Vector Store
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "rag_documents"
    
    # OCR Configuration
    use_mock_ocr: bool = False
    hf_model_name: str = "deepseek-ai/DeepSeek-OCR"
    
    # Text Processing (Semantic Chunking)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


# Global settings instance
settings = Settings()
