"""Embedding model wrapper for OpenAI embeddings."""
from typing import List
from langchain_openai import OpenAIEmbeddings
from src.utils.helpers import log
from config.settings import settings


class EmbeddingModel:
    """Wrapper for OpenAI embedding model."""
    
    def __init__(self):
        """Initialize embedding model."""
        self.log = log
        self.model_name = settings.embedding_model
        
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=settings.openai_api_key
            )
            self.log.info(f"Initialized embedding model: {self.model_name}")
            
        except Exception as e:
            self.log.error(f"Error initializing embedding model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of embedding vectors
        """
        try:
            self.log.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embeddings.embed_documents(texts)
            self.log.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            self.log.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
            
        except Exception as e:
            self.log.error(f"Error generating query embedding: {e}")
            raise
