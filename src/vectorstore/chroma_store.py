"""ChromaDB vector store for document storage and retrieval."""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.utils.helpers import log
from config.settings import settings
from src.embeddings.embedding_model import EmbeddingModel


class ChromaStore:
    """ChromaDB vector store wrapper."""
    
    def __init__(self):
        """Initialize ChromaDB client and collection."""
        self.log = log
        self.persist_dir = settings.chroma_persist_dir
        self.collection_name = settings.collection_name
        self.embedding_model = EmbeddingModel()
        
        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document embeddings"}
            )
            
            self.log.info(f"Initialized ChromaDB collection: {self.collection_name}")
            self.log.info(f"Collection has {self.collection.count()} documents")
            
        except Exception as e:
            self.log.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            metadatas: Optional list of metadata dicts for each chunk
        """
        try:
            self.log.info(f"Adding {len(texts)} documents to vector store")
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Generate IDs
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(texts))]
            
            # Prepare metadatas
            if metadatas is None:
                metadatas = [{"chunk_id": i} for i in range(len(texts))]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            self.log.info(f"Successfully added {len(texts)} documents")
            self.log.info(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            self.log.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            top_k: Number of results to return (default from settings)
            
        Returns:
            List of dicts with 'text', 'metadata', and 'score'
        """
        try:
            if top_k is None:
                top_k = settings.top_k_results
            
            self.log.info(f"Searching for top {top_k} similar documents")
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'text': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            self.log.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            self.log.error(f"Error searching documents: {e}")
            raise
    
    def reset(self):
        """Reset the collection (delete all documents)."""
        try:
            self.log.warning("Resetting collection")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document embeddings"}
            )
            self.log.info("Collection reset successfully")
            
        except Exception as e:
            self.log.error(f"Error resetting collection: {e}")
            raise
    
    def get_count(self) -> int:
        """Get number of documents in collection."""
        return self.collection.count()
