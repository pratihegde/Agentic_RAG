"""Text processor with Safe Semantic Chunking."""
from typing import List
import re
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.helpers import log
from config.settings import settings


class TextProcessor:
    """Process and chunk text using semantic chunking with safety constraints."""
    
    def __init__(self):
        """Initialize text processor with semantic and safety chunkers."""
        self.log = log
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        try:
            # 1. Primary: Semantic Chunker (User requirement)
            embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key
            )
            
            self.semantic_chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile"
            )
            
            # 2. Secondary: Safety Splitter (For overflow constraints)
            # Used when semantic chunks are too large
            self.safety_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            self.log.info(f"Initialized Safe Semantic Chunking (Target Size: {self.chunk_size})")
            
        except Exception as e:
            self.log.error(f"Error initializing chunkers: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using semantic chunking with recursive safety fallback.
        
        Strategy:
        1. Try Semantic Chunking first to keep meaning together.
        2. If a semantic chunk is too large (context danger), 
           force split it using Recursive splitter.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        try:
            self.log.info(f"Chunking text of length {len(text)}")
            
            # Clean text first
            cleaned_text = self.clean_text(text)
            
            # 1. Semantic Pass
            semantic_docs = self.semantic_chunker.create_documents([cleaned_text])
            self.log.info(f"Generated {len(semantic_docs)} initial semantic chunks")
            
            chunks = []
            splits_count = 0
            
            # 2. Safety Pass
            for doc in semantic_docs:
                content = doc.page_content
                
                # Check if chunk exceeds safe size (with 10% tolerance)
                if len(content) > (self.chunk_size * 1.5):
                    # Too big! Force split
                    sub_chunks = self.safety_splitter.split_text(content)
                    chunks.extend(sub_chunks)
                    splits_count += 1
                else:
                    # Safe size, keep as is
                    chunks.append(content)
            
            # 3. Post-Processing: Filter small chunks (User request: audio feedback)
            # Remove noise like page numbers ("23...", "124...")
            MIN_CHUNK_SIZE = 50
            valid_chunks = [c for c in chunks if len(c.strip()) >= MIN_CHUNK_SIZE]
            
            dropped_count = len(chunks) - len(valid_chunks)
            if dropped_count > 0:
                self.log.info(f"Dropped {dropped_count} small chunks (<{MIN_CHUNK_SIZE} chars) to ensure sizable context")
            
            self.log.info(f"Final: {len(valid_chunks)} chunks (Safe & Sizable)")
            
            # Log chunk size statistics
            if valid_chunks:
                avg_size = sum(len(c) for c in valid_chunks) / len(valid_chunks)
                max_size = max(len(c) for c in valid_chunks)
                self.log.info(f"Chunk stats - Avg: {avg_size:.0f} chars, Max: {max_size} chars")
            
            return valid_chunks
            
        except Exception as e:
            self.log.error(f"Error chunking text: {e}")
            raise
    
    def process(self, text: str) -> List[str]:
        """
        Process text: clean and chunk.
        
        Args:
            text: Raw text
            
        Returns:
            List of processed chunks
        """
        cleaned = self.clean_text(text)
        chunks = self.chunk_text(cleaned)
        return chunks
