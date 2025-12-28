"""PDF document loader with high-performance text extraction using PyMuPDF (Fitz)."""
from typing import Tuple
import fitz  # PyMuPDF
from pathlib import Path
from src.utils.helpers import log


class PDFLoader:
    """Load and extract text from PDF documents using PyMuPDF for maximum reliability."""
    
    def __init__(self):
        """Initialize PDF loader."""
        self.log = log
    
    def load(self, pdf_path: str) -> Tuple[str, bool]:
        """
        Load PDF and extract text using PyMuPDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, needs_ocr)
        """
        self.log.info(f"Loading PDF with PyMuPDF: {pdf_path}")
        
        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            text = ""
            needs_ocr = False
            
            # Open PDF with fitz (PyMuPDF)
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            self.log.info(f"PDF has {num_pages} pages")
            
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                
                if page_text.strip():
                    text += page_text + "\n\n"
                else:
                    self.log.warning(f"Page {page_num + 1} appears to have no text (possibly image-based)")
            
            doc.close()
            
            # Check if we got meaningful text
            if len(text.strip()) < 100:
                self.log.warning("Very little text extracted, triggering OCR fallback")
                needs_ocr = True
            
            self.log.info(f"Extracted {len(text)} characters from PDF")
            return text, needs_ocr
            
        except Exception as e:
            self.log.error(f"Error loading PDF with PyMuPDF: {e}")
            raise
