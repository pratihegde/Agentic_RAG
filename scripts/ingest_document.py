"""CLI script to ingest PDF documents into the vector store."""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.ocr_engine import OCREngine
from src.ingestion.text_processor import TextProcessor
from src.vectorstore.chroma_store import ChromaStore
from src.utils.helpers import log


def ingest_document(pdf_path: str, reset: bool = False):
    """
    Ingest a PDF document into the vector store.
    
    Args:
        pdf_path: Path to PDF file
        reset: Whether to reset the vector store before ingesting
    """
    log.info("=" * 80)
    log.info("DOCUMENT INGESTION PIPELINE")
    log.info("=" * 80)
    
    try:
        # Initialize components
        pdf_loader = PDFLoader()
        ocr_engine = OCREngine()
        text_processor = TextProcessor()
        vector_store = ChromaStore()
        
        # Reset if requested
        if reset:
            log.warning("Resetting vector store...")
            vector_store.reset()
        
        # Step 1: Load PDF
        log.info("\n[Step 1/4] Loading PDF...")
        text, needs_ocr = pdf_loader.load(pdf_path)
        
        # Step 2: OCR if needed
        if needs_ocr:
            log.info("\n[Step 2/4] Running OCR (PDF appears to be image-based)...")
            text = ocr_engine.extract_text_from_pdf(pdf_path)
        else:
            log.info("\n[Step 2/4] Skipping OCR (text-based PDF)")
        
        # Step 3: Process and chunk text
        log.info("\n[Step 3/4] Processing and chunking text...")
        chunks = text_processor.process(text)
        
        # Step 4: Add to vector store
        log.info("\n[Step 4/4] Adding chunks to vector store...")
        
        # Create metadata for each chunk
        metadatas = [
            {
                'chunk_id': i,
                'source': Path(pdf_path).name,
                'total_chunks': len(chunks)
            }
            for i in range(len(chunks))
        ]
        
        vector_store.add_documents(chunks, metadatas)
        
        log.info("\n" + "=" * 80)
        log.info("INGESTION COMPLETE!")
        log.info(f"Document: {Path(pdf_path).name}")
        log.info(f"Chunks created: {len(chunks)}")
        log.info(f"Total documents in store: {vector_store.get_count()}")
        log.info("=" * 80)
        
    except Exception as e:
        log.error(f"\nIngestion failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the RAG vector store"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to PDF file to ingest"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store before ingesting"
    )
    
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_file = Path(args.pdf_path)
    if not pdf_file.exists():
        log.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    if not pdf_file.suffix.lower() == '.pdf':
        log.error(f"File is not a PDF: {args.pdf_path}")
        sys.exit(1)
    
    # Run ingestion
    ingest_document(args.pdf_path, args.reset)


if __name__ == "__main__":
    main()
