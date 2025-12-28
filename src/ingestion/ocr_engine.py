"""OCR engine using Replicate for hosted OCR."""
from typing import Optional, List
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from src.utils.helpers import log
from config.settings import settings

# Load environment variables
load_dotenv()


class OCREngine:
    """OCR engine for extracting text from image-based PDFs using Replicate."""
    
    def __init__(self):
        """Initialize OCR engine."""
        self.log = log
        self.use_mock = settings.use_mock_ocr
        
        # Replicate Configuration
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        
        # User-specified model with version hash
        self.model_version = os.getenv("REPLICATE_MODEL_VERSION", "lucataco/deepseek-ocr:cb3b474fbfc56b1664c8c7841550bccecbe7b74c30e45ce938ffca1180b4dff5")
        
        if not self.use_mock:
            self.log.info(f"✅ Initialized OCR Engine (Replicate Mode)")
            self.log.info(f"Using Model Version: {self.model_version.split(':')[-1][:8]}...") # Log partial hash
            if not self.api_token:
                 self.log.warning("⚠️ REPLICATE_API_TOKEN not found in environment!")

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from a single image file using Replicate.
        
        Args:
            image_path: Path to the image file (jpg, png, etc.)
            
        Returns:
            Extracted text string
        """
        if self.use_mock:
            return f"[MOCKED OCR] Content of {image_path}"
            
        try:
            self.log.info(f"Extracting text from image: {image_path}")
            return self._ocr_with_replicate(image_path)
        except Exception as e:
            self.log.error(f"Image OCR failed: {e}")
            return f"[ERROR: OCR failed for image {image_path}]"

    def _ocr_with_replicate(self, image_path: str) -> str:
        """
        Use Replicate API for OCR.
        """
        import replicate
        
        try:
            self.log.info(f"Sending image to Replicate...")
            
            with open(image_path, "rb") as file:
                output = replicate.run(
                    self.model_version,
                    input={
                        "image": file,
                        "task_type": "Free OCR", # As per user snippet
                        "reference_text": "",
                        "resolution_size": "Gundam (Recommended)" # High resolution mode
                    }
                )
            
            self.log.info("✅ OCR Success (Replicate)")
            return str(output)

        except Exception as e:
            self.log.error(f"Replicate API failed: {e}")
            raise

    def _extract_images_from_pdf_pypdf(self, pdf_path: str, temp_dir: str) -> List[str]:
        """
        Extract images from PDF using pypdf (No Poppler required).
        """
        self.log.info("Attempting to extract images using pypdf (No Poppler required)")
        from pypdf import PdfReader
        
        image_paths = []
        try:
            reader = PdfReader(pdf_path)
            
            for i, page in enumerate(reader.pages):
                count = 0
                if hasattr(page, 'images'):
                    for image_file_object in page.images:
                        try:
                            image_path = os.path.join(temp_dir, f"page_{i}_img_{count}.jpg")
                            path_obj = Path(image_path)
                            
                            with open(path_obj, "wb") as fp:
                                fp.write(image_file_object.data)
                                
                            image_paths.append(image_path)
                            count += 1
                        except Exception as e:
                            self.log.warning(f"Failed to save image {count} page {i}: {e}")
            
            if not image_paths:
                self.log.warning("No images found with pypdf extraction.")
                
            return image_paths
            
        except Exception as e:
            self.log.error(f"pypdf extraction failed: {e}")
            return []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from image-based PDF using OCR.
        """
        if self.use_mock:
            return self._mock_ocr(pdf_path)
            
        try:
            self.log.info(f"Running OCR on: {pdf_path}")
            
            import tempfile
            from pdf2image import convert_from_path
            
            # Create temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                image_paths = []
                
                # 1. Try pdf2image (requires Poppler) - kept for users who have it
                try:
                    images = convert_from_path(pdf_path)
                    self.log.info("Converted PDF with pdf2image (Poppler found)")
                    for i, image in enumerate(images):
                        image_path = os.path.join(temp_dir, f"page_{i}.jpg")
                        image.save(image_path, 'JPEG')
                        image_paths.append(image_path)
                except Exception:
                    # Silent fallback
                    image_paths = self._extract_images_from_pdf_pypdf(pdf_path, temp_dir)
                
                if not image_paths:
                    self.log.error("Could not extract any images from PDF!")
                    return ""
                
                # Process images
                all_text = []
                for i, image_path in enumerate(image_paths):
                    self.log.info(f"Processing image {i + 1}/{len(image_paths)} with Replicate")
                    
                    try:
                        text = self._ocr_with_replicate(image_path)
                        all_text.append(text)
                    except Exception as e:
                        self.log.error(f"OCR failed for image {i+1}: {e}")
                        all_text.append(f"[ERROR: OCR failed for Page {i+1}]")
            
            return "\n\n".join(all_text)

        except Exception as e:
            self.log.error(f"Critical OCR Error: {e}")
            self.log.warning("Falling back to simulated output")
            return self._mock_ocr(pdf_path)

    def _mock_ocr(self, pdf_path: str) -> str:
        """Mock OCR for testing purposes."""
        self.log.info(f"Using mock OCR for: {pdf_path}")
        return f"""
        [MOCK OCR OUTPUT - Replicate Simulation]
        
        This content simulates the output of OCR processing for {Path(pdf_path).name}.
        """
