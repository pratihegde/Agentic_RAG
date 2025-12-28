"""Test DeepSeek OCR via Gradio Client"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

def test_gradio_ocr():
    print("Testing DeepSeek OCR via Gradio Client...")
    
    try:
        from gradio_client import Client, file
        
        # Initialize client for the public demo space
        print("Connecting to merterbak/DeepSeek-OCR-Demo...")
        client = Client("merterbak/DeepSeek-OCR-Demo")
        
        print("Connected! Sending request...")
        
        # We need a sample image
        from PIL import Image
        import tempfile
        
        # Create a simple image with text
        img = Image.new('RGB', (400, 200), color='white')
        import ImageDraw
        d = ImageDraw.Draw(img)
        d.text((10,10), "Hello DeepSeek OCR via Gradio", fill='black')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img.save(f.name)
            img_path = f.name
            
        print(f"Created temp image: {img_path}")
        
        # Make prediction
        # Note: The API signature usually takes an image and additional params
        # We'll just try the standard predict
        result = client.predict(
            image=file(img_path),
            api_name="/predict"
        )
        
        print("✅ Success!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Try listing API endpoints if possible
        try:
            client.view_api()
        except:
            pass

if __name__ == "__main__":
    test_gradio_ocr()
