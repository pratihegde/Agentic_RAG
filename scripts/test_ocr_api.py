import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

def test_hf_api():
    print("Testing HuggingFace API connection...")
    
    # Check token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in environment")
        return
    
    masked_token = hf_token[:4] + "*" * (len(hf_token) - 8) + hf_token[-4:]
    print(f"Found HF_TOKEN: {masked_token}")
    
    # Model details
    model_name = "deepseek-ai/DeepSeek-OCR"
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    print(f"Testing API endpoint: {api_url}")
    
    # Create a simple test image (blank)
    from PIL import Image
    import io
    
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    try:
        print("Sending request to HuggingFace Inference API...")
        response = requests.post(api_url, headers=headers, data=img_bytes)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API connection successful!")
            print("Response:", response.json())
        elif response.status_code == 503:
            print("⚠️ Model is loading (503). This is normal for free tier.")
            print("The API is working, but the model needs time to warm up.")
        else:
            print(f"❌ API failed: {response.status_code}")
            print(f"Response (first 500 chars): {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_hf_api()
