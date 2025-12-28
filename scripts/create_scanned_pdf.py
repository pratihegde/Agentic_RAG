"""Create a scanned-style PDF (image-based) for OCR testing."""
from PIL import Image, ImageDraw, ImageFont
import os

# Create an image with text (simulating a scanned document)
width, height = 2480, 3508  # A4 size at 300 DPI
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Add text content (this will be rasterized, requiring OCR to extract)
text_content = """
MACHINE LEARNING FUNDAMENTALS

Introduction to Machine Learning

Machine Learning (ML) is a subset of artificial intelligence that enables 
computers to learn from data without being explicitly programmed. ML algorithms
build mathematical models based on sample data, known as training data, to make
predictions or decisions.

Types of Machine Learning

1. Supervised Learning
   - Uses labeled training data
   - Common tasks: classification, regression
   - Examples: spam detection, price prediction

2. Unsupervised Learning
   - Works with unlabeled data
   - Finds hidden patterns
   - Examples: customer segmentation, anomaly detection

3. Reinforcement Learning
   - Learns through trial and error
   - Receives rewards or penalties
   - Examples: game playing, robotics

Key Concepts

Neural Networks: Computational models inspired by biological neural networks
Deep Learning: Neural networks with multiple layers
Feature Engineering: Selecting and transforming input variables
Model Evaluation: Assessing performance using metrics like accuracy

Applications

- Healthcare: Disease diagnosis, drug discovery
- Finance: Fraud detection, algorithmic trading
- Transportation: Autonomous vehicles, route optimization
- Natural Language: Translation, sentiment analysis

This document is designed for OCR testing purposes.
"""

# Try to use a font, fallback to default if not available
try:
    font = ImageFont.truetype("arial.ttf", 40)
    title_font = ImageFont.truetype("arial.ttf", 60)
except:
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

# Draw text on image
y_position = 200
for line in text_content.strip().split('\n'):
    if line.strip():
        # Use larger font for titles
        if line.isupper() or line.startswith('Introduction'):
            draw.text((200, y_position), line, fill='black', font=title_font)
            y_position += 100
        else:
            draw.text((200, y_position), line, fill='black', font=font)
            y_position += 60

# Save as PDF (image-based, will require OCR)
output_path = 'data/sample/scanned_ml_doc.pdf'
image.save(output_path, 'PDF', resolution=100.0)

print(f'Created scanned PDF for OCR testing: {output_path}')
print('This PDF contains rasterized text that will require OCR to extract.')
