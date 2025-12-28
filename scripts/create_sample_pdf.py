"""Create sample PDF for testing."""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Create PDF
doc = SimpleDocTemplate(
    'data/sample/ai_ml_overview.pdf',
    pagesize=letter,
    rightMargin=72,
    leftMargin=72,
    topMargin=72,
    bottomMargin=18
)

# Get styles
styles = getSampleStyleSheet()
story = []

# Read content
with open('data/sample/sample_document.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Process content
for line in content.split('\n'):
    line = line.strip()
    if not line:
        story.append(Spacer(1, 0.2*inch))
        continue
    
    if line.startswith('# '):
        text = line[2:]
        story.append(Paragraph(text, styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
    elif line.startswith('## '):
        text = line[3:]
        story.append(Paragraph(text, styles['Heading2']))
        story.append(Spacer(1, 0.15*inch))
    elif line.startswith('### '):
        text = line[4:]
        story.append(Paragraph(text, styles['Heading3']))
        story.append(Spacer(1, 0.1*inch))
    else:
        story.append(Paragraph(line, styles['BodyText']))
        story.append(Spacer(1, 0.1*inch))

# Build PDF
doc.build(story)
print('PDF created successfully: data/sample/ai_ml_overview.pdf')
