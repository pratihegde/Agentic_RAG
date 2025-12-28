# Quick Start Guide

## 🚀 Get Started in 4 Steps

### Step 1: Add API Keys

Edit the `.env` file in the project root:

```env
# Required for LLM and embeddings
OPENAI_API_KEY=sk-your-actual-openai-key-here

# Optional: For free hosted OCR (no 6.67GB download!)
HF_TOKEN=hf_your_huggingface_token_here
USE_MOCK_OCR=false
```

**Get HuggingFace Token (Free):**
1. Visit: https://huggingface.co/settings/tokens
2. Click "New token"
3. Copy and paste into `.env`

### Step 2: Ingest a Document

```bash
uv run python scripts/ingest_document.py data/sample/sample_english.pdf --reset
```

This will:
- Load the PDF
- Create semantic chunks
- Generate embeddings
- Store in ChromaDB

### Step 3: Start Chatting!

**Option A: CLI Chat**
```bash
uv run python scripts/chat_cli.py
```

**Option B: Streamlit Web App**
```bash
uv run streamlit run streamlit_app/app.py
```

## 📝 Sample Questions to Try

- "What is machine learning?"
- "Explain the types of AI"
- "What are the applications of deep learning?"
- "How does supervised learning work?"

## 🧪 Test OCR Functionality

We've created a scanned PDF for OCR testing:

```bash
uv run python scripts/ingest_document.py data/sample/scanned_ml_doc.pdf
```

This PDF contains rasterized text that requires OCR to extract.

## 📊 Generate Chat Transcript

1. Run the CLI chat: `uv run python scripts/chat_cli.py`
2. Ask 5-7 questions
3. Type `save` to save the transcript
4. Find it in `outputs/chat_transcripts/`

## ✅ Submission Checklist

- [x] Source code (complete)
- [x] README.md (complete)
- [x] Sample documents (complete)
- [ ] Chat transcript (generate after API key setup)
- [x] .env.example (complete)
- [x] Streamlit app (complete)

## 🎯 What Makes This Special

1. **Semantic Chunking** - Not just fixed-size chunks
2. **4-Agent LangGraph Workflow** - Retriever → Generator → Validator → Final Response
3. **Retry Logic** - Automatically improves invalid answers
4. **Validation** - LLM checks for hallucinations
5. **Both CLI & Web UI** - Flexibility for different use cases

## 🐛 Troubleshooting

**Error: "invalid_api_key"**
- Solution: Add your real OpenAI API key to `.env`

**Error: "Collection has 0 documents"**
- Solution: Run the ingestion script first

**OCR model fails to load**
- Expected: System falls back to mock OCR automatically
- To use real OCR: Set `USE_MOCK_OCR=false` in `.env`

---

**Need help?** Check the full [README.md](README.md) for detailed documentation.
