"""Streamlit web application for RAG system."""
import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import tempfile
from langchain_core.messages import HumanMessage, AIMessage

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.ocr_engine import OCREngine
from src.ingestion.text_processor import TextProcessor
from src.vectorstore.chroma_store import ChromaStore
from src.workflow.graph import RAGWorkflow


# Page config
st.set_page_config(
    page_title="Agentic RAG Workflow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Midnight Blue & Orange Theme
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FF4B2B; /* Vibrant Orange */
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Sidebar Styling - Midnight Grey */
    [data-testid="stSidebar"] {
        background-color: #1E2530;
        border-right: 1px solid #333;
    }
    
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #E0E0E0 !important;
    }

    /* Buttons (Primary) */
    .stButton button[kind="primary"] {
        width: 100%;
        border-radius: 8px;
        background-color: #FF4B2B;
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #FF6B4B;
        box-shadow: 0 4px 12px rgba(255, 75, 43, 0.3);
    }

    /* Subtle Secondary Buttons (Starters) */
    div[data-testid="stBaseButton-secondary"] button {
        background-color: transparent !important;
        border: 1px solid #444 !important;
        color: #AAA !important;
        text-transform: none !important;
    }
    
    div[data-testid="stBaseButton-secondary"] button:hover {
        border-color: #FF4B2B !important;
        color: white !important;
        background-color: rgba(255, 75, 43, 0.05) !important;
    }
    
    /* Ensure starters don't look orange on mobile/small screens */
    .stButton button[kind="secondary"] {
        background-color: transparent !important;
    }

    /* Input Fields */
    .stTextInput>div>div>input {
        border-radius: 8px;
        background-color: #1E2530;
        color: white;
        border: 1px solid #333;
    }
    
    /* Chat Bubbles Styling */
    .stChatMessage {
        background-color: transparent !important;
    }
    
    /* Custom divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #333, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (Fixed Optimization)
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = ChromaStore()
    st.session_state.workflow = RAGWorkflow(st.session_state.vector_store)
    st.session_state.pdf_loader = PDFLoader()
    st.session_state.ocr_engine = OCREngine()
    st.session_state.text_processor = TextProcessor()
    
    st.session_state.chat_history = []
    st.session_state.ingestion_complete = False
    st.session_state.cancel_ingestion = False


def ingest_document(uploaded_file):
    """Ingest uploaded PDF or Image with Cancellation Support."""
    st.session_state.cancel_ingestion = False
    try:
        # Save temp file
        ext = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        pdf_loader = st.session_state.pdf_loader
        ocr_engine = st.session_state.ocr_engine
        text_processor = st.session_state.text_processor
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- CANCELLATION BUTTON ---
        cancel_placeholder = st.empty()
        if cancel_placeholder.button("🔴 Cancel Ingestion", key="cancel_btn"):
            st.session_state.cancel_ingestion = True
            
        def check_cancel():
            if st.session_state.cancel_ingestion:
                st.warning("⚠️ Ingestion Cancelled by User.")
                return True
            return False

        text = ""
        # Step 1: Extraction
        if ext == '.pdf':
            status_text.text("Loading PDF (PyMuPDF)...")
            progress_bar.progress(10)
            text, needs_ocr = pdf_loader.load(tmp_path)
            
            if needs_ocr and not check_cancel():
                status_text.text("Running DeepSeek OCR on PDF pages...")
                progress_bar.progress(20)
                
                # Manual loop over PDF for cancellation support
                reader = PDFLoader() # Standard check
                import fitz
                doc = fitz.open(tmp_path)
                all_pages_text = []
                
                # Check for cancellation within page loop
                with tempfile.TemporaryDirectory() as temp_dir:
                    for i in range(len(doc)):
                        if check_cancel(): break
                        status_text.text(f"OCR: Processing Page {i+1}/{len(doc)}")
                        progress_bar.progress(int(20 + (i/len(doc)) * 40))
                        
                        # Extract page as image
                        page = doc.load_page(i)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # High res
                        img_path = os.path.join(temp_dir, f"page_{i}.jpg")
                        pix.save(img_path)
                        
                        page_text = ocr_engine.extract_text_from_image(img_path)
                        all_pages_text.append(page_text)
                    
                    text = "\n\n".join(all_pages_text)
                doc.close()
        
        elif ext in ['.png', '.jpg', '.jpeg']:
            status_text.text("Processing Image (DeepSeek OCR)...")
            progress_bar.progress(30)
            text = ocr_engine.extract_text_from_image(tmp_path)
            progress_bar.progress(60)

        # Step 2: Chunking & Indexing
        if not check_cancel() and text:
            status_text.text("Semantic Chunking...")
            chunks = text_processor.process(text)
            progress_bar.progress(80)
            
            if not check_cancel():
                status_text.text("Updating Knowledge Base...")
                metadatas = [{'source': uploaded_file.name} for _ in range(len(chunks))]
                st.session_state.vector_store.add_documents(chunks, metadatas)
                progress_bar.progress(100)
                status_text.text("✅ Content Ingested!")
                cancel_placeholder.empty()
                return True, len(chunks)
        
        # Cleanup
        Path(tmp_path).unlink()
        cancel_placeholder.empty()
        return False, 0
        
    except Exception as e:
        st.error(f"Error during ingestion: {e}")
        return False, 0


def main():
    """Main Streamlit app."""
    
    # Header Section
    st.markdown('<div class="main-header">🤖 Agentic RAG Workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An intelligent RAG system featuring query rewriting, agentic routing, and multi-agent self-correction</div>', unsafe_allow_html=True)
    
    # Sidebar - Document Center & System Info
    with st.sidebar:
        st.title("📂 Document Center")
        
        # Document count
        doc_count = st.session_state.vector_store.get_count()
        st.metric("Chunks in Vector Index", doc_count)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # File upload
        st.subheader("Ingest Knowledge")
        uploaded_file = st.file_uploader(
            "Upload Files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            label_visibility="collapsed",
            help="Add documents (PDF or Image) to the agent's knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Document"):
                with st.spinner("Processing..."):
                    success, num_chunks = ingest_document(uploaded_file)
                    if success:
                        st.success(f"Successfully processed {num_chunks} chunks!")
                        st.session_state.ingestion_complete = True
                        st.rerun()
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Reset button
        if st.button("🗑️ Wipe Vector Store"):
            if st.session_state.vector_store.get_count() > 0:
                st.session_state.vector_store.reset()
                st.session_state.chat_history = []
                st.success("Store wiped!")
                st.rerun()
        
        # --- TRANSCRIPT EXPORT (Promoted visibility) ---
        if st.session_state.chat_history:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.subheader("📥 Export Trace")
            
            transcript = "# Agentic RAG Trace\n\n"
            transcript += f"**Session Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            transcript += "---\n\n"
            
            for i, entry in enumerate(st.session_state.chat_history, 1):
                role = "USER" if entry['role'] == 'user' else "AGENT"
                transcript += f"### {role} {i}: {entry['content']}\n\n"
                if 'thinking' in entry:
                    t = entry['thinking']
                    transcript += f"> **Intent**: {t.get('intent', 'N/A')}\n"
                    transcript += f"> **Internal Query**: {t.get('query', 'N/A')}\n\n"
            
            st.download_button(
                label="📥 Download Trace",
                data=transcript,
                file_name=f"rag_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
                key="trace_download_btn"
            )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Restored Original About Section
        st.subheader("ℹ️ About System")
        st.info("""
        This RAG system uses:
        - **Workflow**: LangGraph (4-Agent)
        - **OCR**: DeepSeek-OCR
        - **LLM**: gpt-5-nano
        - **Database**: ChromaDB
        
        **Active Agents:**
        1. Smart Retriever (Rewrite + Route)
        2. Generator
        3. Validator
        4. Final Response
        """)
    
    # Main Chat Area
    st.subheader("💬 Chat with your docs")
    
    # --- SUBTLE CONVERSATION STARTERS ---
    if not st.session_state.chat_history:
        st.write("Click a starter to begin:")
        cols = st.columns(3)
        starters = ["👋 Hello!", "📄 Summarize document", "💡 Key takeaways"]
        for i, starter in enumerate(starters):
            # Use type="secondary" for subtlety
            if cols[i].button(starter, use_container_width=True, type="secondary"):
                st.session_state.chat_history.append({"role": "user", "content": starter})
                st.rerun()

    # Display chat history
    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])
            
            # --- REAL-TIME THINKING DATA (Persistent) ---
            if "thinking" in msg and msg["role"] == "assistant":
                with st.expander("🧠 Agentic Trace"):
                    t = msg["thinking"]
                    st.write(f"**Intent**: `{t['intent'].upper()}`")
                    st.write(f"**Internal Query**: `{t['query']}`")
                    if t['chunks']:
                        st.write("**Retrieved Context**:")
                        for i, c in enumerate(t['chunks']):
                            st.info(f"Chunk {i+1}: {c[:300]}...")
                    else:
                        st.write("*No direct document retrieval.*")

    # Chat Input
    if prompt := st.chat_input("Ask anything..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()

    # Handling the latest message (if assistant hasn't responded yet)
    if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
        latest_prompt = st.session_state.chat_history[-1]['content']
        
        with st.chat_message("assistant"):
            # --- REAL-TIME STATUS (The "Under the Hood" Experience) ---
            with st.status("⚙️ Agentic Workflow Running...", expanded=True) as status_box:
                history_for_agent = []
                for m in st.session_state.chat_history[:-1]:
                    if m['role'] == 'user':
                        history_for_agent.append(HumanMessage(content=m['content']))
                    else:
                        history_for_agent.append(AIMessage(content=m['content']))

                final_state = None
                # Stream the workflow for live updates
                for event in st.session_state.workflow.stream(latest_prompt, history_for_agent):
                    node_name = list(event.keys())[0]
                    state = event[node_name]
                    final_state = state # Track last state
                    
                    if node_name == "retriever":
                        status_box.update(label=f"✅ Intent: {state['intent'].upper()}")
                        st.write(f"**Internal Query**: `{state['processed_query']}`")
                        if state['intent'] == 'retrieval':
                            st.write("🔎 *Searching vector store for relevant chunks...*")
                    
                    elif node_name == "generator":
                        status_box.update(label="✍️ Generating Answer...")
                        st.write("📝 *Synthesizing response from retrieved content...*")
                    
                    elif node_name == "validator":
                        status_box.update(label="⚖️ Validating Content...")
                        if state.get('validation_result'):
                            st.write("✅ *Response verified against sources.*")
                        else:
                            st.write(f"⚠️ *Validation failed ({state.get('validation_reason')}). Retrying...*")
                    
                    elif node_name == "final_response":
                        status_box.update(label="🏁 Finalizing Response...", state="complete")
                
                # Show Final Answer
                if final_state:
                    st.markdown(final_state['final_answer'])
                    
                    # Save to history with thinking data
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": final_state['final_answer'],
                        "thinking": {
                            "query": final_state.get('processed_query', latest_prompt),
                            "chunks": final_state.get('retrieved_chunks', []),
                            "intent": final_state.get('intent', 'conversational')
                        }
                    })
                    st.rerun()


if __name__ == "__main__":
    main()
