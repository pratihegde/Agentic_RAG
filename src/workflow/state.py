"""State schema for LangGraph agentic workflow."""
from typing import TypedDict, List, Optional, Dict, Any


class AgentState(TypedDict):
    """Shared state across all agents in the workflow."""
    
    # Input & Orchestration
    question: str           # The internal input (starts as original)
    original_question: str  # What the user actually typed (for UI)
    processed_query: str    # Rewritten standalone query for retrieval
    chat_history: List[Any] # List of BaseMessage
    intent: str             # 'retrieval' or 'conversational'
    
    # Retriever output
    retrieved_chunks: List[str]
    retrieved_metadata: List[Dict[str, Any]]
    
    # Generator output
    generated_answer: str
    
    # Validator output
    validation_result: bool
    validation_reason: str
    
    # Final output
    final_answer: str
    
    # Control flow
    retry_count: int
    max_retries: int
    
    # Metadata
    sources: List[str]
    confidence: str
