"""LangGraph workflow definition with agentic flow."""
from typing import Dict, Any, Literal, List
from langgraph.graph import StateGraph, END
from src.workflow.state import AgentState
from src.agents.retriever import RetrieverAgent
from src.agents.generator import GeneratorAgent
from src.agents.validator import ValidatorAgent
from src.agents.final_response import FinalResponseAgent
from src.vectorstore.chroma_store import ChromaStore
from src.utils.helpers import log


class RAGWorkflow:
    """LangGraph workflow for RAG with exactly 4 agents for technical compliance."""
    
    def __init__(self, vector_store: ChromaStore):
        """Initialize RAG workflow with consolidated 4-agent setup."""
        self.log = log
        self.vector_store = vector_store
        
        # Initialize exactly 4 agents
        # 1. Smarter Retriever (Handles Rewrite + Route + Retrieve)
        self.retriever = RetrieverAgent(vector_store)
        # 2. Generator
        self.generator = GeneratorAgent()
        # 3. Validator
        self.validator = ValidatorAgent()
        # 4. Final Response Agent (Formatting + Direct Chat)
        self.final_response = FinalResponseAgent()
        
        # Build graph
        self.graph = self._build_graph()
        
        self.log.info("RAG Workflow initialized (Consolidated 4-Agent Architecture)")
    
    def _build_graph(self) -> StateGraph:
        """Build the consolidated LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add exactly 4 nodes
        workflow.add_node("retriever", self.retriever)
        workflow.add_node("generator", self.generator)
        workflow.add_node("validator", self.validator)
        workflow.add_node("final_response", self.final_response)
        
        # Entry point: Retriever now handles initial orchestration
        workflow.set_entry_point("retriever")
        
        # Edges
        # Retriever -> Conditional (Intent Routing)
        workflow.add_conditional_edges(
            "retriever",
            self._route_by_intent,
            {
                "retrieval": "generator",
                "conversational": "final_response"
            }
        )
        
        # RAG path
        workflow.add_edge("generator", "validator")
        
        workflow.add_conditional_edges(
            "validator",
            self._should_retry,
            {
                "retry": "generator",
                "finish": "final_response"
            }
        )
        
        workflow.add_edge("final_response", END)
        
        # Compile graph
        compiled_graph = workflow.compile()
        self.log.info("Consolidated LangGraph workflow compiled")
        return compiled_graph
    
    def _route_by_intent(self, state: AgentState) -> Literal["retrieval", "conversational"]:
        """Route based on intent determined by the Smart Retriever."""
        return state.get('intent', 'retrieval')
    
    def _should_retry(self, state: AgentState) -> Literal["retry", "finish"]:
        """Determine if we should retry generation or finish."""
        if state.get('validation_result', False):
            return "finish"
        
        retry_count = state.get('retry_count', 0)
        max_retries = state.get('max_retries', 2)
        
        if retry_count >= max_retries:
            return "finish"
        
        return "retry"
    
    def stream(self, question: str, chat_history: List[Any] = [], max_retries: int = 2):
        """Stream orchestration events for the workflow."""
        self.log.info(f"Streaming RAG workflow for: {question}")
        
        initial_state: AgentState = {
            'question': question,
            'original_question': question,
            'processed_query': '',
            'intent': 'retrieval',
            'chat_history': chat_history,
            'retrieved_chunks': [],
            'retrieved_metadata': [],
            'generated_answer': '',
            'validation_result': False,
            'validation_reason': '',
            'final_answer': '',
            'retry_count': 0,
            'max_retries': max_retries,
            'sources': [],
            'confidence': ''
        }
        
        # Stream events from the graph
        for event in self.graph.stream(initial_state):
            yield event

    def run(self, question: str, chat_history: List[Any] = [], max_retries: int = 2) -> Dict[str, Any]:
        """Run the RAG workflow for a question (Non-streaming)."""
        # We can just call stream and take the last result, or use invoke
        # Using invoke is more efficient for single-shot runs
        initial_state: AgentState = {
            'question': question,
            'original_question': question,
            'processed_query': '',
            'intent': 'retrieval',
            'chat_history': chat_history,
            'retrieved_chunks': [],
            'retrieved_metadata': [],
            'generated_answer': '',
            'validation_result': False,
            'validation_reason': '',
            'final_answer': '',
            'retry_count': 0,
            'max_retries': max_retries,
            'sources': [],
            'confidence': ''
        }
        return self.graph.invoke(initial_state)
