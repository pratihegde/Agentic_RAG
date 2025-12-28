"""Smart Retriever Agent - Entry point for RAG workflow.
Consolidates Query Rewriting, Intent Routing, and Document Retrieval into a single node.
"""
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.vectorstore.chroma_store import ChromaStore
from src.utils.helpers import log
from src.workflow.state import AgentState
from config.settings import settings


class RetrieverAgent:
    """Agent responsible for orchestration (rewriting/routing) and retrieval."""
    
    def __init__(self, vector_store: ChromaStore):
        """Initialize orchestrating retriever agent."""
        self.vector_store = vector_store
        self.log = log
        
        # Use gpt-5-nano as requested
        self.llm = ChatOpenAI(model=settings.llm_model, temperature=0)
        
        # Combined Orchestration Prompt
        orchestration_prompt = """You are the Orchestrator for an Agentic RAG system.
Your job is to analyze the user's question and determine two things:
1. INTENT: Is this a conversation/small talk ('conversational') OR does it require document/file knowledge ('retrieval')?
2. PROCESSED QUERY: Resolve context (it, that, they) using history to create a standalone search term.

ROUTING RULES:
- If the user mentions "summary", "summarize", "document", "the file", "research", or asks about core content -> ALWAYS 'retrieval'.
- Statements like "My name is Jeff" or "Hello" -> 'conversational'.
- If in doubt, choose 'retrieval' to be safe.

REWRITING RULES:
- DO NOT turn statements into questions. If I say "My name is Jeff", the processed query should be "My name is Jeff".
- Only resolve ambiguous pronouns like 'it' based on previous turns.

Output your response as a JSON object with:
- intent: 'retrieval' or 'conversational'
- processed_query: str
"""
        
        self.orchestrator_prompt = ChatPromptTemplate.from_messages([
            ("system", orchestration_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{original_question}")
        ])
    
    def __call__(self, state: AgentState) -> AgentState:
        """Process the question, route, and retrieve if necessary."""
        self.log.info(f"Smart Retriever: Analyzing input [ {state['question']} ]")
        
        try:
            # 1. Orchestration Call (Rewrite + Route)
            chain = self.orchestrator_prompt | self.llm
            response = chain.invoke({
                "original_question": state['question'],
                "chat_history": state['chat_history']
            })
            
            # Parse response
            try:
                # Handle cases where model might wrap JSON in backticks
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                orchestration = json.loads(content)
                state['intent'] = orchestration.get('intent', 'retrieval')
                state['processed_query'] = orchestration.get('processed_query', state['question'])
                state['original_question'] = state['question'] # Save what they typed
                
                self.log.info(f"Orchestration Result: Intent={state['intent']} | Processed Query={state['processed_query']}")
                
            except Exception as parse_err:
                self.log.error(f"Failed to parse orchestration JSON: {parse_err}")
                state['intent'] = 'retrieval'
                state['processed_query'] = state['question']
                state['original_question'] = state['question']
            
            # 2. Retrieval Case
            if state['intent'] == 'retrieval':
                self.log.info(f"Performing retrieval for: {state['processed_query']}")
                chunks_with_metadata = self.vector_store.similarity_search(
                    state['processed_query'],
                    top_k=settings.top_k_results
                )
                
                if not chunks_with_metadata:
                    self.log.warning("No relevant chunks found")
                    state['retrieved_chunks'] = []
                    state['retrieved_metadata'] = []
                else:
                    # Filter noise
                    MIN_CHUNK_SIZE = 50
                    valid_chunks = []
                    valid_metadata = [] 
                    for doc in chunks_with_metadata:
                        content = doc['text'].strip()
                        if len(content) >= MIN_CHUNK_SIZE:
                            valid_chunks.append(content)
                            valid_metadata.append(doc['metadata'])
                    
                    state['retrieved_chunks'] = valid_chunks
                    state['retrieved_metadata'] = valid_metadata
                    self.log.info(f"Retrieved {len(valid_chunks)} valid chunks")
                    
                    # Verbose logging for debugging
                    for i, chunk in enumerate(valid_chunks):
                        self.log.info(f"Retrieved Chunk {i+1} Snippet: {chunk[:200]}...")
            else:
                self.log.info("Skipping retrieval (Conversational path)")
                state['retrieved_chunks'] = []
                state['retrieved_metadata'] = []
                
            return state
            
        except Exception as e:
            self.log.error(f"Error in Smart Retriever: {e}")
            state['intent'] = 'conversational'
            state['processed_query'] = state['question']
            state['original_question'] = state['question']
            state['retrieved_chunks'] = []
            state['retrieved_metadata'] = []
            return state
