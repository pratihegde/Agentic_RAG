"""Final Response Agent - Formats and returns the validated answer.
Optimized for performance by using zero-latency Python formatting for RAG responses.
"""
from typing import Dict, Any
from src.utils.helpers import log
from src.workflow.state import AgentState
from config.settings import settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class FinalResponseAgent:
    """Agent responsible for formatting the final response."""
    
    def __init__(self):
        """Initialize final response agent."""
        self.log = log
        # Use gpt-5-nano as requested
        self.llm = ChatOpenAI(model=settings.llm_model, temperature=0.7)
        
        # Prompt ONLY for conversational mode (Mode B)
        # RAG mode now uses zero-latency Python formatting
        conversational_prompt = """You are an intelligent Agentic RAG Assistant. 
You have access to users' documents and can answer questions or summarize them.

CONVERSATIONAL MODE:
- If the user greets you or says something like "My name is Jeff", respond warmly and acknowledge it.
- Your goal is to be a helpful document assistant. 
- If the user asks about documents and you are in this mode, it means retrieval found no specific matches, but you should still try to be helpful based on chat history.
- Never say "I don't have the document content" unless it's strictly true (e.g., no documents uploaded).

User's Original Query: {original_question}
System Interpretation: {processed_query}
"""
        
        self.conversational_prompt = ChatPromptTemplate.from_messages([
            ("system", conversational_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "User: {original_question}")
        ])

    def __call__(self, state: AgentState) -> AgentState:
        """Generate final response with performance optimization."""
        self.log.info("Final Response Agent: Generating output")
        
        try:
            generated_answer = state.get('generated_answer', '')
            original_question = state.get('original_question', state['question'])
            
            # --- OPTIMIZATION: Zero-Latency RAG Formatting ---
            if generated_answer:
                self.log.info("Mode: RAG (Fast Python Formatting)")
                
                # Determine confidence
                confidence = "High" if state.get('validation_result', False) else "Low"
                if state.get('retry_count', 0) > 0 and not state.get('validation_result'):
                    confidence = "Medium (Corrected)"
                
                # Build response directly in Python (No LLM call)
                final_answer = f"{generated_answer}\n\n"
                final_answer += "---\n"
                final_answer += f"**Metadata**\n"
                final_answer += f"- **Confidence Score**: {confidence}"
                
                state['final_answer'] = final_answer
                state['confidence'] = confidence
                
            # --- CONVERSATIONAL PATH: Requires LLM Context ---
            else:
                self.log.info("Mode: Conversational (LLM Response)")
                chain = self.conversational_prompt | self.llm
                response = chain.invoke({
                    "original_question": original_question,
                    "processed_query": state.get('processed_query', original_question),
                    "chat_history": state['chat_history']
                })
                
                state['final_answer'] = response.content
                state['confidence'] = "N/A"
            
            return state
            
        except Exception as e:
            self.log.error(f"Error in Final Response Agent: {e}")
            # Fallback to current best available answer on error
            state['final_answer'] = state.get('generated_answer', "I encountered an error generating the final response.")
            return state
