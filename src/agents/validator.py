"""Validator Agent - Validates answer quality and checks for hallucinations."""
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.utils.helpers import log
from src.workflow.state import AgentState
from config.settings import settings


class ValidatorAgent:
    """Agent responsible for validating generated answers."""
    
    def __init__(self):
        """Initialize validator agent with LLM."""
        self.log = log
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.0  # Use low temperature for consistent validation
        )
        
        # Create validation prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict validator that checks if an answer is properly supported by the provided context.

Your job is to determine if the answer:
1. Is factually supported by the context
2. Does not contain hallucinated information
3. Does not make claims beyond what the context provides
4. Is relevant to the question

Respond with ONLY:
- "VALID" if the answer meets all criteria
- "INVALID: [reason]" if the answer fails any criteria"""),
            ("user", """Context:
{context}

Question: {question}

Answer: {answer}

Validation:""")
        ])
        
        self.log.info("Validator Agent initialized")
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Validate the generated answer.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with validation result
        """
        self.log.info("Validator Agent: Validating answer")
        
        try:
            # Prepare context
            # Prepare validation context
            context_str = "\n\n".join([
                f"[Chunk {i+1}]: {chunk}"
                for i, chunk in enumerate(state['retrieved_chunks'])
            ])
            
            # Context Safety: Truncate context
            MAX_CONTEXT_CHARS = 12000
            if len(context_str) > MAX_CONTEXT_CHARS:
                context_str = context_str[:MAX_CONTEXT_CHARS] + "... [TRUNCATED]"

            # Validate answer
            messages = self.prompt.format_messages(
                context=context_str,
                question=state['question'],
                answer=state['generated_answer']
            )
            
            response = self.llm.invoke(messages)
            validation_response = response.content.strip()
            
            self.log.info(f"Validation response: {validation_response}")
            
            # Parse validation result
            if validation_response.startswith("VALID"):
                state['validation_result'] = True
                state['validation_reason'] = "Answer is properly supported by context"
            else:
                state['validation_result'] = False
                # Extract reason
                if ":" in validation_response:
                    reason = validation_response.split(":", 1)[1].strip()
                else:
                    reason = "Answer not properly supported by context"
                state['validation_reason'] = reason
                
                # Increment retry count
                state['retry_count'] = state.get('retry_count', 0) + 1
                
                self.log.warning(f"Validation failed: {reason} (Retry {state['retry_count']}/{state['max_retries']})")
            
            return state
            
        except Exception as e:
            self.log.error(f"Error in Validator Agent: {e}")
            # On error, mark as invalid to trigger retry
            state['validation_result'] = False
            state['validation_reason'] = f"Validation error: {str(e)}"
            state['retry_count'] = state.get('retry_count', 0) + 1
            return state
