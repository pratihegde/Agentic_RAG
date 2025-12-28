"""Generator Agent - Generates answers using LLM."""
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.utils.helpers import log
from src.workflow.state import AgentState
from config.settings import settings


class GeneratorAgent:
    """Agent responsible for generating answers using LLM."""
    
    def __init__(self):
        """Initialize generator agent with LLM."""
        self.log = log
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.7
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context.
            
Rules:
1. Answer ONLY based on the provided context
2. If the context doesn't contain enough information, say so
3. Be concise and accurate
4. Cite specific parts of the context when possible
5. Do not make up information"""),
            ("user", """Context:
{context}

Question: {question}

Answer:""")
        ])
        
        self.log.info("Generator Agent initialized")
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Generate answer based on retrieved chunks.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with generated answer
        """
        self.log.info("Generator Agent: Generating answer")
        
        try:
            # Prepare context from retrieved chunks
            context_str = "\n\n".join([
                f"[Chunk {i+1}]: {chunk}"
                for i, chunk in enumerate(state['retrieved_chunks'])
            ])
            
            if not context_str:
                state['generated_answer'] = "I don't have enough information to answer this question."
                return state
            
            # Context Safety: Truncate context to prevent token overflow
            # GPT-3.5-turbo-0125 has 16k context. We reserve space for system prompt + response.
            MAX_CONTEXT_CHARS = 12000  # Approx 3000-4000 tokens
            if len(context_str) > MAX_CONTEXT_CHARS:
                self.log.warning(f"Context too long ({len(context_str)} chars). Truncating to {MAX_CONTEXT_CHARS} chars.")
                context_str = context_str[:MAX_CONTEXT_CHARS] + "... [TRUNCATED]"
            
            # Generate answer using existing prompt template
            messages = self.prompt.format_messages(
                context=context_str,
                question=state['question']
            )
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            self.log.info(f"Generated answer: {answer[:100]}...")
            
            # Update state
            state['generated_answer'] = answer
            
            return state
            
        except Exception as e:
            self.log.error(f"Error in Generator Agent: {e}")
            state['generated_answer'] = f"Error generating answer: {str(e)}"
            return state
