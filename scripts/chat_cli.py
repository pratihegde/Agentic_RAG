"""Interactive CLI chat interface for RAG system."""
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.chroma_store import ChromaStore
from src.workflow.graph import RAGWorkflow
from src.utils.helpers import log


class ChatCLI:
    """Interactive chat CLI for RAG system."""
    
    def __init__(self):
        """Initialize chat CLI."""
        self.vector_store = ChromaStore()
        self.workflow = RAGWorkflow(self.vector_store)
        self.chat_history = []
        
        log.info("Chat CLI initialized")
    
    def run(self):
        """Run interactive chat loop."""
        print("\n" + "=" * 80)
        print("RAG AGENTIC SYSTEM - INTERACTIVE CHAT")
        print("=" * 80)
        print(f"\nVector store has {self.vector_store.get_count()} documents")
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - Type 'quit' or 'exit' to end the session")
        print("  - Type 'save' to save chat transcript")
        print("=" * 80 + "\n")
        
        while True:
            try:
                # Get user input
                question = input("\n🤔 You: ").strip()
                
                if not question:
                    continue
                
                # Check for commands
                if question.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'save':
                    self.save_transcript()
                    continue
                
                # Run workflow
                print("\n🤖 Assistant: Processing...")
                
                result = self.workflow.run(question)
                
                # Display answer
                print("\n" + "-" * 80)
                print(result['final_answer'])
                print("-" * 80)
                
                # Save to history
                self.chat_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'question': question,
                    'answer': result['final_answer'],
                    'metadata': {
                        'confidence': result.get('confidence', 'N/A'),
                        'sources': result.get('sources', []),
                        'retries': result.get('retry_count', 0)
                    }
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                log.error(f"Error processing question: {e}")
                print(f"\n❌ Error: {e}")
    
    def save_transcript(self):
        """Save chat transcript to file."""
        if not self.chat_history:
            print("\n⚠️  No chat history to save")
            return
        
        try:
            # Create output directory
            output_dir = Path("outputs/chat_transcripts")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"chat_transcript_{timestamp}.md"
            
            # Write transcript
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# RAG System Chat Transcript\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Total Questions:** {len(self.chat_history)}\n\n")
                f.write("---\n\n")
                
                for i, entry in enumerate(self.chat_history, 1):
                    f.write(f"## Question {i}\n\n")
                    f.write(f"**Time:** {entry['timestamp']}\n\n")
                    f.write(f"**Question:** {entry['question']}\n\n")
                    f.write(f"**Answer:**\n\n{entry['answer']}\n\n")
                    f.write("---\n\n")
            
            print(f"\n✅ Transcript saved to: {filename}")
            log.info(f"Chat transcript saved: {filename}")
            
        except Exception as e:
            log.error(f"Error saving transcript: {e}")
            print(f"\n❌ Error saving transcript: {e}")


def main():
    """Main entry point."""
    chat = ChatCLI()
    
    try:
        chat.run()
    except Exception as e:
        log.error(f"Chat CLI error: {e}")
        sys.exit(1)
    finally:
        # Auto-save on exit if there's history
        if chat.chat_history:
            print("\n💾 Auto-saving transcript...")
            chat.save_transcript()


if __name__ == "__main__":
    main()
