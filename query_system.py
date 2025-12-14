#!/usr/bin/env python3
"""
Query System (CLI)
==================
Command-line interface for RAG system

Usage:
    python query_system.py "your question here"
    python query_system.py --interactive
"""

import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel


from src import Retriever, Generator, CitationFormatter
import config

console = Console()


class RAGSystem:
    """
    Complete RAG pipeline
    """
    
    def __init__(self):
        console.print("[cyan]Loading RAG system...[/cyan]")
        
        self.retriever = Retriever()
        self.generator = Generator()
        self.formatter = CitationFormatter()
        
        console.print("[green]✓[/green] RAG system ready\n")
    
    def query(
        self,
        question: str,
        show_reasoning: bool = False,
        show_contexts: bool = False
    ) -> dict:
        """
        Execute full RAG pipeline
        
        Args:
            question: User question
            show_reasoning: Show DeepSeek R1's reasoning
            show_contexts: Show retrieved contexts
            
        Returns:
            Dict with answer, citations, metadata
        """
        # 1. Retrieve
        console.print(f"[bold cyan]Question:[/bold cyan] {question}\n")
        
        results = self.retriever.retrieve(question, k=config.TOP_K)
        
        if not results:
            console.print("[yellow]⚠ No relevant documents found[/yellow]")
            return {
                "answer": "Üzgünüm, kaynaklarda bu soruyla ilgili bilgi bulunamadı.",
                "citations": [],
                "contexts": ""
            }
        
        # Show contexts if requested
        if show_contexts:
            console.print("\n[bold cyan]Retrieved Contexts:[/bold cyan]")
            for i, (doc, score) in enumerate(results, 1):
                console.print(f"\n[dim]Context {i} (score: {score:.3f}):[/dim]")
                console.print(doc.page_content[:300] + "...")
            console.print("")
        
        # 2. Format contexts for LLM
        formatted_contexts = self.retriever.format_contexts_for_llm(results)
        
        # 3. Generate answer
        result = self.generator.generate_with_citations(
            question=question,
            retrieval_results=results,
            formatted_contexts=formatted_contexts
        )
        
        # 4. Display answer
        console.print("[bold green]Answer:[/bold green]")
        console.print(Panel(result["answer"], border_style="green"))
        
        # 5. Show reasoning if requested
        if show_reasoning and result.get("reasoning"):
            console.print("\n[bold yellow]Reasoning (DeepSeek R1's thought process):[/bold yellow]")
            console.print(Panel(result["reasoning"][:500] + "...", border_style="yellow"))
        
        # 6. Show citations
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        self.formatter.format_table(result["citations"])
        
        # 7. Token usage
        console.print(f"\n[dim]Token usage: {result['total_tokens']} total[/dim]")
        
        return result


def interactive_mode():
    """
    Interactive query mode
    """
    rag = RAGSystem()
    
    console.print("[bold cyan]Interactive Mode[/bold cyan]")
    console.print("Type your questions (or 'quit' to exit)\n")
    
    while True:
        try:
            question = console.input("[bold cyan]Question:[/bold cyan] ")
            
            if not question.strip():
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            
            console.print("")
            rag.query(question, show_reasoning=False, show_contexts=False)
            console.print("\n" + "=" * 70 + "\n")
        
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}\n")


def single_query_mode(question: str, show_reasoning: bool = False, show_contexts: bool = False):
    """
    Single query mode
    """
    rag = RAGSystem()
    rag.query(question, show_reasoning=show_reasoning, show_contexts=show_contexts)


def main():
    parser = argparse.ArgumentParser(
        description="RAG System - Query documents with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_system.py "Yapay zeka nedir?"
  python query_system.py "What is machine learning?" --reasoning
  python query_system.py --interactive
        """
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (leave empty for interactive mode)"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode"
    )
    
    parser.add_argument(
        "-r", "--reasoning",
        action="store_true",
        help="Show DeepSeek R1's reasoning process"
    )
    
    parser.add_argument(
        "-c", "--contexts",
        action="store_true",
        help="Show retrieved contexts"
    )
    
    args = parser.parse_args()
    
    # Check vector store exists
    if not config.VECTORSTORE_PATH.with_suffix(".index").exists():
        console.print("[red]✗ Vector store not found[/red]")
        console.print("\n[yellow]Please run setup first:[/yellow]")
        console.print("  python setup_pipeline.py")
        return
    
    # Route to appropriate mode
    if args.interactive or not args.question:
        interactive_mode()
    else:
        single_query_mode(
            args.question,
            show_reasoning=args.reasoning,
            show_contexts=args.contexts
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"\n[red]Fatal error:[/red] {e}")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
