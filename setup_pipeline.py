"""
Setup Pipeline
==============
Build vector store from PDF documents

Usage:
    python setup_pipeline.py

Steps:
1. Scan data/raw/ for PDFs
2. Process into chunks
3. Generate embeddings (BGE-M3)
4. Build Faiss index
5. Save vector store
"""
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


from src import EmbeddingService, FaissVectorStore
from src.document_processor import GPTDocumentProcessor
import config

console = Console()


def main():
    console.print("[bold cyan]RAG SYSTEM SETUP[/bold cyan]")
    console.print("=" * 70)
    
    # Step 1: Find PDFs
    console.print("\n[bold]Step 1: Scanning for documents...[/bold]")
    pdf_files = list(config.RAW_DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        console.print(f"[red]✗[/red] No PDF files found in {config.RAW_DATA_DIR}")
        console.print(f"\n[yellow]Please add PDF files to:[/yellow]")
        console.print(f"  {config.RAW_DATA_DIR.absolute()}")
        return
    
    console.print(f"[green]✓[/green] Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        console.print(f"  • {pdf.name}")
    
    # Step 2: Process documents
    console.print("\n[bold]Step 2: Processing documents...[/bold]")
    
    if config.USE_GPT_CHUNKING:
        processor = GPTDocumentProcessor(use_gpt_chunking=True)
        console.print("[cyan]Using GPT-4o-mini guided chunking[/cyan]")
        console.print("[yellow]⚠ This will use OpenAI API (cost: ~$0.07 per 10 PDFs)[/yellow]")
    else:
        processor = GPTDocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        console.print("[cyan]Using uniform chunking[/cyan]")
    
    documents = processor.process_documents([str(p) for p in pdf_files])
    
    if not documents:
        console.print("[red]✗[/red] No documents processed")
        return
    
    processor.print_summary(documents)
    processor.save_chunks(documents)
    
    # Step 3: Generate embeddings
    console.print("\n[bold]Step 3: Generating embeddings (BGE-M3)...[/bold]")
    console.print("[yellow]This may take a while depending on document size...[/yellow]")
    
    embedding_service = EmbeddingService()
    
    # Extract texts
    texts = [doc.page_content for doc in documents]
    
    # Embed
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Embedding documents...", total=None)
        embeddings = embedding_service.embed_documents(texts, show_progress=True)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Generated {len(embeddings)} embeddings")
    
    # Step 4: Build vector store
    console.print("\n[bold]Step 4: Building Faiss index...[/bold]")
    vector_store = FaissVectorStore(dimension=config.EMBEDDING_DIM)
    vector_store.add_documents(documents, embeddings)
    
    # Step 5: Save
    console.print("\n[bold]Step 5: Saving vector store...[/bold]")
    vector_store.save()
    vector_store.print_stats()
    
    # Done
    console.print("\n[bold green]✓ SETUP COMPLETE![/bold green]")
    console.print("=" * 70)
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Test retrieval: python src/retriever.py")
    console.print("  2. Query system: python query_system.py")
    console.print("  3. Launch UI: streamlit run app.py")
    console.print("\n[dim]Vector store location:[/dim]")
    console.print(f"  {config.VECTORSTORE_PATH.absolute()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]✗ Setup failed:[/red] {e}")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
