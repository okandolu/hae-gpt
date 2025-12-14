"""Document Processor with GPT-4o-mini Guided Chunking.

This module provides intelligent document processing with GPT-guided chunking
for optimal semantic segmentation of PDF documents.

Intelligent Chunking Strategy:
    - Abstract: Single chunk
    - References: Skip
    - Tables: Single chunk (preserve structure)
    - Content: GPT-4o-mini decides optimal splits

Cost: Approximately $0.07 per 10 PDFs (500 pages)
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import PyPDF2
import pdfplumber
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

import config
from .gpt_chunker import GPTGuidedChunker

console = Console()


class GPTDocumentProcessor:
    """GPT-guided intelligent document processor.

    Processes PDF documents with intelligent chunking using either GPT-4o-mini
    for semantic segmentation or traditional uniform chunking.

    Attributes:
        use_gpt_chunking: Whether GPT-guided chunking is enabled.
        gpt_chunker: GPTGuidedChunker instance (if enabled).
        text_splitter: RecursiveCharacterTextSplitter instance (if GPT disabled).
    """

    def __init__(self, use_gpt_chunking: bool = True):
        """Initialize document processor with chunking strategy.

        Args:
            use_gpt_chunking: Use GPT-4o-mini (True) or uniform chunking (False).
        """
        self.use_gpt_chunking = use_gpt_chunking
        
        if use_gpt_chunking:
            self.gpt_chunker = GPTGuidedChunker()
            console.print("[green]✓[/green] GPT-guided chunking enabled")
        else:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                separators=config.CHUNK_SEPARATORS,
            )
            console.print("[yellow]⚠[/yellow] Using uniform chunking (GPT disabled)")
    
    def extract_text_with_metadata(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text, tables, and metadata from PDF document.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of dictionaries containing page data with text, metadata, and tables.
        """
        filename = Path(pdf_path).name
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Text extraction
                    text = page.extract_text() or ""
                    
                    # Table extraction
                    tables = []
                    has_table = False
                    if config.EXTRACT_TABLES:
                        tables = page.extract_tables()
                        has_table = len(tables) > 0
                    
                    # Add tables to text
                    if has_table:
                        table_text = "\n\n[TABLE DATA]\n"
                        for table in tables:
                            for row in table:
                                if row:
                                    table_text += " | ".join([
                                        str(cell).strip() if cell else "" 
                                        for cell in row
                                    ]) + "\n"
                        text += table_text
                    
                    pages_data.append({
                        "text": text,
                        "page": page_num,
                        "total_pages": total_pages,
                        "filename": filename,
                        "has_table": has_table,
                        "source": str(Path(pdf_path).absolute()),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    })
        
        except Exception as e:
            console.print(f"[red]✗[/red] Error processing {pdf_path}: {e}")
            raise
        
        return pages_data
    
    def process_documents(
        self, 
        file_paths: List[str],
        show_progress: bool = True
    ) -> List[Document]:
        """
        Process multiple documents with GPT-guided chunking
        """
        all_documents = []
        
        # Progress tracking
        iterator = tqdm(
            file_paths, 
            desc="Processing documents",
            disable=not show_progress
        )
        
        for file_path in iterator:
            file_path = Path(file_path)
            
            if not file_path.exists():
                console.print(f"[yellow]⚠[/yellow] File not found: {file_path}")
                continue
            
            if file_path.suffix != ".pdf":
                console.print(f"[yellow]⚠[/yellow] Skipping non-PDF: {file_path}")
                continue
            
            console.print(f"\n[cyan]Processing:[/cyan] {file_path.name}")
            
            # Extract pages
            pages_data = self.extract_text_with_metadata(str(file_path))
            
            # Process each page
            for page_data in pages_data:
                text = page_data["text"]
                
                if not text.strip():
                    continue
                
                # GPT-guided chunking
                if self.use_gpt_chunking:
                    chunks_data = self.gpt_chunker.chunk_text(
                        text=text,
                        metadata={
                            "filename": page_data["filename"],
                            "page": page_data["page"],
                            "total_pages": page_data["total_pages"],
                            "source": page_data["source"],
                            "created_at": page_data["created_at"],
                        }
                    )
                    
                    # Convert to LangChain Documents
                    for chunk_data in chunks_data:
                        doc = Document(
                            page_content=chunk_data["content"],
                            metadata={
                                **chunk_data["metadata"],
                                "chunk_strategy": chunk_data["chunk_strategy"],
                            }
                        )
                        all_documents.append(doc)
                
                else:
                    # Fallback: Uniform chunking
                    chunks = self.text_splitter.split_text(text)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "filename": page_data["filename"],
                                "page": page_data["page"],
                                "total_pages": page_data["total_pages"],
                                "chunk_id": i,
                                "has_table": page_data["has_table"],
                                "source": page_data["source"],
                                "created_at": page_data["created_at"],
                                "chunk_strategy": "uniform",
                            }
                        )
                        all_documents.append(doc)
        
        # Summary
        console.print(f"\n[green]✓[/green] Processed {len(file_paths)} files")
        console.print(f"[green]✓[/green] Created {len(all_documents)} chunks")

        # GPT statistics
        if self.use_gpt_chunking:
            console.print("\n[bold cyan]GPT-4o-mini Usage:[/bold cyan]")
            self.gpt_chunker.print_stats()
        
        return all_documents
    
    def print_summary(self, documents: List[Document]):
        """
        Processing summary with chunking strategy breakdown
        """
        if not documents:
            console.print("[yellow]No documents to summarize[/yellow]")
            return
        
        # Stats
        total_chunks = len(documents)
        unique_files = len(set(d.metadata["filename"] for d in documents))
        avg_chunk_length = sum(len(d.page_content) for d in documents) / total_chunks
        
        # Strategy breakdown
        strategy_counts = {}
        for doc in documents:
            strategy = doc.metadata.get("chunk_strategy", "unknown")
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Table
        from rich.table import Table
        
        table = Table(title="Document Processing Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Files", str(unique_files))
        table.add_row("Total Chunks", str(total_chunks))
        table.add_row("Avg Chunk Length", f"{avg_chunk_length:.0f} chars")
        
        console.print(table)
        
        # Strategy breakdown
        console.print("\n[bold cyan]Chunking Strategy Breakdown:[/bold cyan]")
        for strategy, count in sorted(strategy_counts.items()):
            pct = (count / total_chunks) * 100
            console.print(f"  {strategy:20s}: {count:4d} ({pct:5.1f}%)")
    
    def save_chunks(self, documents: List[Document], output_path: str = None):
        """
        Save chunks with strategy metadata
        """
        output_path = output_path or config.CHUNKS_PATH
        
        # Create parent directories if they don't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "content_length": len(doc.page_content),
                "chunk_strategy": doc.metadata.get("chunk_strategy", "unknown"),
            }
            for doc in documents
        ]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]✓[/green] Saved chunks to: {output_path}")


