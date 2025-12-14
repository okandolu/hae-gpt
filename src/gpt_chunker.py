"""GPT-Guided Intelligent Chunking Module.

This module provides GPT-4o-mini guided intelligent text chunking for optimal
semantic segmentation of documents for RAG systems.

Chunking Rules:
    1. Abstract: Single chunk
    2. References: Skip
    3. Tables: Single chunk (preserve structure)
    4. Other content: GPT-4o-mini decides optimal splits

Cost: Approximately $0.07 per 10 PDFs (500 pages)
"""

import json
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

import config

console = Console()


class GPTGuidedChunker:
    """Intelligent text chunking with GPT-4o-mini.

    Uses GPT-4o-mini to determine optimal semantic boundaries for splitting
    documents into meaningful chunks for RAG systems.

    Attributes:
        client: OpenAI client for GPT-4o-mini API.
        total_tokens_used: Running count of tokens consumed.
        total_cost: Estimated total cost of chunking operations.
    """

    SYSTEM_PROMPT = """You are an expert at splitting academic/technical documents into meaningful chunks for RAG systems.

Your task: Analyze the text and decide WHERE to split it into chunks.

Rules:
1. Each chunk should be semantically complete (one coherent idea/topic)
2. Prefer natural boundaries: section breaks, paragraph breaks, topic changes
3. Optimal chunk size: 500-1500 characters (but prioritize meaning over size)
4. Don't break mid-sentence or mid-paragraph unless absolutely necessary

Output format (JSON):
{
  "split_indices": [0, 523, 1247, 2103],
  "reasoning": "Split at section boundary, then at topic change..."
}

split_indices = character positions where splits should occur (start of each chunk)
"""

    USER_PROMPT_TEMPLATE = """Analyze this text and decide where to split it into chunks:

TEXT:
{text}

CHARACTER COUNT: {char_count}

Decide the optimal split points. Return JSON with split_indices."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key required for GPT-guided chunking")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Stats
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        console.print("[green]✓[/green] GPT-4o-mini chunker initialized")
    
    def _classify_section(self, text: str) -> str:
        """Classify document section type without calling GPT.

        Args:
            text: Text content to classify.

        Returns:
            Section type: 'abstract', 'references', 'appendix', or 'content'.
        """
        text_lower = text.lower().strip()[:200]  # First 200 chars

        # Abstract detection
        if re.match(r"^(abstract|özet)\s*[:|\n]", text_lower):
            return "abstract"
        
        # References detection
        if re.match(r"^(references|bibliography|kaynaklar)", text_lower):
            return "references"
        
        # Appendix detection
        if re.match(r"^(appendix|ek)\s*[:\d]", text_lower):
            return "appendix"
        
        return "content"
    
    def _has_table_markers(self, text: str) -> bool:
        """Detect if text contains table markers.

        Args:
            text: Text content to check.

        Returns:
            True if table markers are detected, False otherwise.
        """
        table_markers = [
            "[TABLE DATA]",
            "| --- | --- |",
            "\t|\t",  # Tab-separated
        ]
        
        return any(marker in text for marker in table_markers)
    
    def _ask_gpt_for_splits(self, text: str) -> List[int]:
        """
        Ask GPT-4o-mini: Where should we split this text?
        """
        # First do rough split for very long text (GPT context limit)
        MAX_CHARS = 12000  # ~3K tokens

        if len(text) > MAX_CHARS:
            # Analyze first MAX_CHARS, then recurse
            splits = self._ask_gpt_for_splits(text[:MAX_CHARS])
            remaining_splits = self._ask_gpt_for_splits(text[MAX_CHARS:])
            return splits + [s + MAX_CHARS for s in remaining_splits]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self.USER_PROMPT_TEMPLATE.format(
                            text=text,
                            char_count=len(text)
                        )
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500,
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            split_indices = result.get("split_indices", [0])
            
            # Stats
            usage = response.usage
            self.total_tokens_used += usage.total_tokens
            self.total_cost += (usage.prompt_tokens * 0.15 / 1_000_000 + 
                               usage.completion_tokens * 0.60 / 1_000_000)
            
            return split_indices
        
        except Exception as e:
            console.print(f"[red]✗[/red] GPT error: {e}")
            # Fallback: simple split every 1000 chars
            return list(range(0, len(text), 1000))
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Main chunking function
        
        Returns:
            List of {"content": str, "metadata": dict, "chunk_strategy": str}
        """
        metadata = metadata or {}
        section_type = self._classify_section(text)

        # Rule 1: Abstract → Single chunk
        if section_type == "abstract":
            console.print(f"[cyan]→[/cyan] Abstract detected (single chunk)")
            return [{
                "content": text,
                "metadata": {**metadata, "section_type": "abstract"},
                "chunk_strategy": "single_abstract"
            }]
        
        # Rule 2: References/Appendix → SKIP
        if section_type in ["references", "appendix"]:
            console.print(f"[yellow]⊘[/yellow] {section_type.title()} detected (skipped)")
            return []

        # Rule 3: Table → Single chunk (don't split!)
        if self._has_table_markers(text):
            console.print(f"[cyan]→[/cyan] Table detected (preserving as single chunk)")
            return [{
                "content": text,
                "metadata": {**metadata, "has_table": True},
                "chunk_strategy": "single_table"
            }]

        # Rule 4: Normal content → Ask GPT
        console.print(f"[cyan]→[/cyan] Asking GPT-4o-mini for optimal splits...")
        split_indices = self._ask_gpt_for_splits(text)
        
        # Create chunks
        chunks = []
        for i in range(len(split_indices)):
            start = split_indices[i]
            end = split_indices[i + 1] if i + 1 < len(split_indices) else len(text)
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": i,
                        "total_chunks": len(split_indices),
                    },
                    "chunk_strategy": "gpt_guided"
                })
        
        console.print(f"[green]✓[/green] Created {len(chunks)} GPT-guided chunks")
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Token usage and cost statistics
        """
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_cost_per_page": round(self.total_cost / max(1, self.total_tokens_used / 650), 4),
        }
    
    def print_stats(self):
        """
        Display statistics nicely
        """
        from rich.table import Table
        
        stats = self.get_stats()
        
        table = Table(title="GPT Chunking Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Tokens Used", f"{stats['total_tokens']:,}")
        table.add_row("Total Cost", f"${stats['total_cost_usd']:.4f}")
        table.add_row("Avg Cost/Page", f"${stats['avg_cost_per_page']:.4f}")
        
        console.print(table)
