"""Citation Formatter Module for Multi-Mode Support.

This module provides citation formatting capabilities for multiple presentation
styles and output formats, supporting both patient-friendly and academic citations.

Modes:
    - patient/patient_personalized: Simple, accessible format
    - academic: APA/Vancouver style academic citations
"""

from typing import Any, Dict, List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


class CitationFormatter:
    """Citation Formatter for Multiple Presentation Styles.

    Formats source citations in various styles appropriate for different
    audiences and output formats. Supports patient-friendly and academic
    citation formats with multiple output types.

    Supported Formats:
        - Markdown: For web and document display
        - Plain text: For terminal and simple outputs
        - JSON: For API responses and structured data
        - Rich console: For CLI with colors and panels
        - Table: For tabular display of multiple citations

    Examples:
        >>> formatter = CitationFormatter()
        >>> markdown = formatter.format_markdown(citations, mode="patient")
        >>> json_data = formatter.format_json(citations)
    """

    @staticmethod
    def format_markdown(citations: List[Dict[str, Any]], mode: str = "patient") -> str:
        """Format citations as Markdown.

        Args:
            citations: List of citation dictionaries with metadata.
            mode: Generation mode - 'patient', 'patient_personalized', or 'academic'.

        Returns:
            Markdown-formatted citation string.
        """
        if not citations:
            return "_No sources cited_"
        
        if mode == "academic":
            return CitationFormatter._format_markdown_academic(citations)
        else:
            return CitationFormatter._format_markdown_patient(citations)
    
    @staticmethod
    def _format_markdown_patient(citations: List[Dict[str, Any]]) -> str:
        """Format citations for patient mode using Vancouver style in Turkish.

        Args:
            citations: List of citation dictionaries.

        Returns:
            Turkish-formatted Markdown citation string.
        """
        lines = ["## 📚 Kaynaklar\n"]

        for i, cite in enumerate(citations, 1):
            # NEW: Use full reference if available (from Excel)
            if cite.get('reference'):
                citation_line = f"**[{i}]** {cite['reference']}"
                citation_line += f" (Sayfa {cite['page']})"
            else:
                # Fallback: Filename
                filename = cite['filename'].replace('.pdf', '')
                citation_line = f"**[{i}]** {filename}, Sayfa {cite['page']}."

            # Similarity score
            citation_line += f" [Benzerlik: {cite['similarity']:.3f}]"
            lines.append(citation_line)

            # Excerpt (English quote)
            lines.append(f"\n   > Alıntı: \"{cite['excerpt']}\"")

            # Note if table exists
            if cite.get('has_table'):
                lines.append(f"   > *Not: Tablo verisi içerir*")
            
            lines.append("")  # Blank line
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_markdown_academic(citations: List[Dict[str, Any]]) -> str:
        """Format citations for academic mode using Vancouver style.

        Args:
            citations: List of citation dictionaries.

        Returns:
            Academic-formatted Markdown citation string with full references.
        """
        lines = ["## 📚 Kaynaklar\n"]

        for i, cite in enumerate(citations, 1):
            # Use full reference if available (from Excel)
            if cite.get('reference'):
                vancouver_citation = cite['reference']
                apa_citation = f"[{i}] {vancouver_citation}"
            else:
                # Fallback: File-based format
                filename = cite['filename'].replace('.pdf', '')
                section = cite['section']
                page = cite['page']
                apa_citation = (
                    f"[{i}] *{filename}* (n.d.). "
                    f"{section}. Page {page}."
                )
            
            # Add relevance score
            apa_citation += f" [Relevance score: {cite['similarity']:.3f}]"
            
            lines.append(apa_citation)
            
            # Add excerpt as annotation
            lines.append(f"\n   > Excerpt: \"{cite['excerpt']}\"")
            
            if cite.get('has_table'):
                lines.append("   > *Note: Contains tabular data*")
            
            lines.append("")  # Blank line
        
        return "\n".join(lines)
    
    @staticmethod
    def format_plain_text(citations: List[Dict[str, Any]], mode: str = "patient") -> str:
        """Format citations as plain text.

        Args:
            citations: List of citation dictionaries.
            mode: Generation mode - 'patient' or 'academic'.

        Returns:
            Plain text formatted citation string.
        """
        if not citations:
            return "No sources cited."
        
        if mode == "academic":
            header = "REFERENCES"
        else:
            header = "SOURCES / KAYNAKLAR"
        
        lines = ["=" * 80, header, "=" * 80, ""]
        
        for i, cite in enumerate(citations, 1):
            if mode == "academic":
                # Academic: Use full reference if available
                if cite.get('reference'):
                    lines.append(f"[{i}] {cite['reference']}")
                    lines.append(f"    Relevance: {cite['similarity']:.3f}")
                else:
                    lines.append(f"[{i}] {cite['filename'].replace('.pdf', '')} (n.d.)")
                    lines.append(f"    {cite['section']}, Page {cite['page']}")
                    lines.append(f"    Relevance: {cite['similarity']:.3f}")
            else:
                # Patient-friendly: Use full reference if available
                if cite.get('reference'):
                    lines.append(f"[{i}] {cite['reference']}")
                    lines.append(f"    Sayfa: {cite['page']}")
                    lines.append(f"    Benzerlik: {cite['similarity']:.3f}")
                else:
                    lines.append(f"[{i}] {cite['filename']} (Sayfa {cite['page']})")
                    lines.append(f"    Bölüm: {cite['section']}")
                    lines.append(f"    Benzerlik: {cite['similarity']:.3f}")
            
            lines.append(f"    Excerpt: {cite['excerpt'][:150]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_json(citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format citations as JSON for API responses.

        Args:
            citations: List of citation dictionaries.

        Returns:
            Dictionary with total count and list of source metadata.
        """
        return {
            "total_sources": len(citations),
            "sources": [
                {
                    "index": i,
                    "filename": cite["filename"],
                    "page": cite["page"],
                    "section": cite["section"],
                    "similarity_score": round(cite["similarity"], 4),
                    "excerpt": cite["excerpt"],
                    "has_table": cite.get("has_table", False),
                    "reference": cite.get("reference"),  # NEW
                    "publication_year": cite.get("publication_year"),  # NEW
                }
                for i, cite in enumerate(citations, 1)
            ]
        }
    
    @staticmethod
    def format_rich_panel(citations: List[Dict[str, Any]], mode: str = "patient") -> None:
        """Display citations as rich console panels.

        Args:
            citations: List of citation dictionaries.
            mode: Generation mode - 'patient' or 'academic'.
        """
        if not citations:
            console.print("[yellow]No sources cited[/yellow]")
            return
        
        for i, cite in enumerate(citations, 1):
            if mode == "academic":
                # Academic format with full reference
                if cite.get('reference'):
                    citation_text = f"""[bold cyan]📄 Full Reference:[/bold cyan]

{cite['reference']}

[dim]File:[/dim] {cite['filename']} (Page {cite['page']})
[dim]Relevance Score:[/dim] {cite['similarity']:.3f}
{f"[yellow]⚠  Contains tabular data[/yellow]" if cite.get('has_table') else ""}

[dim]Excerpt:[/dim]
[italic]{cite['excerpt']}[/italic]
"""
                else:
                    citation_text = f"""[bold cyan]📄 {cite['filename'].replace('.pdf', '')}[/bold cyan]

[dim]Section:[/dim] {cite['section']} (Page {cite['page']})
[dim]Relevance Score:[/dim] {cite['similarity']:.3f}
{f"[yellow]⚠  Contains tabular data[/yellow]" if cite.get('has_table') else ""}

[dim]Excerpt:[/dim]
[italic]{cite['excerpt']}[/italic]
"""
            else:
                # Patient-friendly format with full reference
                if cite.get('reference'):
                    citation_text = f"""[bold cyan]📖 Tam Referans:[/bold cyan]

{cite['reference']}

[dim]Dosya:[/dim] {cite['filename']} (Sayfa {cite['page']})
[dim]Benzerlik:[/dim] {cite['similarity']:.1%}
{f"[yellow]⚠  Tablo içerir[/yellow]" if cite.get('has_table') else ""}

[dim]Alıntı (İngilizce):[/dim]
[italic]{cite['excerpt']}[/italic]
"""
                else:
                    citation_text = f"""[bold cyan]📄 {cite['filename']}[/bold cyan] (Sayfa {cite['page']})

[dim]Bölüm:[/dim] {cite['section']}
[dim]Benzerlik:[/dim] {cite['similarity']:.1%}
{f"[yellow]⚠  Tablo içerir[/yellow]" if cite.get('has_table') else ""}

[dim]Alıntı (İngilizce):[/dim]
[italic]{cite['excerpt']}[/italic]
"""
            
            panel = Panel(
                citation_text,
                title=f"[bold]{'Reference' if mode == 'academic' else 'Kaynak'} {i}[/bold]",
                border_style="cyan",
                expand=False
            )
            
            console.print(panel)
    
    @staticmethod
    def format_table(citations: List[Dict[str, Any]], mode: str = "patient") -> None:
        """Display citations as a rich table.

        Args:
            citations: List of citation dictionaries.
            mode: Generation mode - 'patient' or 'academic'.
        """
        if not citations:
            console.print("[yellow]No sources cited[/yellow]")
            return
        
        title = "📚 References" if mode == "academic" else "📚 Retrieved Sources"
        
        table = Table(
            title=title,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", style="cyan")
        table.add_column("Page", justify="center", width=6)
        table.add_column("Similarity" if mode == "academic" else "Benzerlik", 
                        justify="right", width=10)
        
        for i, cite in enumerate(citations, 1):
            # Color code by similarity
            if cite['similarity'] > 0.8:
                score_color = "green"
            elif cite['similarity'] > 0.7:
                score_color = "yellow"
            else:
                score_color = "red"
            
            # Use full reference if available, otherwise filename
            if cite.get('reference'):
                source_display = cite['reference'][:60] + "..." if len(cite['reference']) > 60 else cite['reference']
            else:
                source_display = cite['filename']
                if mode == "academic":
                    source_display = source_display.replace('.pdf', '')
            
            table.add_row(
                str(i),
                source_display,
                str(cite['page']),
                f"[{score_color}]{cite['similarity']:.3f}[/{score_color}]"
            )
        
        console.print(table)
    
    @staticmethod
    def format_inline(citations: List[Dict[str, Any]]) -> str:
        """Format citations inline in compact format.

        Args:
            citations: List of citation dictionaries.

        Returns:
            Compact inline citation string.

        Examples:
            >>> result = formatter.format_inline(citations)
            >>> print(result)
            [1: file.pdf p.5 (0.85)] [2: paper.pdf p.12 (0.78)]
        """
        if not citations:
            return ""
        
        inline_cites = []
        for i, cite in enumerate(citations, 1):
            inline_cites.append(
                f"[{i}: {cite['filename']} p.{cite['page']} ({cite['similarity']:.2f})]"
            )
        
        return " ".join(inline_cites)


if __name__ == "__main__":
    # Test
    test_citations = [
        {
            "filename": "7_AP-220224-1792.pdf",
            "page": 5,
            "section": "Introduction to Hereditary Angioedema",
            "similarity": 0.857,
            "excerpt": "Hereditary angioedema is a rare genetic disorder characterized by recurrent episodes of severe swelling...",
            "has_table": False,
            "reference": "Long LH, Fujioka T, Craig TJ, Hitomi H. Long-term outcome of Cl-esterase inhibitor deficiency. Asian Pac J Allergy Immunol. 2024;42(3):222-232. doi: 10.12932/ap-220224-1792",
            "publication_year": 2024
        },
        {
            "filename": "hae_treatment_guidelines.pdf",
            "page": 12,
            "section": "Acute Attack Management",
            "similarity": 0.823,
            "excerpt": "C1 inhibitor replacement therapy is the first-line treatment for acute HAE attacks...",
            "has_table": True,
            "reference": "Maurer M, Magerl M, Betschel S, et al. The international WAO/EAACI guideline for the management of hereditary angioedema-The 2021 revision and update. Allergy. 2022;77(7):1961-1990.",
            "publication_year": 2022
        },
        {
            "filename": "hae_pathophysiology.pdf",
            "page": 3,
            "section": "Bradykinin Pathway",
            "similarity": 0.795,
            "excerpt": "The deficiency of functional C1 inhibitor leads to uncontrolled activation of the kallikrein-kinin system...",
            "has_table": False
        }
    ]
    
    formatter = CitationFormatter()
    
    # Test Patient Mode
    console.print("\n[bold]PATIENT MODE:[/bold]")
    console.print("=" * 70)
    formatter.format_rich_panel(test_citations, mode="patient")
    
    # Test Academic Mode
    console.print("\n\n[bold]ACADEMIC MODE:[/bold]")
    console.print("=" * 70)
    formatter.format_rich_panel(test_citations, mode="academic")
    
    # Table
    console.print("\n[bold]Table Format (Patient):[/bold]")
    formatter.format_table(test_citations, mode="patient")
    
    console.print("\n[bold]Table Format (Academic):[/bold]")
    formatter.format_table(test_citations, mode="academic")
    
    # Markdown
    console.print("\n[bold]Markdown Format (Patient):[/bold]")
    md = formatter.format_markdown(test_citations, mode="patient")
    console.print(Markdown(md))
    
    console.print("\n[bold]Markdown Format (Academic):[/bold]")
    md = formatter.format_markdown(test_citations, mode="academic")
    console.print(Markdown(md))
