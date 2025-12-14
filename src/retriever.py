"""Retriever Module.

This module provides intelligent document retrieval with cross-lingual support
using BGE-M3 embeddings for semantic matching between Turkish queries and
English documents.

Features:
    - BGE-M3 cross-lingual semantic search
    - Duplicate result deduplication
    - Metadata enrichment for citations
    - NotebookLM-style context formatting
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rich.console import Console


from .embedding_service import EmbeddingService
from .vector_store import FaissVectorStore
import config

console = Console()


class Retriever:
    """Intelligent retrieval system for cross-lingual RAG.

    Implements semantic search that enables Turkish queries to retrieve relevant
    English documents using BGE-M3 multilingual embeddings.

    Strategy:
        Turkish query → BGE-M3 embedding → Semantic matching → English docs

    Attributes:
        vector_store: FAISS vector store instance for similarity search.
        embedding_service: BGE-M3 embedding service for query encoding.

    Examples:
        >>> retriever = Retriever()
        >>> results = retriever.retrieve("HAE nedir?", k=5, threshold=0.5)
        >>> print(f"Found {len(results)} relevant documents")
    """

    def __init__(
        self,
        vector_store: Optional[FaissVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None
    ) -> None:
        """Initialize Retriever with vector store and embedding service.

        Args:
            vector_store: Pre-loaded FAISS vector store. If None, loads from disk.
            embedding_service: Pre-initialized embedding service. If None, creates new instance.
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service

        # Lazy load if not provided
        if self.vector_store is None:
            console.print("[cyan]Loading vector store...[/cyan]")
            self.vector_store = FaissVectorStore.load()

        if self.embedding_service is None:
            console.print("[cyan]Loading embedding service...[/cyan]")
            self.embedding_service = EmbeddingService()

        console.print("[green]✓[/green] Retriever ready")

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        deduplicate: bool = True
    ) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents for a user query.

        Embeds the query and performs semantic search in the vector store,
        optionally deduplicating near-duplicate results.

        Args:
            query: User question in Turkish or English.
            k: Number of results to return. Defaults to config.TOP_K.
            threshold: Minimum similarity score (0-1). Defaults to config.SIMILARITY_THRESHOLD.
            deduplicate: Whether to remove near-duplicate results.

        Returns:
            List of (Document, similarity_score) tuples sorted by score descending.

        Examples:
            >>> retriever = Retriever()
            >>> results = retriever.retrieve("HAE tedavisi nedir?", k=3)
            >>> for doc, score in results:
            ...     print(f"Score: {score:.3f}, File: {doc.metadata['filename']}")
        """
        k = k or config.TOP_K
        threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD
        
        # Embed query
        console.print(f"[cyan]Query:[/cyan] {query[:100]}...")
        query_embedding = self.embedding_service.embed_query(query)
        
        # Search
        results = self.vector_store.search(
            query_embedding,
            k=k * 2 if deduplicate else k,  # Get more if deduplicating
            threshold=threshold
        )
        
        # Deduplicate if needed
        if deduplicate:
            results = self._deduplicate_results(results)
            results = results[:k]  # Trim to k
        
        console.print(f"[green]✓[/green] Retrieved {len(results)} documents")
        
        # Log results
        if results:
            console.print("[cyan]Top results:[/cyan]")
            for i, (doc, score) in enumerate(results[:3], 1):
                console.print(
                    f"  {i}. {doc.metadata['filename']} "
                    f"(p.{doc.metadata['page']}) - Score: {score:.3f}"
                )
        
        return results
    
    def _deduplicate_results(
        self,
        results: List[Tuple[Document, float]],
        similarity_threshold: float = 0.95
    ) -> List[Tuple[Document, float]]:
        """Remove near-duplicate results from search results.

        Filters out overlapping chunks by comparing content similarity.
        If two chunks have >95% word overlap, only the higher-scoring one is kept.

        Args:
            results: List of (Document, score) tuples from search.
            similarity_threshold: Jaccard similarity threshold above which results
                are considered duplicates.

        Returns:
            Deduplicated list of (Document, score) tuples.

        Examples:
            >>> results = [(doc1, 0.9), (doc2, 0.85), (doc3, 0.8)]
            >>> deduplicated = retriever._deduplicate_results(results)
        """
        if len(results) <= 1:
            return results
        
        deduplicated = []
        seen_content = []
        
        for doc, score in results:
            content = doc.page_content
            
            # Check against existing
            is_duplicate = False
            for seen in seen_content:
                overlap = self._content_overlap(content, seen)
                if overlap > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append((doc, score))
                seen_content.append(content)
        
        if len(deduplicated) < len(results):
            console.print(
                f"[yellow]ℹ[/yellow] Removed {len(results) - len(deduplicated)} "
                f"duplicate chunks"
            )
        
        return deduplicated
    
    @staticmethod
    def _content_overlap(text1: str, text2: str) -> float:
        """Calculate content overlap between two text strings.

        Uses Jaccard similarity on word sets from the first 200 characters
        of each text for efficient comparison.

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Jaccard similarity score between 0.0 and 1.0.

        Examples:
            >>> overlap = Retriever._content_overlap("Hello world", "Hello there")
            >>> print(f"Overlap: {overlap:.2f}")
        """
        # Use first N characters for comparison (faster)
        n = 200
        text1_prefix = text1[:n].lower()
        text2_prefix = text2[:n].lower()
        
        # Jaccard similarity on words
        words1 = set(text1_prefix.split())
        words2 = set(text2_prefix.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def format_contexts_for_llm(
        self,
        results: List[Tuple[Document, float]],
        include_metadata: bool = True
    ) -> str:
        """Format retrieved contexts for LLM prompt.

        Creates a structured text representation of retrieved documents
        suitable for inclusion in LLM prompts, with optional metadata.

        Args:
            results: List of (Document, score) tuples from retrieval.
            include_metadata: Whether to include source metadata (filename, page, etc.).

        Returns:
            Formatted string with numbered contexts ready for LLM consumption.

        Examples:
            >>> retriever = Retriever()
            >>> results = retriever.retrieve("HAE nedir?", k=3)
            >>> formatted = retriever.format_contexts_for_llm(results)
            >>> print(formatted)
        """
        if not results:
            return "No relevant contexts found."
        
        formatted = []
        
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            
            context_block = f"\n[Context {i}]"
            
            if include_metadata:
                # NEW: Use full reference if available
                if meta.get('reference'):
                    context_block += f"\nSource: {meta['reference']}"
                    if meta.get('publication_year'):
                        context_block += f" ({meta['publication_year']})"
                else:
                    # Fallback: Filename
                    context_block += f"\nSource: {meta['filename']} (Page {meta['page']})"

                # Section info (always add)
                if meta.get('section'):
                    context_block += f"\nSection: {meta.get('section', 'Unknown')}"
                
                context_block += f"\nSimilarity: {score:.3f}"
                
                if meta.get('has_table'):
                    context_block += "\n[Contains table data]"
            
            context_block += f"\n\nContent:\n{doc.page_content}\n"
            context_block += "-" * 80
            
            formatted.append(context_block)
        
        return "\n".join(formatted)
    
    def get_context_metadata(
        self,
        results: List[Tuple[Document, float]]
    ) -> List[Dict[str, Any]]:
        """Extract metadata from retrieval results for citations.

        Processes retrieved documents and extracts relevant metadata fields
        including filename, page, section, similarity, and bibliographic info.

        Args:
            results: List of (Document, score) tuples from retrieval.

        Returns:
            List of dictionaries containing citation-ready metadata.

        Examples:
            >>> retriever = Retriever()
            >>> results = retriever.retrieve("HAE nedir?", k=3)
            >>> metadata = retriever.get_context_metadata(results)
            >>> for meta in metadata:
            ...     print(f"{meta['filename']}, page {meta['page']}")
        """
        metadata_list = []
        
        for doc, score in results:
            meta_dict = {
                "filename": doc.metadata.get("filename", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "section": doc.metadata.get("section", "Unknown Section"),
                "similarity": score,
                "excerpt": doc.page_content[:200] + "...",  # First 200 chars
                "has_table": doc.metadata.get("has_table", False),
            }

            # NEW: Add reference metadata
            if "reference" in doc.metadata:
                meta_dict["reference"] = doc.metadata["reference"]
            if "publication_year" in doc.metadata:
                meta_dict["publication_year"] = doc.metadata["publication_year"]
            
            metadata_list.append(meta_dict)
        
        return metadata_list

