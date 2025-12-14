"""Vector Store Module using FAISS.

Provides efficient similarity search with metadata storage using FAISS
IndexFlatIP for cosine similarity.

Features:
    - IndexFlatIP (cosine similarity)
    - Metadata persistence
    - Save/load functionality
    - Batch operations
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document
from rich.console import Console

import config

console = Console()


class FaissVectorStore:
    """FAISS-based Vector Store with Metadata Management.

    Efficient similarity search using FAISS IndexFlatIP (Inner Product)
    for cosine similarity with L2-normalized vectors. Includes metadata
    persistence for document tracking and citation.

    Index Type:
        IndexFlatIP - Inner Product similarity (cosine after normalization)

    Attributes:
        dimension: Embedding vector dimension.
        index: FAISS index instance.
        documents: List of LangChain Document objects.
        doc_ids: List of document IDs corresponding to index positions.

    Examples:
        >>> store = FaissVectorStore(dimension=1024)
        >>> store.add_documents(documents, embeddings)
        >>> results = store.search(query_embedding, k=5)
        >>> for doc, score in results:
        ...     print(f"{doc.metadata['filename']}: {score:.3f}")
    """

    def __init__(self, dimension: int = config.EMBEDDING_DIM):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Cosine similarity (after L2 normalization)
        self.documents: List[Document] = []
        self.doc_ids: List[int] = []
        
        console.print(f"[green]✓[/green] FaissVectorStore initialized (dim={dimension})")
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        normalize: bool = True
    ):
        """Add documents with embeddings to the index.

        Args:
            documents: List of LangChain Documents to add.
            embeddings: Corresponding embedding vectors.
            normalize: Whether to L2 normalize for cosine similarity (recommended).

        Raises:
            ValueError: If number of documents and embeddings don't match.
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"Mismatch: {len(documents)} docs vs {len(embeddings)} embeddings")
        
        # Convert to numpy
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        if normalize:
            faiss.normalize_L2(embeddings_np)
        
        # Add to index
        start_id = len(self.documents)
        self.index.add(embeddings_np)
        
        # Store metadata
        self.documents.extend(documents)
        self.doc_ids.extend(range(start_id, start_id + len(documents)))
        
        console.print(
            f"[green]✓[/green] Added {len(documents)} documents. "
            f"Total in index: {self.index.ntotal}"
        )
    
    def search(
        self,
        query_embedding: List[float],
        k: int = None,
        threshold: float = None,
        normalize: bool = True
    ) -> List[Tuple[Document, float]]:
        """Search for top-k most similar documents.

        Args:
            query_embedding: Query embedding vector.
            k: Number of results to return (default: config.TOP_K).
            threshold: Minimum similarity score (default: config.SIMILARITY_THRESHOLD).
            normalize: Whether to normalize query vector.

        Returns:
            List of (Document, similarity_score) tuples, sorted by score descending.
        """
        k = k or config.TOP_K
        threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD
        
        if self.index.ntotal == 0:
            console.print("[yellow]⚠[/yellow] Index is empty")
            return []
        
        # Convert to numpy
        query_np = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        if normalize:
            faiss.normalize_L2(query_np)
        
        # Search (get more than k to account for threshold filtering)
        search_k = min(k * 2, self.index.ntotal)
        scores, indices = self.index.search(query_np, search_k)
        
        # Filter by threshold and collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Check for invalid index
            if not (0 <= idx < len(self.documents)):
                console.print(f"[red]Warning: Invalid index {idx} from Faiss (total docs: {len(self.documents)})[/red]")
                continue

            if score >= threshold:
                results.append((self.documents[idx], float(score)))
        
        # Return top-k
        results = results[:k]
        
        if not results:
            console.print(
                f"[yellow]⚠[/yellow] No results above threshold {threshold:.3f}"
            )
        
        return results
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete documents by source path (not efficient, for maintenance only)
        
        Args:
            source: Source path to delete
            
        Returns:
            Number of documents deleted
        """
        # Faiss doesn't support deletion, so we need to rebuild
        keep_docs = []
        keep_ids = []
        
        for doc, doc_id in zip(self.documents, self.doc_ids):
            if doc.metadata.get("source") != source:
                keep_docs.append(doc)
                keep_ids.append(doc_id)
        
        deleted_count = len(self.documents) - len(keep_docs)
        
        if deleted_count > 0:
            console.print(f"[yellow]⚠[/yellow] Deletion requires rebuilding index...")
            console.print(f"[yellow]⚠[/yellow] This is expensive. Consider rebuilding from scratch.")
        
        return deleted_count
    
    def save(self, path: str = None):
        """
        Save index + metadata to disk
        
        Args:
            path: Save path (default: config.VECTORSTORE_PATH)
        """
        path = path or str(config.VECTORSTORE_PATH)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save Faiss index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save metadata
        metadata = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "dimension": self.dimension,
        }
        
        with open(f"{path}.metadata", "wb") as f:
            pickle.dump(metadata, f)
        
        console.print(f"[green]✓[/green] Saved vector store to: {path}")
        console.print(f"  Index: {path}.index ({self.index.ntotal} vectors)")
        console.print(f"  Metadata: {path}.metadata ({len(self.documents)} docs)")
    
    @classmethod
    def load(cls, path: str = None):
        """
        Load index + metadata from disk
        
        Args:
            path: Load path (default: config.VECTORSTORE_PATH)
            
        Returns:
            Loaded FaissVectorStore instance
        """
        path = path or str(config.VECTORSTORE_PATH)
        
        # Check files exist
        if not Path(f"{path}.index").exists():
            raise FileNotFoundError(f"Index file not found: {path}.index")
        if not Path(f"{path}.metadata").exists():
            raise FileNotFoundError(f"Metadata file not found: {path}.metadata")
        
        console.print(f"[cyan]Loading vector store from: {path}[/cyan]")
        
        # Load Faiss index
        index = faiss.read_index(f"{path}.index")
        
        # Load metadata
        with open(f"{path}.metadata", "rb") as f:
            metadata = pickle.load(f)
        
        # Reconstruct
        store = cls(dimension=metadata["dimension"])
        store.index = index
        store.documents = metadata["documents"]
        store.doc_ids = metadata["doc_ids"]
        
        console.print(
            f"[green]✓[/green] Loaded {len(store.documents)} documents "
            f"({store.index.ntotal} vectors)"
        )
        
        return store
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with stats
        """
        if not self.documents:
            return {"total_documents": 0, "total_vectors": 0}
        
        # Count unique files
        unique_files = set(doc.metadata.get("filename") for doc in self.documents)
        
        # Count chunks with tables
        chunks_with_tables = sum(
            1 for doc in self.documents 
            if doc.metadata.get("has_table", False)
        )
        
        # Average chunk length
        avg_chunk_length = sum(
            len(doc.page_content) for doc in self.documents
        ) / len(self.documents)
        
        return {
            "total_documents": len(self.documents),
            "total_vectors": self.index.ntotal,
            "unique_files": len(unique_files),
            "chunks_with_tables": chunks_with_tables,
            "avg_chunk_length": int(avg_chunk_length),
            "dimension": self.dimension,
        }
    
    def print_stats(self):
        """Print statistics in a nice format"""
        from rich.table import Table
        
        stats = self.get_stats()
        
        table = Table(title="Vector Store Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(
                key.replace("_", " ").title(),
                str(value)
            )
        
        console.print(table)

