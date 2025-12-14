# -*- coding: utf-8 -*-
"""Embedding Service Module.

This module provides BGE-M3 multilingual embedding generation with automatic
fallback between local sentence-transformers and HuggingFace API.

Features:
    - Local sentence-transformers support for fast embedding
    - HuggingFace API fallback when local unavailable
    - Batch processing with progress tracking
    - Cross-lingual similarity testing
"""

import time
from typing import List

import numpy as np
from rich.console import Console

console = Console()

# Try sentence-transformers (local, fast)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Fallback: HuggingFace API
try:
    from huggingface_hub import InferenceClient
    HF_INFERENCE_AVAILABLE = True
except ImportError:
    HF_INFERENCE_AVAILABLE = False

import config


class EmbeddingService:
    """BGE-M3 Embedding Service with Local and API Support.

    Provides multilingual embedding generation using BGE-M3 model with automatic
    fallback between local sentence-transformers and HuggingFace API.

    The service automatically selects the best available method:
        1. Local (sentence-transformers) - Fast, no API calls required
        2. HuggingFace Inference API - Fallback when local not available

    Attributes:
        use_local: Whether local model is being used.
        model: Local SentenceTransformer model instance (if local).
        client: HuggingFace InferenceClient instance (if API).

    Examples:
        >>> service = EmbeddingService(use_local=True)
        >>> embedding = service.embed_query("HAE nedir?")
        >>> print(f"Embedding dimension: {len(embedding)}")
        1024
    """

    def __init__(self, use_local: bool = True):
        """Initialize the embedding service.

        Args:
            use_local: Whether to try local model first (recommended).
        """
        self.use_local = use_local
        self.model = None
        self.client = None
        
        # Try local first
        if use_local and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._init_local()
        else:
            self._init_api()
    
    def _init_local(self):
        """Initialize local sentence-transformers model.

        Downloads and initializes the BGE-M3 model locally using sentence-transformers.
        Falls back to API mode if initialization fails.
        """
        try:
            console.print("[cyan]Initializing BGE-M3 (local)...[/cyan]")
            console.print("[yellow]First time: Downloading model (~2GB)...[/yellow]")
            
            self.model = SentenceTransformer('BAAI/bge-m3')
            
            # Test
            test_emb = self.model.encode("test", convert_to_numpy=True)
            assert len(test_emb) == config.EMBEDDING_DIM
            
            console.print(f"[green]✓[/green] BGE-M3 ready (local mode)")
            console.print(f"  Dimension: {config.EMBEDDING_DIM}")
            console.print(f"  Mode: Local (no API calls)")
            
            self.use_local = True
        
        except Exception as e:
            console.print(f"[red]✗[/red] Local init failed: {e}")
            console.print("[yellow]Falling back to API mode...[/yellow]")
            self._init_api()
    
    def _init_api(self):
        """Initialize HuggingFace Inference API.

        Sets up the HuggingFace Inference Client using the API key from config.

        Raises:
            ImportError: If huggingface_hub is not installed.
            ValueError: If HUGGINGFACE_API_KEY is not configured.
        """
        if not HF_INFERENCE_AVAILABLE:
            raise ImportError("huggingface_hub not installed")
        
        hf_token = config.HUGGINGFACE_API_KEY
        if not hf_token:
            raise ValueError("HUGGINGFACE_API_KEY not found")
        
        try:
            console.print("[cyan]Initializing BGE-M3 (API)...[/cyan]")
            
            self.client = InferenceClient(token=hf_token)
            
            # Test
            test_result = self.client.feature_extraction(
                "test",
                model=config.EMBEDDING_MODEL
            )
            
            if isinstance(test_result, list):
                test_emb = test_result
            else:
                test_emb = test_result.tolist()
            
            assert len(test_emb) == config.EMBEDDING_DIM
            
            console.print(f"[green]✓[/green] BGE-M3 ready (API mode)")
            console.print(f"  Dimension: {config.EMBEDDING_DIM}")
            console.print("[yellow]⚠ API is slow. Install: pip install sentence-transformers[/yellow]")
            
            self.use_local = False
            self.model = None
        
        except Exception as e:
            console.print(f"[red]✗[/red] API init failed: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            List of floats representing the embedding vector (dimension: 1024).
        """
        if self.use_local and self.model:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        
        # API mode
        result = self.client.feature_extraction(text, model=config.EMBEDDING_MODEL)
        
        if isinstance(result, list):
            return result
        return result.tolist()
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = True
    ) -> List[List[float]]:
        """Embed multiple documents in batch.

        Args:
            texts: List of document texts to embed.
            batch_size: Batch size for processing (default: from config).
            show_progress: Whether to display progress bar.

        Returns:
            List of embedding vectors, one per document.
        """
        batch_size = batch_size or config.BATCH_SIZE
        
        # Local mode (fast)
        if self.use_local and self.model:
            console.print(f"[cyan]Embedding {len(texts)} documents (local)...[/cyan]")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            console.print(f"[green]✓[/green] Embedded {len(embeddings)} documents")
            return [emb.tolist() for emb in embeddings]
        
        # API mode (slow)
        console.print(f"[yellow]API mode: Embedding {len(texts)} documents...[/yellow]")
        console.print("[yellow]This will take 1-2 hours (rate limited)[/yellow]")
        
        embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Embedding (API)")
        else:
            iterator = texts
        
        for text in iterator:
            try:
                emb = self.embed_query(text)
                embeddings.append(emb)
                time.sleep(0.1)  # Rate limit
            except Exception as e:
                console.print(f"[red]✗[/red] Error: {e}")
                embeddings.append([0.0] * config.EMBEDDING_DIM)
        
        console.print(f"[green]✓[/green] Embedded {len(embeddings)} documents")
        return embeddings
    
    def test_cross_lingual(self):
        """Test cross-lingual similarity between Turkish and English queries.

        Demonstrates the multilingual capabilities of BGE-M3 by computing
        similarity scores between semantically equivalent Turkish-English pairs.
        """
        test_pairs = [
            ("yapay zeka nedir", "what is artificial intelligence"),
            ("makine öğrenmesi", "machine learning"),
            ("derin öğrenme", "deep learning"),
        ]
        
        console.print("\n[cyan]Cross-Lingual Test:[/cyan]")
        console.print("=" * 70)
        
        for tr, en in test_pairs:
            tr_emb = self.embed_query(tr)
            en_emb = self.embed_query(en)
            
            similarity = np.dot(tr_emb, en_emb) / (
                np.linalg.norm(tr_emb) * np.linalg.norm(en_emb)
            )
            
            color = "green" if similarity > 0.75 else "yellow"
            console.print(f"[{color}]TR: {tr:30s} | EN: {en:35s} | Sim: {similarity:.3f}[/{color}]")
        
        console.print("=" * 70)


if __name__ == "__main__":
    service = EmbeddingService(use_local=True)
    service.test_cross_lingual()
