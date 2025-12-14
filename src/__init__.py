# -*- coding: utf-8 -*-
"""RAG System - Source Package.

This package contains core modules for the RAG (Retrieval-Augmented Generation)
system designed for Hereditary Angioedema (HAE) medical question answering.

Modules:
    embedding_service: BGE-M3 embedding generation service.
    vector_store: FAISS-based vector storage and retrieval.
    retriever: Intelligent document retrieval with cross-lingual support.
    generator: DeepSeek R1 answer generation with multi-mode support.
    citation_formatter: Citation formatting for different presentation styles.

Examples:
    >>> from src import Retriever, Generator, CitationFormatter
    >>> retriever = Retriever()
    >>> generator = Generator()
    >>> formatter = CitationFormatter()
"""

from .embedding_service import EmbeddingService
from .vector_store import FaissVectorStore
from .retriever import Retriever
from .generator import Generator
from .citation_formatter import CitationFormatter

__all__ = [
    'EmbeddingService',
    'FaissVectorStore',
    'Retriever',
    'Generator',
    'CitationFormatter'
]

__version__ = '1.0.0'
__author__ = 'HAE RAG System Team'
