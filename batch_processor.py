# -*- coding: utf-8 -*-
"""Batch Query Processor Module.

Process multiple questions through RAG pipeline with database persistence.

Features:
    - Single question processing
    - Batch processing with progress tracking
    - Full RAG pipeline integration (Retriever + Generator)
    - Database persistence
    - Error handling
"""

import time
from typing import Any, Dict, List, Optional

from rich.console import Console

from src import CitationFormatter, Generator, Retriever

import config
from batch_query_db import BatchQueryDB

console = Console()


class BatchQueryProcessor:
    """Process batch queries with full RAG pipeline.

    This class handles batch processing of questions through the complete
    RAG pipeline, including retrieval and generation with database persistence.

    Attributes:
        retriever: Retriever instance for document retrieval.
        generator: Generator instance for answer generation.
        db: BatchQueryDB instance for result persistence.
    """

    def __init__(self, retriever=None, generator=None, db=None):
        """Initialize processor with RAG components.

        Args:
            retriever: Retriever instance (optional, will create if None).
            generator: Generator instance (optional, will create if None).
            db: BatchQueryDB instance (optional, will create if None).
        """
        console.print("[cyan]Initializing Batch Query Processor...[/cyan]")
        
        # Initialize components
        try:
            self.retriever = retriever or Retriever()
            self.generator = generator or Generator()
            self.db = db or BatchQueryDB()
            
            console.print("[green]✓[/green] Batch processor ready")
        
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize: {e}")
            raise
    
    def process_single_question(
        self,
        question: str,
        mode: str = "patient",
        patient_info: Optional[str] = None,
        threshold: float = None,
        top_k: int = None,
        validate_quality: bool = False
    ) -> Dict[str, Any]:
        """Process a single question through RAG pipeline.

        Args:
            question: User question to process.
            mode: Generation mode ('patient', 'patient_personalized', 'academic').
            patient_info: Patient clinical information (for personalized mode).
            threshold: Similarity threshold (default: config.SIMILARITY_THRESHOLD).
            top_k: Number of contexts to retrieve (default: config.TOP_K).
            validate_quality: Whether to run quality validation.

        Returns:
            Dictionary containing answer, citations, metadata, and timing information.
        """
        start_time = time.time()
        
        threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD
        top_k = top_k or config.TOP_K
        
        try:
            # 1. Retrieve contexts
            retrieval_results = self.retriever.retrieve(
                query=question,
                k=top_k,
                threshold=threshold
            )
            
            # 2. Format contexts for LLM
            formatted_contexts = self.retriever.format_contexts_for_llm(retrieval_results)
            
            # 3. Generate answer
            result = self.generator.generate_with_citations(
                question=question,
                retrieval_results=retrieval_results,
                formatted_contexts=formatted_contexts,
                mode=mode,
                patient_info=patient_info,
                validate_quality=validate_quality
            )
            
            # 4. Add metadata
            query_time = time.time() - start_time
            
            result.update({
                "question": question,
                "mode": mode,
                "patient_info": patient_info or "",
                "threshold": threshold,
                "top_k": top_k,
                "retrieval_results": retrieval_results,
                "query_time": query_time,
            })
            
            return result
        
        except Exception as e:
            console.print(f"[red]✗[/red] Error processing question: {e}")
            
            # Return error result
            return {
                "question": question,
                "answer": f"❌ Error: {str(e)}",
                "mode": mode,
                "patient_info": patient_info or "",
                "threshold": threshold,
                "top_k": top_k,
                "retrieval_results": [],
                "citations": [],
                "reasoning": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "query_time": time.time() - start_time,
                "error": str(e)
            }
    
    def process_questions(
        self,
        questions: List[str],
        mode: str = "patient",
        patient_info: Optional[str] = None,
        patient_infos: Optional[List[str]] = None,
        threshold: float = None,
        top_k: int = None,
        progress_callback=None,
        save_to_db: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple questions through RAG pipeline.

        Args:
            questions: List of questions to process.
            mode: Generation mode ('patient', 'patient_personalized', 'academic').
            patient_info: Default patient info (used if patient_infos not provided).
            patient_infos: List of patient info per question (optional).
            threshold: Similarity threshold for retrieval.
            top_k: Number of contexts to retrieve.
            progress_callback: Callback function(current, total, message) for progress updates.
            save_to_db: Whether to save results to database.

        Returns:
            List of result dictionaries, one per question.
        """
        results = []
        total = len(questions)
        
        console.print(f"\n[bold cyan]Processing {total} questions...[/bold cyan]")
        
        for i, question in enumerate(questions, 1):
            # Progress callback
            if progress_callback:
                progress_callback(i, total, f"Processing: {question[:50]}...")
            
            # Determine patient info for this question
            current_patient_info = None
            if patient_infos and (i - 1) < len(patient_infos):
                current_patient_info = patient_infos[i - 1]
            else:
                current_patient_info = patient_info
            
            # Process question
            try:
                result = self.process_single_question(
                    question=question,
                    mode=mode,
                    patient_info=current_patient_info,
                    threshold=threshold,
                    top_k=top_k
                )
                
                results.append(result)
                
                # Save to database
                if save_to_db and self.db:
                    try:
                        self.db.insert_query(result)
                    except Exception as e:
                        console.print(f"[yellow]⚠️[/yellow] DB save failed: {e}")
                
                console.print(f"[green]✓[/green] {i}/{total}: {question[:50]}...")
            
            except Exception as e:
                console.print(f"[red]✗[/red] {i}/{total} failed: {e}")
                results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "mode": mode,
                    "error": str(e)
                })
        
        console.print(f"\n[green]✓[/green] Batch processing complete: {len(results)}/{total} succeeded")
        
        return results
    
    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from batch processing results.

        Args:
            results: List of result dictionaries from batch processing.

        Returns:
            Dictionary containing statistics including total/successful/failed counts,
            average similarity, average tokens used, and timing information.
        """
        if not results:
            return {}
        
        # Filter successful results
        successful = [r for r in results if "error" not in r]
        
        if not successful:
            return {
                "total": len(results),
                "successful": 0,
                "failed": len(results)
            }
        
        # Calculate stats
        results_with_citations = [r for r in successful if r.get('citations')]
        if results_with_citations:
            avg_similarity = sum(r['citations'][0]['similarity'] for r in results_with_citations) / len(results_with_citations)
        else:
            avg_similarity = 0.0

        avg_tokens = sum(
            r.get('total_tokens', 0)
            for r in successful
        ) / len(successful)

        avg_time = sum(
            r.get('query_time', 0)
            for r in successful
        ) / len(successful)

        return {
            "total": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "avg_similarity": round(avg_similarity, 3),
            "avg_tokens": round(avg_tokens, 0),
            "avg_time_seconds": round(avg_time, 2),
            "total_time_seconds": round(sum(r.get('query_time', 0) for r in successful), 2)
        }
