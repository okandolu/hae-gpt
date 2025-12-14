"""Generator Module for Multi-Mode Answer Generation.

This module provides the Generator class that uses DeepSeek R1 for generating
answers with reasoning support across three different modes: patient-friendly,
personalized patient, and academic.

Features:
    - DeepSeek R1 integration for reasoning-capable generation
    - Multi-mode support with mode-specific prompts and temperatures
    - Cross-lingual support (Turkish query → English context → Turkish answer)
    - Quality validation and citation integration
    - Token usage tracking

Modes:
    - patient: Hasta bilgilendirme (8. sınıf, empatik)
    - patient_personalized: Hasta özel (klinik bilgilerle)
    - academic: Akademisyen (teknik, detaylı)
"""

from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from rich.console import Console

import config

console = Console()


class Generator:
    """DeepSeek R1 Answer Generator with Multi-Mode Support.

    Provides answer generation capabilities with three distinct modes tailored
    for different audiences: general patients, personalized patient care, and
    academic/medical professionals.

    Capabilities:
        - Read English contexts
        - Detect the question's language and answer STRICTLY in that same language
        - Mode-specific prompts and temperature
        - Reasoning transparency
        - Faithfulness enforcement
        - Quality validation (optional)

    Modes:
        - patient: Simple, empathetic language for general patient education
        - patient_personalized: Customized advice based on clinical information
        - academic: Technical, detailed responses with academic citations

    Attributes:
        client: OpenAI client configured for DeepSeek R1 API.

    Examples:
        >>> generator = Generator()
        >>> result = generator.generate(
        ...     question="HAE nedir?",
        ...     contexts="Context about HAE...",
        ...     mode="patient"
        ... )
        >>> print(result['answer'])
    """

    def __init__(self) -> None:
        """Initialize Generator with DeepSeek R1 client.

        Establishes connection to DeepSeek R1 API and performs a test
        request to verify connectivity.

        Raises:
            Exception: If API connection fails or credentials are invalid.
        """
        console.print("[cyan]Initializing DeepSeek R1...[/cyan]")
        
        try:
            self.client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
                timeout=config.DEEPSEEK_TIMEOUT,
            )
            
            # Test connection
            test_response = self.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            
            console.print(f"[green]✓[/green] DeepSeek R1 ready: {config.DEEPSEEK_MODEL}")
        
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize DeepSeek R1: {e}")
            raise
    
    def _get_system_prompt(self, mode: str, patient_info: Optional[str] = None) -> str:
        """Get mode-specific system prompt.

        Retrieves the appropriate system prompt template based on the selected
        mode and optionally formats it with patient clinical information.

        Args:
            mode: Generation mode - 'patient', 'patient_personalized', or 'academic'.
            patient_info: Patient clinical information for personalized mode.
                Ignored for other modes.

        Returns:
            Formatted system prompt string ready for LLM.

        Examples:
            >>> generator = Generator()
            >>> prompt = generator._get_system_prompt("patient")
            >>> print(prompt[:100])
        """
        if mode == "patient":
            return config.SYSTEM_PROMPT_PATIENT
        
        elif mode == "patient_personalized":
            if patient_info and patient_info.strip():
                return config.SYSTEM_PROMPT_PATIENT_PERSONALIZED.format(
                    patient_info=patient_info
                )
            else:
                # Fallback to general patient mode if no info provided
                console.print("[yellow]⚠ [/yellow] No patient info provided, using general patient mode")
                return config.SYSTEM_PROMPT_PATIENT
        
        elif mode == "academic":
            return config.SYSTEM_PROMPT_ACADEMIC
        
        else:
            console.print(f"[yellow]⚠ [/yellow] Unknown mode '{mode}', defaulting to patient")
            return config.SYSTEM_PROMPT_PATIENT
    
    def _get_temperature(self, mode: str) -> float:
        """Get temperature setting for specified mode.

        Args:
            mode: Generation mode ('patient', 'patient_personalized', 'academic').

        Returns:
            Temperature value (0.0-1.0) for the specified mode.
        """
        return config.MODE_TEMPERATURES.get(mode, config.DEEPSEEK_TEMPERATURE)

    def generate(
        self,
        question: str,
        contexts: str,
        mode: str = "patient",
        patient_info: Optional[str] = None,
        include_reasoning: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate answer from question and retrieved contexts.

        Uses DeepSeek R1 to generate an answer in the same language as the
        question, based on provided English contexts. Supports multi-mode
        generation with different prompts and temperatures.

        Args:
            question: User question (auto-detects language for response).
            contexts: Retrieved contexts (English, pre-formatted).
            mode: Generation mode - 'patient', 'patient_personalized', or 'academic'.
            patient_info: Patient clinical information for personalized mode.
            include_reasoning: Whether to return R1's reasoning process.
            temperature: Override default mode temperature (0.0-1.0).
            max_tokens: Override default max tokens for generation.

        Returns:
            Dictionary containing:
                - answer (str): Generated answer text.
                - reasoning (str, optional): R1's reasoning process if requested.
                - prompt_tokens (int): Number of tokens in prompt.
                - completion_tokens (int): Number of tokens in completion.
                - total_tokens (int): Total tokens used.
                - mode (str): Mode used for generation.
                - temperature (float): Temperature used.

        Raises:
            Exception: If generation fails due to API errors.

        Examples:
            >>> generator = Generator()
            >>> result = generator.generate(
            ...     question="HAE nedir?",
            ...     contexts="Context about HAE...",
            ...     mode="patient"
            ... )
            >>> print(result['answer'])
            >>> print(f"Tokens used: {result['total_tokens']}")
        """
        # Get mode-specific settings
        system_prompt = self._get_system_prompt(mode, patient_info)
        
        if temperature is None:
            temperature = self._get_temperature(mode)
        
        max_tokens = max_tokens or config.DEEPSEEK_MAX_TOKENS
        
        # Build user prompt
        user_prompt = config.USER_PROMPT_TEMPLATE.format(
            question=question,
            contexts=contexts
        )

        # Combine language detection with system prompt into single system message
        combined_system_prompt = (
            "Detect the user's question language and reply STRICTLY in that language. "
            "If the question is multilingual, mirror the dominant language. Do NOT translate.\n\n"
            f"{system_prompt}"
        )

        messages = [
            {"role": "system", "content": combined_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        console.print(f"[cyan]Generating answer (mode: {mode}, temp: {temperature})...[/cyan]")
        
        try:
            response = self.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract response
            answer = response.choices[0].message.content
            reasoning = response.choices[0].message.reasoning_content if include_reasoning else None
            
            # Usage stats
            usage = response.usage
            
            result = {
                "answer": answer,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "mode": mode,
                "temperature": temperature,
            }
            
            if include_reasoning and reasoning:
                result["reasoning"] = reasoning
            
            console.print(f"[green]✓[/green] Generated answer ({usage.completion_tokens} tokens)")
            
            return result
        
        except Exception as e:
            console.print(f"[red]✗[/red] Generation failed: {e}")
            raise
    
    def generate_with_citations(
        self,
        question: str,
        retrieval_results: List[Tuple[Any, float]],
        formatted_contexts: str,
        mode: str = "patient",
        patient_info: Optional[str] = None,
        validate_quality: bool = False
    ) -> Dict[str, Any]:
        """Generate answer with automatic citation formatting.

        Combines answer generation with citation metadata extraction and
        optional quality validation. Returns a complete result package
        ready for display.

        Args:
            question: User question text.
            retrieval_results: Raw retrieval results as (Document, score) tuples.
            formatted_contexts: Pre-formatted context string for LLM.
            mode: Generation mode - 'patient', 'patient_personalized', or 'academic'.
            patient_info: Patient clinical information for personalized mode.
            validate_quality: Whether to run quality validation checks.

        Returns:
            Dictionary containing:
                - answer (str): Generated answer text.
                - citations (List[Dict]): Citation metadata for each source.
                - reasoning (str): R1's reasoning process.
                - prompt_tokens (int): Tokens in prompt.
                - completion_tokens (int): Tokens in completion.
                - total_tokens (int): Total tokens used.
                - mode (str): Mode used.
                - temperature (float): Temperature used.
                - quality_check (Dict, optional): Validation results if requested.

        Examples:
            >>> generator = Generator()
            >>> result = generator.generate_with_citations(
            ...     question="HAE tedavisi nedir?",
            ...     retrieval_results=[(doc1, 0.85), (doc2, 0.78)],
            ...     formatted_contexts="...",
            ...     mode="patient",
            ...     validate_quality=True
            ... )
            >>> for citation in result['citations']:
            ...     print(f"{citation['filename']}, p.{citation['page']}")
        """
        # Generate answer
        result = self.generate(
            question=question,
            contexts=formatted_contexts,
            mode=mode,
            patient_info=patient_info,
            include_reasoning=True
        )
        
        # Add citation metadata
        citations = []
        for i, (doc, score) in enumerate(retrieval_results, 1):
            citation = {
                "index": i,
                "filename": doc.metadata.get("filename", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "section": doc.metadata.get("section", "Unknown"),
                "similarity": score,
                "excerpt": doc.page_content[:200].strip(),
                "has_table": doc.metadata.get("has_table", False),
            }
            
            # NEW: Add metadata fields from enrichment
            if "reference" in doc.metadata:
                citation["reference"] = doc.metadata["reference"]
            if "publication_year" in doc.metadata:
                citation["publication_year"] = doc.metadata["publication_year"]
            if "citation_format" in doc.metadata:
                citation["citation_format"] = doc.metadata["citation_format"]
            
            citations.append(citation)
        
        result["citations"] = citations
        
        # Optional quality validation
        if validate_quality:
            quality_check = config.validate_answer_quality(
                answer=result["answer"],
                mode=mode,
                contexts=formatted_contexts
            )
            result["quality_check"] = quality_check
            
            # Log warnings/issues
            if not quality_check["passed"]:
                console.print(f"[yellow]⚠  Quality issues detected:[/yellow]")
                for issue in quality_check["critical_issues"]:
                    console.print(f"  [red]✗[/red] {issue}")
            if quality_check["warnings"]:
                for warning in quality_check["warnings"]:
                    console.print(f"  [yellow]⚠ [/yellow] {warning}")
        
        return result
    
