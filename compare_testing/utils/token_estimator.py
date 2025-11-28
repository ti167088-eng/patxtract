"""
Token Estimation Utility

Provides token counting and estimation functionality for OCR text data
to determine when document chunking is needed for AI models.
"""

import re
from typing import List, Dict, Any, Tuple


class TokenEstimator:
    """Estimates token count for OCR text data."""

    def __init__(self):
        """Initialize the token estimator."""
        # Average tokens per character for different content types
        self.char_to_token_ratios = {
            'standard_text': 0.25,  # ~4 characters per token
            'medical_text': 0.30,   # More medical terminology = slightly more tokens
            'ocr_text': 0.28,       # OCR text has formatting/whitespace
            'mixed_content': 0.27   # Average for mixed content
        }

    def estimate_tokens_from_text(self, text: str, content_type: str = 'ocr_text') -> int:
        """
        Estimate token count from raw text.

        Args:
            text: Raw text content
            content_type: Type of content for ratio selection

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Clean the text
        cleaned_text = self._clean_text(text)
        char_count = len(cleaned_text)

        # Get appropriate ratio
        ratio = self.char_to_token_ratios.get(content_type, self.char_to_token_ratios['ocr_text'])

        # Estimate tokens
        estimated_tokens = int(char_count * ratio)

        return estimated_tokens

    def estimate_tokens_from_pages(self, pages: List[Dict[str, Any]], content_type: str = 'ocr_text') -> int:
        """
        Estimate total token count from page data.

        Args:
            pages: List of page dictionaries with 'text' field
            content_type: Type of content for ratio selection

        Returns:
            Total estimated token count
        """
        total_tokens = 0

        for page in pages:
            page_text = page.get('text', '')
            page_tokens = self.estimate_tokens_from_text(page_text, content_type)
            total_tokens += page_tokens

        return total_tokens

    def estimate_tokens_from_extracted_data(self, extracted_data: Dict[str, Any]) -> int:
        """
        Estimate token count from extracted data structure.

        Args:
            extracted_data: Extracted data dictionary with pages

        Returns:
            Total estimated token count
        """
        pages = extracted_data.get('pages', [])
        return self.estimate_tokens_from_pages(pages)

    def calculate_optimal_chunk_size(self,
                                   total_tokens: int,
                                   model_context_limit: int,
                                   safety_buffer: float = 0.2) -> int:
        """
        Calculate optimal chunk size in tokens.

        Args:
            total_tokens: Total token count in document
            model_context_limit: Model's maximum context window
            safety_buffer: Safety buffer ratio (default 20%)

        Returns:
            Optimal chunk size in tokens
        """
        usable_tokens = int(model_context_limit * (1 - safety_buffer))

        if total_tokens <= usable_tokens:
            return total_tokens  # No chunking needed

        return usable_tokens

    def calculate_pages_per_chunk(self,
                                 pages: List[Dict[str, Any]],
                                 max_tokens_per_chunk: int,
                                 content_type: str = 'ocr_text') -> int:
        """
        Calculate how many pages should be in each chunk.

        Args:
            pages: List of page dictionaries
            max_tokens_per_chunk: Maximum tokens per chunk
            content_type: Type of content for ratio selection

        Returns:
            Number of pages per chunk
        """
        if not pages:
            return 0

        # Calculate average tokens per page
        total_tokens = self.estimate_tokens_from_pages(pages, content_type)
        avg_tokens_per_page = total_tokens / len(pages)

        # Calculate pages per chunk
        pages_per_chunk = max(1, int(max_tokens_per_chunk / avg_tokens_per_page))

        return pages_per_chunk

    def needs_chunking(self,
                      pages: List[Dict[str, Any]],
                      model_context_limit: int,
                      content_type: str = 'ocr_text') -> Tuple[bool, int]:
        """
        Determine if document needs chunking.

        Args:
            pages: List of page dictionaries
            model_context_limit: Model's maximum context window
            content_type: Type of content for ratio selection

        Returns:
            Tuple of (needs_chunking, total_tokens)
        """
        total_tokens = self.estimate_tokens_from_pages(pages, content_type)
        usable_limit = int(model_context_limit * 0.8)  # 80% of limit for safety

        needs_chunking = total_tokens > usable_limit

        return needs_chunking, total_tokens

    def create_chunks(self,
                     pages: List[Dict[str, Any]],
                     pages_per_chunk: int,
                     overlap_ratio: float = 0.1) -> List[List[Dict[str, Any]]]:
        """
        Create overlapping chunks from pages.

        Args:
            pages: List of page dictionaries
            pages_per_chunk: Number of pages per chunk
            overlap_ratio: Overlap ratio between chunks (0.1 = 10%)

        Returns:
            List of page chunks
        """
        if not pages:
            return []

        if len(pages) <= pages_per_chunk:
            return [pages]

        chunks = []
        overlap_pages = max(1, int(pages_per_chunk * overlap_ratio))

        start_idx = 0

        # CRITICAL FIX: Add safety limits to prevent infinite loops
        max_iterations = len(pages) * 2  # Safety limit
        iteration_count = 0
        seen_chunks = set()  # Prevent duplicate chunks

        while start_idx < len(pages) and iteration_count < max_iterations:
            end_idx = min(start_idx + pages_per_chunk, len(pages))

            # CRITICAL FIX: Prevent infinite loop - ensure we always move forward
            if end_idx <= start_idx:
                break

            chunk_key = (start_idx, end_idx)
            if chunk_key in seen_chunks:
                break

            chunk = pages[start_idx:end_idx]
            chunks.append(chunk)
            seen_chunks.add(chunk_key)

            # Calculate next start with guaranteed progress
            next_start = end_idx - overlap_pages
            if next_start >= end_idx or next_start >= len(pages):
                break

            start_idx = next_start
            iteration_count += 1

        # CRITICAL FIX: Safety fallback in case of infinite loop
        if iteration_count >= max_iterations:
            # Simple non-overlapping chunks as fallback
            chunks = []
            for i in range(0, len(pages), pages_per_chunk):
                chunks.append(pages[i:i + pages_per_chunk])

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Clean text for more accurate token estimation.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ''

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common OCR artifacts that don't contribute to meaning
        text = re.sub(r'[^\w\s\-\.,;:!@#$%^&*()\[\]{}"\'/?<>|\\`~]', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def get_model_context_limits(self) -> Dict[str, int]:
        """
        Get context limits for different models.

        Returns:
            Dictionary of model names to their token limits
        """
        return {
            'qwen_2_5_7b_instruct_patient': 32000,     # Qwen_2_5_7B_Instruct_Patient
            'gpt-oss-20b_patient': 128000,             # GPT-OSS-20B_Patient (exact algorithm name)
            'qwen3_vl_8b_instruct_patient': 128000,    # Qwen3_VL_8B_Instruct_Patient (vision model)
            'gemma_3_4b_it_patient': 70000,  # Gemma 3 4B: Use 70,000 tokens (73% of 96K) with 15% buffer            # Gemma_3_4B_IT_Patient (96K context window)
            'mistral_7b_instruct_patient': 32000,       # Mistral_7B_Instruct_Patient (32K context window)
            'claude_3_5_sonnet': 200000,
            'gpt_4_turbo': 128000,
            'gpt_35_turbo': 16385,
            # Add more models as needed
        }

    def get_optimal_chunk_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get optimal chunking configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Configuration dictionary
        """
        limits = self.get_model_context_limits()
        model_limit = limits.get(model_name, 32000)  # Default to 32K

        # Model-specific configurations
        configs = {
            'qwen_2_5_7b_instruct_patient': {
                'max_tokens_per_chunk': int(model_limit * 0.75),  # 75% of limit
                'overlap_ratio': 0.15,  # 15% overlap
                'safety_buffer': 0.25   # 25% safety buffer
            },
            'gpt-oss-20b_patient': {
                'max_tokens_per_chunk': int(model_limit * 0.8),   # 80% of limit
                'overlap_ratio': 0.1,   # 10% overlap
                'safety_buffer': 0.2    # 20% safety buffer
            },
            'qwen3_vl_8b_instruct_patient': {
                'max_tokens_per_chunk': int(model_limit * 0.7),   # 70% of limit (vision needs more space)
                'overlap_ratio': 0.2,   # 20% overlap (vision needs more context)
                'safety_buffer': 0.3    # 30% safety buffer (vision processing)
            },
            'gemma_3_4b_it_patient': {
                'max_tokens_per_chunk': int(model_limit * 0.85),  # 85% of limit (larger context can handle more)
                'overlap_ratio': 0.05,  # 5% overlap (less overlap needed with larger context)
                'safety_buffer': 0.15   # 15% safety buffer (more efficient with larger context)
            },
            'mistral_7b_instruct_patient': {
                'max_tokens_per_chunk': int(model_limit * 0.75),  # 75% of limit (conservative with 32K context)
                'overlap_ratio': 0.15,  # 15% overlap (good for patient identification)
                'safety_buffer': 0.25   # 25% safety buffer (conservative approach)
            }
        }

        return configs.get(model_name, {
            'max_tokens_per_chunk': int(model_limit * 0.8),
            'overlap_ratio': 0.1,
            'safety_buffer': 0.2
        })


def estimate_document_size(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to get document size analysis.

    Args:
        pages: List of page dictionaries

    Returns:
        Analysis dictionary with token estimates
    """
    estimator = TokenEstimator()

    total_chars = sum(len(page.get('text', '')) for page in pages)
    total_tokens = estimator.estimate_tokens_from_pages(pages)
    avg_tokens_per_page = total_tokens / len(pages) if pages else 0

    return {
        'total_pages': len(pages),
        'total_characters': total_chars,
        'estimated_tokens': total_tokens,
        'average_tokens_per_page': avg_tokens_per_page,
        'model_limits': estimator.get_model_context_limits()
    }