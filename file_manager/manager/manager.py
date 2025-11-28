"""
PatXtract File Manager

Main orchestration layer for processing PDF documents through the extraction pipeline.
Coordinates between extraction routing, OCR/scanned text extraction, and provides
comprehensive metadata handling.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..extractor.extraction_routing import ExtractionRouter

logger = logging.getLogger(__name__)


class FileManager:
    """
    Main file manager for PatXtract system.

    Orchestrates the complete document processing pipeline:
    1. Document intake and validation
    2. Intelligent extraction method routing
    3. Text extraction (OCR or scanned)
    4. Metadata enrichment
    5. Result formatting and output
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize file manager.

        Args:
            config_path: Optional path to configuration file
        """
        self.extraction_router = ExtractionRouter(config_path)
        self.processing_stats = {
            "total_documents": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_processing_time": 0.0
        }

        logger.info("File Manager initialized")

    def process_pdf(self, pdf_path: Path,
                   method: Optional[str] = None,
                   include_analysis: bool = False) -> Dict[str, Any]:
        """
        Process a single PDF file through the complete pipeline.

        Args:
            pdf_path: Path to PDF file
            method: Optional extraction method ('scanned', 'ocr', or None for auto-detect)
            include_analysis: Whether to include document analysis in results

        Returns:
            Comprehensive result dictionary with extracted text and metadata
        """
        if not pdf_path.exists():
            return self._create_error_result(str(pdf_path), "File not found", None)

        start_time = time.time()
        logger.info(f"Starting processing for: {pdf_path}")

        # Initialize result structure
        result = {
            "success": False,
            "file_path": str(pdf_path),
            "file_name": pdf_path.name,
            "file_size": pdf_path.stat().st_size,
            "processing_timestamp": datetime.now().isoformat(),
            "processing_method": method,
            "include_analysis": include_analysis,
            "analysis": None,
            "extraction": None,
            "metadata": {},
            "processing_time": 0.0,
            "errors": []
        }

        try:
            # Step 1: Document Analysis (if requested)
            if include_analysis:
                logger.info("Performing document analysis...")
                result["analysis"] = self.extraction_router.analyze_document(pdf_path)

                # Use analysis to determine method if not specified
                if method is None and result["analysis"].get("recommended_method"):
                    method = result["analysis"]["recommended_method"]
                    logger.info(f"Analysis recommends extraction method: {method}")

            # Step 2: Text Extraction
            logger.info(f"Extracting text using method: {method}")
            extraction_result = self.extraction_router.extract_text(pdf_path, method)
            result["extraction"] = extraction_result

            # Step 3: Metadata Enrichment
            result["metadata"] = self._enrich_metadata(pdf_path, extraction_result, result["analysis"])

            # Step 4: Success determination
            result["success"] = extraction_result.get("success", False)

            if result["success"]:
                logger.info(f"Successfully processed: {pdf_path}")
            else:
                logger.warning(f"Processing completed with issues: {pdf_path}")
                if "errors" not in result:
                    result["errors"] = []
                result["errors"].extend(extraction_result.get("errors", []))

        except Exception as e:
            error_msg = f"Unexpected error during processing: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)

        # Calculate processing time
        result["processing_time"] = time.time() - start_time

        # Update statistics
        self._update_stats(result)

        return result

    def process_multiple_pdfs(self, pdf_paths: List[Path],
                             method: Optional[str] = None,
                             include_analysis: bool = False) -> Dict[str, Any]:
        """
        Process multiple PDF files.

        Args:
            pdf_paths: List of paths to PDF files
            method: Optional extraction method for all files
            include_analysis: Whether to include document analysis

        Returns:
            Batch processing results
        """
        batch_start_time = time.time()
        logger.info(f"Starting batch processing of {len(pdf_paths)} files")

        batch_result = {
            "batch_id": f"batch_{int(batch_start_time)}",
            "total_files": len(pdf_paths),
            "successful_files": 0,
            "failed_files": 0,
            "total_processing_time": 0.0,
            "results": [],
            "summary": {}
        }

        for pdf_path in pdf_paths:
            result = self.process_pdf(pdf_path, method, include_analysis)
            batch_result["results"].append(result)

            if result["success"]:
                batch_result["successful_files"] += 1
            else:
                batch_result["failed_files"] += 1

        # Calculate batch statistics
        batch_result["total_processing_time"] = time.time() - batch_start_time
        batch_result["summary"] = self._generate_batch_summary(batch_result["results"])

        logger.info(f"Batch processing completed: {batch_result['successful_files']}/{batch_result['total_files']} successful")

        return batch_result

    def _enrich_metadata(self, pdf_path: Path, extraction_result: Dict[str, Any],
                        analysis_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrich result with additional metadata."""
        metadata = {
            "file_info": {
                "path": str(pdf_path),
                "name": pdf_path.name,
                "extension": pdf_path.suffix.lower(),
                "size_bytes": pdf_path.stat().st_size,
                "size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
                "modified_time": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
            },
            "extraction_info": {
                "method_used": extraction_result.get("extraction_method", "unknown"),
                "processing_time": extraction_result.get("processing_time", 0.0),
                "total_pages": extraction_result.get("total_pages", 0),
                "total_characters": extraction_result.get("total_characters", 0)
            }
        }

        # Add analysis metadata if available
        if analysis_result:
            metadata["analysis_info"] = {
                "recommended_method": analysis_result.get("recommended_method"),
                "confidence": analysis_result.get("confidence", 0.0),
                "reason": analysis_result.get("reason", ""),
                "has_selectable_text": analysis_result.get("has_selectable_text", False)
            }

        # Add performance metrics
        if extraction_result.get("total_characters", 0) > 0 and extraction_result.get("processing_time", 0) > 0:
            chars_per_second = extraction_result["total_characters"] / extraction_result["processing_time"]
            metadata["performance"] = {
                "characters_per_second": round(chars_per_second, 2),
                "processing_efficiency": "high" if chars_per_second > 1000 else "medium" if chars_per_second > 500 else "low"
            }

        return metadata

    def _update_stats(self, result: Dict[str, Any]):
        """Update processing statistics."""
        self.processing_stats["total_documents"] += 1
        self.processing_stats["total_processing_time"] += result["processing_time"]

        if result["success"]:
            self.processing_stats["successful_extractions"] += 1
        else:
            self.processing_stats["failed_extractions"] += 1

    def _generate_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for batch processing."""
        total_chars = sum(r.get("extraction", {}).get("total_characters", 0) for r in results if r.get("success"))
        total_pages = sum(r.get("extraction", {}).get("total_pages", 0) for r in results if r.get("success"))
        avg_processing_time = sum(r.get("processing_time", 0) for r in results) / len(results) if results else 0

        method_counts = {}
        for result in results:
            method = result.get("extraction", {}).get("extraction_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        return {
            "total_characters_extracted": total_chars,
            "total_pages_processed": total_pages,
            "average_processing_time_seconds": round(avg_processing_time, 2),
            "extraction_methods_used": method_counts,
            "success_rate": round(len([r for r in results if r.get("success")]) / len(results) * 100, 2) if results else 0
        }

    def _create_error_result(self, file_path: str, error_message: str,
                           processing_time: Optional[float]) -> Dict[str, Any]:
        """Create a standardized error result."""
        return {
            "success": False,
            "file_path": file_path,
            "processing_timestamp": datetime.now().isoformat(),
            "processing_time": processing_time or 0.0,
            "errors": [error_message],
            "analysis": None,
            "extraction": None,
            "metadata": {}
        }

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()

    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            "total_documents": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_processing_time": 0.0
        }

    def get_supported_methods(self) -> Dict[str, bool]:
        """Get information about supported extraction methods."""
        return self.extraction_router.get_supported_methods()

    def export_results(self, results: Dict[str, Any], output_path: Path) -> bool:
        """
        Export processing results to JSON file.

        Args:
            results: Results dictionary to export
            output_path: Path for output JSON file

        Returns:
            True if export successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Results exported to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting results to {output_path}: {e}")
            return False

    def get_text_summary(self, result: Dict[str, Any]) -> str:
        """
        Generate a text summary of processing results.

        Args:
            result: Processing result dictionary

        Returns:
            Formatted text summary
        """
        if not result.get("success"):
            return f"❌ Processing failed for {result.get('file_name', 'unknown file')}\nErrors: {', '.join(result.get('errors', []))}"

        extraction = result.get("extraction", {})
        metadata = result.get("metadata", {})

        summary_lines = [
            f"✅ Successfully processed: {result.get('file_name', 'unknown file')}",
            f"📄 Pages: {extraction.get('total_pages', 0)}",
            f"📝 Characters: {extraction.get('total_characters', 0):,}",
            f"⏱️  Processing time: {result.get('processing_time', 0):.2f} seconds",
            f"🔧 Method: {extraction.get('extraction_method', 'unknown')}",
            f"📊 File size: {metadata.get('file_info', {}).get('size_mb', 0):.2f} MB"
        ]

        # Add performance info if available
        performance = metadata.get("performance", {})
        if performance:
            summary_lines.append(f"⚡ Speed: {performance.get('characters_per_second', 0):,.0f} chars/sec")

        return "\n".join(summary_lines)