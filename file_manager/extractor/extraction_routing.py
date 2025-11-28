"""
PatXtract Extraction Routing

Intelligently routes PDF documents to appropriate extraction methods based on
document characteristics such as text presence, image quality, and complexity.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

from .ocr.paddleocr import PaddleOCRExtractor
from .ocr.tesseract import TesseractExtractor
from .ocr.pdf_ocr_processor import EnhancedPaddleOCR
from .ocr.hybrid_ocr_processor import HybridOCRProcessor
from .scanned.scan import ScannedTextExtractor

logger = logging.getLogger(__name__)


class ExtractionRouter:
    """
    Intelligent routing system for document extraction methods.

    Analyzes PDF documents to determine the optimal extraction method:
    - Scanned text extraction for PDFs with selectable text
    - OCR extraction for image-based documents
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize extraction router.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)

        # Initialize extractors
        self.scanned_extractor = ScannedTextExtractor()
        self.ocr_extractor = None

        # Initialize OCR extractors based on config
        if self.config.get('ocr', {}).get('enable_paddleocr', True):
            try:
                base_ocr_extractor = PaddleOCRExtractor(self.config.get('ocr', {}))

                # Check if table extraction is enabled (master switch)
                if self.config.get('ocr', {}).get('enable_table_extraction', True):
                    self.ocr_extractor = HybridOCRProcessor(base_ocr_extractor, self.config.get('ocr', {}))
                    logger.info("Hybrid OCR processor initialized with table structure recognition")
                else:
                    self.ocr_extractor = EnhancedPaddleOCR(base_ocr_extractor, self.config.get('ocr', {}))
                    logger.info("Enhanced PaddleOCR initialized (table extraction disabled)")

            except ImportError:
                logger.warning("PaddleOCR not available, using fallback")

        # Fallback to Tesseract
        if self.ocr_extractor is None:
            try:
                self.ocr_extractor = TesseractExtractor(self.config.get('ocr', {}))
                logger.info("Tesseract initialized as OCR engine")
            except ImportError:
                logger.error("No OCR engine available")

        logger.info("Extraction Router initialized")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)

        # Default config
        return {
            "routing": {
                "min_text_length": 50,
                "min_text_confidence": 0.7,
                "image_quality_threshold": 0.3,
                "prefer_scanned_text": True
            },
            "ocr": {
                "enable_paddleocr": True,
                "enable_table_recognition": False,
                "enable_table_extraction": False,  # Master switch for table detection and processing
                "confidence_threshold": 60.0,
                "fallback_engines": ["tesseract"],
                "table_detection_confidence": 0.6,
                "table_processing_timeout": 15,
                "min_horizontal_lines": 3,
                "min_vertical_lines": 2,
                "min_intersections": 4
            }
        }

    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Analyze PDF document to determine extraction method.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with analysis results and recommended method
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        analysis = {
            "file_path": str(pdf_path),
            "file_size": pdf_path.stat().st_size,
            "pages": [],
            "has_selectable_text": False,
            "recommended_method": None,
            "confidence": 0.0,
            "reason": ""
        }

        try:
            with fitz.open(pdf_path) as doc:
                analysis["total_pages"] = len(doc)
                total_text_length = 0
                pages_with_text = 0

                for page_num, page in enumerate(doc):
                    page_info = self._analyze_page(page, page_num + 1)
                    analysis["pages"].append(page_info)

                    if page_info["has_text"]:
                        analysis["has_selectable_text"] = True
                        pages_with_text += 1
                        total_text_length += page_info["text_length"]

                # Determine extraction method
                analysis = self._determine_extraction_method(
                    analysis,
                    total_text_length,
                    pages_with_text,
                    len(doc)
                )

        except Exception as e:
            logger.error(f"Error analyzing document {pdf_path}: {e}")
            analysis["recommended_method"] = "ocr"
            analysis["reason"] = f"Analysis failed, defaulting to OCR: {str(e)}"
            analysis["confidence"] = 0.5

        return analysis

    def _analyze_page(self, page, page_num: int) -> Dict[str, Any]:
        """Analyze individual page for text content and characteristics."""
        page_info = {
            "page_number": page_num,
            "has_text": False,
            "text_length": 0,
            "text_sample": "",
            "image_count": 0,
            "has_images": False
        }

        # Extract text
        text = page.get_text()
        if text.strip():
            page_info["has_text"] = True
            page_info["text_length"] = len(text.strip())
            page_info["text_sample"] = text[:200] + "..." if len(text) > 200 else text

        # Check for images
        image_list = page.get_images()
        page_info["image_count"] = len(image_list)
        page_info["has_images"] = len(image_list) > 0

        return page_info

    def _determine_extraction_method(self, analysis: Dict[str, Any],
                                    total_text_length: int,
                                    pages_with_text: int,
                                    total_pages: int) -> Dict[str, Any]:
        """Determine the optimal extraction method based on analysis."""
        routing_config = self.config.get('routing', {})
        min_text_length = routing_config.get('min_text_length', 50)
        prefer_scanned = routing_config.get('prefer_scanned_text', True)

        # Calculate text coverage
        text_coverage = pages_with_text / total_pages if total_pages > 0 else 0

        # Decision logic
        if analysis["has_selectable_text"] and total_text_length >= min_text_length:
            if prefer_scanned or text_coverage > 0.5:
                analysis["recommended_method"] = "scanned"
                analysis["confidence"] = 0.9
                analysis["reason"] = f"Document has {total_text_length} characters of selectable text across {pages_with_text}/{total_pages} pages"
            else:
                analysis["recommended_method"] = "scanned"
                analysis["confidence"] = 0.7
                analysis["reason"] = "Document has selectable text but mixed content"
        else:
            analysis["recommended_method"] = "ocr"
            analysis["confidence"] = 0.8
            if analysis["has_selectable_text"]:
                analysis["reason"] = f"Document has text but insufficient length ({total_text_length} chars), using OCR for better extraction"
            else:
                analysis["reason"] = "Document appears to be image-based, OCR required"

        return analysis

    def extract_text(self, pdf_path: Path, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text from PDF using recommended or specified method.

        Args:
            pdf_path: Path to PDF file
            method: Optional extraction method ('scanned', 'ocr', or None for auto-detect)

        Returns:
            Dictionary with extraction results
        """
        # Determine method
        if method is None:
            analysis = self.analyze_document(pdf_path)
            method = analysis["recommended_method"]
            logger.info(f"Auto-selected extraction method: {method}")
        else:
            logger.info(f"Using specified extraction method: {method}")

        # Perform extraction
        try:
            if method == "scanned":
                result = self.scanned_extractor.extract_scanned_text(pdf_path)
            elif method == "ocr":
                if self.ocr_extractor is None:
                    raise RuntimeError("OCR extractor not available")
                result = self.ocr_extractor.extract_text(pdf_path)
            else:
                raise ValueError(f"Unknown extraction method: {method}")

            # Add metadata
            result["extraction_method"] = method
            result["file_path"] = str(pdf_path)

            return result

        except Exception as e:
            logger.error(f"Extraction failed with method {method}: {e}")
            return {
                "success": False,
                "error": str(e),
                "extraction_method": method,
                "file_path": str(pdf_path)
            }

    def get_supported_methods(self) -> Dict[str, bool]:
        """
        Get information about supported extraction methods.

        Returns:
            Dictionary with method availability
        """
        return {
            "scanned": True,  # Always available with PyMuPDF
            "ocr": self.ocr_extractor is not None,
            "paddleocr": isinstance(self.ocr_extractor, PaddleOCRExtractor) if self.ocr_extractor else False,
            "tesseract": isinstance(self.ocr_extractor, TesseractExtractor) if self.ocr_extractor else False,
            "hybrid_ocr": isinstance(self.ocr_extractor, HybridOCRProcessor) if self.ocr_extractor else False,
            "table_recognition": isinstance(self.ocr_extractor, HybridOCRProcessor) if self.ocr_extractor else False
        }