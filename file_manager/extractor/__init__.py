"""
PatXtract Extractor Package

Handles text extraction from PDF documents using OCR and scanned text methods.
Intelligently routes documents to appropriate extraction methods.
"""

from .extraction_routing import ExtractionRouter
from .ocr.paddleocr import PaddleOCRExtractor
from .ocr.tesseract import TesseractExtractor

__all__ = ['ExtractionRouter', 'PaddleOCRExtractor', 'TesseractExtractor']