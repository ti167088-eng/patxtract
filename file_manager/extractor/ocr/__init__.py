"""
PatXtract OCR Package

Contains OCR engine implementations for extracting text from image-based documents.
Includes PaddleOCR as primary engine and Tesseract as fallback.
"""

from .paddleocr import PaddleOCRExtractor
from .tesseract import TesseractExtractor

__all__ = ['PaddleOCRExtractor', 'TesseractExtractor']