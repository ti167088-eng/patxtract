"""
PatXtract Scanned Text Package

Handles extraction of text from PDF documents that already contain selectable text.
No OCR processing required for faster and more accurate extraction.
"""

from .scan import ScannedTextExtractor

__all__ = ['ScannedTextExtractor']