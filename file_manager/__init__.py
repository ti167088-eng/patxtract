"""
PatXtract File Manager Package

A comprehensive medical document processing system that handles PDF documents,
routes them to appropriate extraction methods, and provides structured output.
"""

from .manager.manager import FileManager
from .extractor.extraction_routing import ExtractionRouter

__version__ = "1.0.0"
__all__ = ['FileManager', 'ExtractionRouter']