"""
PatXtract Scanned Text Extractor

Handles extraction of text from PDF documents that already contain selectable text.
No OCR processing required for faster and more accurate extraction.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
import json

logger = logging.getLogger(__name__)


class ScannedTextExtractor:
    """
    Extracts text from PDF documents with existing selectable text.

    Provides fast, accurate text extraction without the need for OCR processing.
    Preserves document structure, formatting, and metadata.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize scanned text extractor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.config.setdefault('preserve_whitespace', True)
        self.config.setdefault('extract_images', False)
        self.config.setdefault('include_metadata', True)

        logger.info("Scanned Text Extractor initialized")

    def extract_scanned_text(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text from PDF with existing selectable text.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted text and metadata
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        start_time = time.time()
        logger.info(f"Starting scanned text extraction for: {pdf_path}")

        result = {
            "success": False,
            "file_path": str(pdf_path),
            "file_name": pdf_path.name,
            "extraction_method": "scanned",
            "pages": [],
            "total_pages": 0,
            "total_characters": 0,
            "processing_time": 0.0,
            "metadata": {},
            "errors": []
        }

        try:
            with fitz.open(pdf_path) as doc:
                result["total_pages"] = len(doc)

                # Extract document metadata
                if self.config.get('include_metadata', True):
                    result["metadata"] = self._extract_metadata(doc)

                # Process each page
                for page_num, page in enumerate(doc):
                    page_result = self._extract_page_text(page, page_num + 1)
                    result["pages"].append(page_result)

                    if page_result["success"]:
                        result["total_characters"] += page_result["text_length"]

                result["success"] = True
                logger.info(f"Successfully extracted {result['total_characters']} characters from {len(doc)} pages")

        except Exception as e:
            error_msg = f"Error during scanned text extraction: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)

        # Calculate processing time
        result["processing_time"] = time.time() - start_time

        return result

    def _extract_metadata(self, doc) -> Dict[str, Any]:
        """Extract document metadata."""
        metadata = {}

        try:
            # Basic metadata
            metadata["title"] = doc.metadata.get("title", "")
            metadata["author"] = doc.metadata.get("author", "")
            metadata["subject"] = doc.metadata.get("subject", "")
            metadata["creator"] = doc.metadata.get("creator", "")
            metadata["producer"] = doc.metadata.get("producer", "")
            metadata["creation_date"] = doc.metadata.get("creationDate", "")
            metadata["modification_date"] = doc.metadata.get("modDate", "")

            # Page size information
            if doc.page_count > 0:
                first_page = doc[0]
                rect = first_page.rect
                metadata["page_size"] = {
                    "width": rect.width,
                    "height": rect.height,
                    "unit": "points"
                }

            # PDF version
            try:
                metadata["pdf_version"] = doc.pdf_version()
            except AttributeError:
                metadata["pdf_version"] = "Unknown"

        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            metadata["extraction_error"] = str(e)

        return metadata

    def _extract_page_text(self, page, page_num: int) -> Dict[str, Any]:
        """Extract text from individual page."""
        page_result = {
            "page_number": page_num,
            "success": False,
            "text": "",
            "text_length": 0,
            "text_blocks": [],
            "font_info": [],
            "images": [],
            "errors": []
        }

        try:
            # Extract full page text
            text = page.get_text()

            # Debug logging
            logger.debug(f"Page {page_num}: Extracted {len(text)} characters of raw text")
            if text.strip():
                logger.debug(f"Page {page_num}: Text preview: {text[:100]}...")

            if text.strip():
                page_result["success"] = True
                page_result["text"] = text
                page_result["text_length"] = len(text.strip())

                # Extract text blocks with formatting
                if self.config.get('extract_blocks', True):
                    page_result["text_blocks"] = self._extract_text_blocks(page)

                # Extract font information
                if self.config.get('extract_fonts', False):
                    page_result["font_info"] = self._extract_font_info(page)

            else:
                page_result["errors"].append("No selectable text found on page")
                logger.debug(f"Page {page_num}: No selectable text found")

            # Extract image information
            if self.config.get('extract_images', False):
                page_result["images"] = self._extract_image_info(page)

        except Exception as e:
            error_msg = f"Error extracting text from page {page_num}: {str(e)}"
            logger.error(error_msg)
            page_result["errors"].append(error_msg)

        return page_result

    def _extract_text_blocks(self, page) -> List[Dict[str, Any]]:
        """Extract text blocks with position and formatting information."""
        blocks = []

        try:
            text_blocks = page.get_text("dict")

            for block in text_blocks.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    block_info = {
                        "type": "text",
                        "bbox": block.get("bbox", []),
                        "lines": []
                    }

                    for line in block.get("lines", []):
                        line_info = {
                            "bbox": line.get("bbox", []),
                            "spans": [],
                            "text": ""
                        }

                        line_text_parts = []
                        for span in line.get("spans", []):
                            span_info = {
                                "text": span.get("text", ""),
                                "bbox": span.get("bbox", []),
                                "font": span.get("font", ""),
                                "size": span.get("size", 0),
                                "flags": span.get("flags", 0),
                                "color": span.get("color", 0)
                            }
                            line_info["spans"].append(span_info)
                            line_text_parts.append(span_info["text"])

                        line_info["text"] = " ".join(line_text_parts)
                        block_info["lines"].append(line_info)

                    blocks.append(block_info)

        except Exception as e:
            logger.warning(f"Error extracting text blocks: {e}")

        return blocks

    def _extract_font_info(self, page) -> List[Dict[str, Any]]:
        """Extract font information from page."""
        fonts = []

        try:
            text_dict = page.get_text("dict")
            font_set = set()

            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font_info = {
                                "font": span.get("font", ""),
                                "size": span.get("size", 0),
                                "flags": span.get("flags", 0),
                                "color": span.get("color", 0)
                            }
                            font_signature = f"{font_info['font']}_{font_info['size']}_{font_info['flags']}"

                            if font_signature not in font_set:
                                font_set.add(font_signature)
                                fonts.append(font_info)

        except Exception as e:
            logger.warning(f"Error extracting font info: {e}")

        return fonts

    def _extract_image_info(self, page) -> List[Dict[str, Any]]:
        """Extract image information from page."""
        images = []

        try:
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                img_info = {
                    "index": img_index,
                    "xref": img[0],
                    "width": img[2],
                    "height": img[3],
                    "bpc": img[4],  # bits per component
                    "colorspace": img[5]
                }
                images.append(img_info)

        except Exception as e:
            logger.warning(f"Error extracting image info: {e}")

        return images

    def extract_text_only(self, pdf_path: Path) -> str:
        """
        Extract only the text content from PDF (simplified version).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Concatenated text from all pages
        """
        try:
            with fitz.open(pdf_path) as doc:
                all_text = []
                for page in doc:
                    text = page.get_text()
                    if text.strip():
                        all_text.append(text.strip())

                return "\n\n".join(all_text)

        except Exception as e:
            logger.error(f"Error extracting text only from {pdf_path}: {e}")
            return ""

    def get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in PDF."""
        try:
            with fitz.open(pdf_path) as doc:
                return len(doc)
        except Exception as e:
            logger.error(f"Error getting page count from {pdf_path}: {e}")
            return 0