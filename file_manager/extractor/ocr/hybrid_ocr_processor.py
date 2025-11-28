"""
Hybrid OCR Processor

Intelligent combination of regular OCR and table structure recognition.
Uses fast pattern detection to identify table regions and applies appropriate processing.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .table_ocr_processor import TableOCRProcessor
from .fast_table_detector import FastTableDetector

logger = logging.getLogger(__name__)


class HybridOCRProcessor:
    """
    Hybrid OCR processor with intelligent table detection.

    Uses regular OCR for fast text extraction and table structure recognition
    for detected table regions, providing optimized performance and accuracy.
    """

    def __init__(self, base_ocr_engine, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Hybrid OCR processor.

        Args:
            base_ocr_engine: Base OCR engine (PaddleOCRExtractor)
            config: Configuration dictionary
        """
        self.base_ocr = base_ocr_engine
        self.config = config or {}

        # Table detection settings (master switch)
        self.config.setdefault('enable_table_extraction', True)  # Master switch
        self.config.setdefault('enable_table_detection', self.config.get('enable_table_extraction', True))
        self.config.setdefault('table_detection_confidence', 0.6)
        self.config.setdefault('min_table_rows', 2)
        self.config.setdefault('min_table_columns', 2)
        self.config.setdefault('table_processing_timeout', 15)

        # Initialize fast table detector and table processor
        self.fast_detector = None
        self.table_processor = None

        if self.config.get('enable_table_extraction', True):
            try:
                # Initialize fast table detector (OpenCV-based)
                self.fast_detector = FastTableDetector(self.config)
                logger.info("✅ Fast table detector initialized (OpenCV)")

                # Initialize table structure processor
                self.table_processor = TableOCRProcessor(self.config)
                logger.info("✅ Table structure recognition initialized")

            except Exception as e:
                logger.warning(f"⚠️  Table detection initialization failed: {e}")
                logger.info("   Falling back to regular OCR only")
                self.config['enable_table_detection'] = False
        else:
            logger.info("📝 Table extraction disabled - using regular OCR only")

    def extract_text_with_tables(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract text with intelligent table detection and processing.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary containing extracted text and table data
        """
        start_time = time.time()

        result = {
            "success": False,
            "content": [],
            "text": "",
            "tables": [],
            "has_tables": False,
            "processing_method": "hybrid",
            "processing_time": 0,
            "table_detection_time": 0,
            "error": None,
            "image_path": str(image_path)
        }

        try:
            logger.info(f"🔍 Starting hybrid OCR processing: {image_path}")

            if not image_path.exists():
                result["error"] = f"Image file not found: {image_path}"
                return result

            # Step 1: Fast regular OCR for baseline text
            regular_result = self.base_ocr.extract_text(image_path)
            result["processing_time"] = time.time() - start_time

            if not regular_result.get("success"):
                result["error"] = regular_result.get("error", "Regular OCR failed")
                return result

            result["text"] = regular_result.get("text", "")

            # Step 2: Fast table detection using OpenCV (10-20ms) - only if enabled
            table_detection_start = time.time()
            has_tables = False
            table_confidence = 0.0

            if self.config.get('enable_table_extraction', True) and self.fast_detector:
                has_tables, table_confidence = self._detect_tables_with_opencv(image_path)
                result["detection_method"] = "opencv_line_detection"
            else:
                result["detection_method"] = "disabled"

            result["table_detection_time"] = time.time() - table_detection_start

            if has_tables and self.table_processor:
                logger.info(f"📊 Tables detected (confidence: {table_confidence:.1%}), running structure recognition...")

                # Step 3: Extract table structure
                try:
                    table_result = self.table_processor.extract_table_structure(image_path)

                    if table_result.get("success") and table_result.get("tables"):
                        result["has_tables"] = True
                        result["tables"] = table_result["tables"]

                        # Step 4: Combine regular text with structured tables
                        result["content"] = self._combine_text_and_tables(regular_result, table_result)
                        result["text"] = self._format_hybrid_output(result["content"])

                        logger.info(f"✅ Hybrid extraction completed: {len(result['tables'])} table(s) found")
                    else:
                        logger.warning("⚠️  Table structure recognition failed, using regular OCR text")
                        result["content"] = [{"type": "text", "content": regular_result["text"]}]

                except Exception as e:
                    logger.error(f"❌ Table processing error: {e}")
                    result["content"] = [{"type": "text", "content": regular_result["text"]}]
            else:
                # No tables detected, use regular text
                result["content"] = [{"type": "text", "content": regular_result["text"]}]

            result["success"] = True
            result["processing_time"] = time.time() - start_time

            return result

        except Exception as e:
            logger.error(f"❌ Hybrid OCR processing failed: {e}")
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time
            return result

    def _detect_tables_with_opencv(self, image_path: Path) -> Tuple[bool, float]:
        """
        Fast table detection using OpenCV line detection.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (has_tables, confidence_score)
        """
        try:
            if not self.fast_detector:
                logger.warning("Fast table detector not initialized")
                return False, 0.0

            # Run ultra-fast OpenCV table detection (10-20ms)
            detection_result = self.fast_detector.detect_table_presence(image_path)

            has_tables = detection_result["has_table"]
            confidence = detection_result["confidence"]

            logger.debug(f"OpenCV table detection: {has_tables} ({confidence:.1%}) "
                        f"({detection_result['processing_time']:.1f}ms) - {detection_result['reason']}")

            return has_tables, confidence

        except Exception as e:
            logger.error(f"❌ OpenCV table detection error: {e}")
            return False, 0.0

    
    def _combine_text_and_tables(self, regular_result: Dict[str, Any], table_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Combine regular OCR text with structured table data.

        Args:
            regular_result: Regular OCR result
            table_result: Table OCR result

        Returns:
            Combined content list
        """
        content = []

        # Add regular text content
        if regular_result.get("text"):
            content.append({
                "type": "text",
                "content": regular_result["text"],
                "confidence": regular_result.get("confidence", 0.0)
            })

        # Add structured table data
        for table in table_result.get("tables", []):
            table_content = {
                "type": "table",
                "table_id": table.get("table_number", 0),
                "structure": table.get("structure", {}),
                "confidence": table.get("confidence", 0.0)
            }

            # Add markdown representation
            if table.get("structure", {}).get("grid"):
                table_content["markdown"] = self._format_table_as_markdown(table["structure"]["grid"])

            content.append(table_content)

        return content

    def _format_table_as_markdown(self, grid: List[List[str]]) -> str:
        """Format table grid as markdown."""
        if not grid or not grid[0]:
            return ""

        max_cols = max(len(row) for row in grid)
        markdown_lines = []

        # Header row
        header_row = [cell if cell else "" for cell in grid[0][:max_cols]]
        while len(header_row) < max_cols:
            header_row.append("")
        markdown_lines.append("| " + " | ".join(header_row) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        # Data rows
        for row in grid[1:]:
            formatted_row = [cell if cell else "" for cell in row[:max_cols]]
            while len(formatted_row) < max_cols:
                formatted_row.append("")
            markdown_lines.append("| " + " | ".join(formatted_row) + " |")

        return "\n".join(markdown_lines)

    def _format_hybrid_output(self, content: List[Dict[str, Any]]) -> str:
        """Format hybrid content for text output."""
        formatted_parts = []

        for item in content:
            if item["type"] == "text":
                formatted_parts.append(item["content"])
            elif item["type"] == "table":
                formatted_parts.append(f"\n--- TABLE {item['table_id']} ---")
                if item.get("markdown"):
                    formatted_parts.append(item["markdown"])
                if item.get("structure", {}).get("grid"):
                    formatted_parts.append("\n--- STRUCTURED DATA ---")
                    for row in item["structure"]["grid"]:
                        formatted_parts.append(" | ".join(row))

        return "\n".join(formatted_parts)