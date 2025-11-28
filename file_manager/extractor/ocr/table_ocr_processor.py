"""
Table OCR Processor

Uses PaddleOCR table structure recognition to extract table data with proper columns and rows.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from paddleocr import PaddleOCR
    TABLE_OCR_AVAILABLE = True
except ImportError:
    TABLE_OCR_AVAILABLE = False

logger = logging.getLogger(__name__)


class TableOCRProcessor:
    """
    PaddleOCR Table Structure Recognition Processor.

    Uses ch_ppstructure_mobile_v2.0_SLANet model for proper table extraction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Table OCR processor.

        Args:
            config: Configuration dictionary
        """
        if not TABLE_OCR_AVAILABLE:
            raise ImportError("PaddleOCR is required for table processing")

        self.config = config or {}
        self.config.setdefault('lang', 'ch')  # Chinese+English for table processing
        self.config.setdefault('use_gpu', True)
        self.config.setdefault('gpu_mem', 8000)
        self.config.setdefault('gpu_device_id', 0)

        # Initialize PaddleOCR with table structure recognition
        try:
            logger.info("🔧 Initializing PaddleOCR with Table Structure Recognition...")

            self.table_ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.config['lang'],
                use_gpu=self.config['use_gpu']
            )

            logger.info("✅ Table OCR initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Table OCR: {e}")
            raise

    def extract_table_structure(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract table structure and content from image.

        Args:
            image_path: Path to image file containing table

        Returns:
            Dictionary containing table structure and content
        """
        if not image_path.exists():
            return {
                "success": False,
                "error": f"Image file not found: {image_path}",
                "tables": []
            }

        try:
            logger.info(f"📊 Processing table image: {image_path}")

            # Use PaddleOCR table structure recognition
            result = self.table_ocr.ocr(str(image_path), cls=True)

            if not result:
                return {
                    "success": False,
                    "error": "No table detected in image",
                    "tables": []
                }

            tables = []

            # Process OCR results for table structure
            for idx, res in enumerate(result):
                if len(res) > 0:
                    table_data = self._process_table_result(res, idx + 1)
                    if table_data:
                        tables.append(table_data)

            logger.info(f"✅ Extracted {len(tables)} table(s) from image")

            return {
                "success": True,
                "tables": tables,
                "total_tables": len(tables),
                "image_path": str(image_path)
            }

        except Exception as e:
            logger.error(f"❌ Table extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tables": []
            }

    def _process_table_result(self, ocr_result: List, table_num: int) -> Optional[Dict[str, Any]]:
        """
        Process OCR result to extract table structure.

        Args:
            ocr_result: OCR result from PaddleOCR
            table_num: Table number identifier

        Returns:
            Table data dictionary with structure
        """
        try:
            if not ocr_result:
                return None

            # Extract text and bounding boxes
            table_data = {
                "table_number": table_num,
                "success": True,
                "cells": [],
                "structure": None,
                "raw_text": []
            }

            # Process each detected element
            for line in ocr_result:
                if line and len(line) >= 2:
                    # Get text and bounding box
                    text_box = line[0]  # Bounding box
                    text_info = line[1]  # Text and confidence

                    if text_info and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]

                        if text.strip():  # Only add non-empty text
                            cell_data = {
                                "text": text,
                                "confidence": confidence,
                                "bbox": text_box,
                                "normalized_position": self._normalize_position(text_box)
                            }
                            table_data["cells"].append(cell_data)
                            table_data["raw_text"].append(text)

            # Try to infer table structure
            table_data["structure"] = self._infer_table_structure(table_data["cells"])

            return table_data

        except Exception as e:
            logger.error(f"❌ Error processing table result: {e}")
            return None

    def _normalize_position(self, bbox: List) -> Dict[str, float]:
        """
        Normalize bounding box position for table structure analysis.

        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            Normalized position dictionary
        """
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            return {
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1
            }
        return {}

    def _infer_table_structure(self, cells: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Infer table structure (rows and columns) from cell positions.

        Args:
            cells: List of cell data with position information

        Returns:
            Inferred table structure
        """
        if not cells:
            return {"rows": 0, "columns": 0, "grid": []}

        try:
            # Group cells by approximate Y positions (rows)
            y_positions = [cell["normalized_position"]["center_y"] for cell in cells]

            # Sort and group into rows using simple clustering
            sorted_positions = sorted(zip(y_positions, range(len(y_positions))))
            rows = self._cluster_into_rows(sorted_positions)

            # For each row, sort cells by X position (columns)
            table_grid = []
            for row_indices in rows:
                row_cells = [cells[i] for i in row_indices]
                # Sort by X position within the row
                row_cells.sort(key=lambda c: c["normalized_position"]["center_x"])
                table_grid.append([cell["text"] for cell in row_cells])

            max_cols = max(len(row) for row in table_grid) if table_grid else 0

            return {
                "rows": len(table_grid),
                "columns": max_cols,
                "grid": table_grid,
                "raw_cells": cells
            }

        except Exception as e:
            logger.error(f"❌ Error inferring table structure: {e}")
            return {"rows": 0, "columns": 0, "grid": []}

    def _cluster_into_rows(self, sorted_positions: List, threshold: float = 10.0) -> List[List[int]]:
        """
        Cluster Y positions into rows.

        Args:
            sorted_positions: List of (y_position, index) tuples sorted by Y
            threshold: Maximum distance between cells in same row

        Returns:
            List of rows, each containing cell indices
        """
        if not sorted_positions:
            return []

        rows = []
        current_row = [sorted_positions[0][1]]
        current_y = sorted_positions[0][0]

        for y_pos, idx in sorted_positions[1:]:
            if abs(y_pos - current_y) <= threshold:
                # Same row
                current_row.append(idx)
            else:
                # New row
                rows.append(current_row)
                current_row = [idx]
                current_y = y_pos

        rows.append(current_row)  # Add last row
        return rows

    def format_table_as_markdown(self, table_data: Dict[str, Any]) -> str:
        """
        Convert table data to markdown format.

        Args:
            table_data: Table data dictionary

        Returns:
            Markdown formatted table
        """
        if not table_data.get("success") or not table_data.get("structure", {}).get("grid"):
            return "No table data available"

        structure = table_data["structure"]
        grid = structure["grid"]

        if not grid:
            return "Empty table"

        # Create markdown table
        markdown_lines = []

        # Determine column count from longest row
        max_cols = structure["columns"]

        for row_idx, row in enumerate(grid):
            # Ensure all rows have same number of columns
            formatted_row = [cell if cell else "" for cell in row]
            while len(formatted_row) < max_cols:
                formatted_row.append("")

            # Join cells with pipe separator
            row_text = " | ".join(formatted_row)
            markdown_lines.append(f"| {row_text} |")

            # Add header separator after first row (or assume first row is header)
            if row_idx == 0:
                separator = " | ".join(["---"] * max_cols)
                markdown_lines.append(f"| {separator} |")

        return "\n".join(markdown_lines)


def download_table_model():
    """
    Download the table structure recognition model.

    Returns:
        bool: True if download successful
    """
    try:
        logger.info("📥 Downloading PaddleOCR table structure model...")

        # Initialize PaddleOCR to trigger model download
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch'
        )

        logger.info("✅ Table structure model downloaded successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download table model: {e}")
        return False