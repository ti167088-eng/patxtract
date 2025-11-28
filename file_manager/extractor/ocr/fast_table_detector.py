"""
Fast Table Detector

Uses OpenCV line detection for ultra-fast table presence identification.
Runs in 10-20ms per page to determine if SLANet table processing should be triggered.
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class FastTableDetector:
    """
    Ultra-fast table presence detector using OpenCV line detection.

    Detects horizontal and vertical lines to identify table grid structures
    without running expensive table structure models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fast table detector.

        Args:
            config: Configuration dictionary with detection parameters
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for FastTableDetector")

        self.config = config or {}

        # Detection parameters
        self.config.setdefault('min_horizontal_lines', 3)
        self.config.setdefault('min_vertical_lines', 2)
        self.config.setdefault('min_intersections', 4)
        self.config.setdefault('line_detection_threshold', 50)
        self.config.setdefault('min_line_length', 50)
        self.config.setdefault('max_line_gap', 10)

        # Performance optimization
        self.config.setdefault('resize_max_dimension', 1000)  # Resize for faster processing
        self.config.setdefault('gaussian_blur_kernel', (3, 3))

        logger.info("Fast Table Detector initialized with OpenCV line detection")

    def detect_table_presence(self, image_path: Path) -> Dict[str, Any]:
        """
        Detect if image contains table structures.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with detection results and confidence
        """
        start_time = cv2.getTickCount()

        result = {
            "has_table": False,
            "confidence": 0.0,
            "horizontal_lines": 0,
            "vertical_lines": 0,
            "intersections": 0,
            "processing_time": 0.0,
            "image_size": None,
            "detection_method": "opencv_line_detection",
            "reason": ""
        }

        try:
            if not image_path.exists():
                result["reason"] = f"Image file not found: {image_path}"
                return result

            # Load and preprocess image
            image, original_size = self._load_and_preprocess(image_path)
            result["image_size"] = original_size

            # Detect lines
            horizontal_lines = self._detect_horizontal_lines(image)
            vertical_lines = self._detect_vertical_lines(image)

            result["horizontal_lines"] = len(horizontal_lines)
            result["vertical_lines"] = len(vertical_lines)

            # Calculate intersections
            intersections = self._count_intersections(horizontal_lines, vertical_lines)
            result["intersections"] = intersections

            # Make decision
            has_table, confidence, reason = self._make_table_decision(
                len(horizontal_lines), len(vertical_lines), intersections
            )

            result["has_table"] = has_table
            result["confidence"] = confidence
            result["reason"] = reason

        except Exception as e:
            logger.error(f"❌ Table detection failed: {e}")
            result["reason"] = f"Detection error: {str(e)}"

        finally:
            # Calculate processing time
            processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
            result["processing_time"] = processing_time

            logger.debug(f"Table detection: {result['has_table']} ({result['confidence']:.1%}) "
                        f"in {processing_time:.1f}ms")

        return result

    def _load_and_preprocess(self, image_path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Load and preprocess image for line detection."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        original_size = (image.shape[1], image.shape[0])  # (width, height)

        # Resize for performance if needed
        max_dim = self.config['resize_max_dimension']
        if max(image.shape) > max_dim:
            scale = max_dim / max(image.shape)
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.config['gaussian_blur_kernel'], 0)

        # Apply adaptive threshold for better line detection
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Invert colors (lines should be white on black)
        binary = cv2.bitwise_not(binary)

        return binary, original_size

    def _detect_horizontal_lines(self, image: np.ndarray) -> np.ndarray:
        """Detect horizontal lines using morphological operations."""
        # Create horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.config['min_line_length'], 1)
        )

        # Apply morphological operations
        horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Find lines using HoughLinesP as backup method
        lines = cv2.HoughLinesP(
            horizontal,
            rho=1,
            theta=np.pi/180,
            threshold=self.config['line_detection_threshold'],
            minLineLength=self.config['min_line_length'],
            maxLineGap=self.config['max_line_gap']
        )

        if lines is None:
            return np.array([])

        # Filter for nearly horizontal lines (angle within 15 degrees)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 20:  # Nearly horizontal
                horizontal_lines.append([x1, y1, x2, y2])

        return np.array(horizontal_lines) if horizontal_lines else np.array([])

    def _detect_vertical_lines(self, image: np.ndarray) -> np.ndarray:
        """Detect vertical lines using morphological operations."""
        # Create vertical kernel
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, self.config['min_line_length'])
        )

        # Apply morphological operations
        vertical = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Find lines using HoughLinesP
        lines = cv2.HoughLinesP(
            vertical,
            rho=1,
            theta=np.pi/180,
            threshold=self.config['line_detection_threshold'],
            minLineLength=self.config['min_line_length'],
            maxLineGap=self.config['max_line_gap']
        )

        if lines is None:
            return np.array([])

        # Filter for nearly vertical lines (angle within 15 degrees)
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 20:  # Nearly vertical
                vertical_lines.append([x1, y1, x2, y2])

        return np.array(vertical_lines) if vertical_lines else np.array([])

    def _count_intersections(self, horizontal_lines: np.ndarray,
                           vertical_lines: np.ndarray) -> int:
        """Count intersections between horizontal and vertical lines."""
        if len(horizontal_lines) == 0 or len(vertical_lines) == 0:
            return 0

        intersections = 0
        tolerance = 10  # Pixels tolerance for intersection

        for h_line in horizontal_lines:
            x1_h, y1_h, x2_h, y2_h = h_line

            for v_line in vertical_lines:
                x1_v, y1_v, x2_v, y2_v = v_line

                # Check if lines intersect
                # Simple bounding box intersection check
                if (min(x1_h, x2_h) - tolerance <= x1_v <= max(x1_h, x2_h) + tolerance and
                    min(y1_v, y2_v) - tolerance <= y1_h <= max(y1_v, y2_v) + tolerance):
                    intersections += 1

        return intersections

    def _make_table_decision(self, horizontal_count: int, vertical_count: int,
                           intersections: int) -> Tuple[bool, float, str]:
        """Make final decision about table presence."""
        min_h = self.config['min_horizontal_lines']
        min_v = self.config['min_vertical_lines']
        min_i = self.config['min_intersections']

        # Calculate confidence based on how well criteria are met
        h_score = min(horizontal_count / min_h, 2.0) / 2.0  # Cap at 1.0, weight 50%
        v_score = min(vertical_count / min_v, 2.0) / 2.0   # Cap at 1.0, weight 30%
        i_score = min(intersections / min_i, 2.0) / 2.0    # Cap at 1.0, weight 20%

        confidence = (h_score * 0.5 + v_score * 0.3 + i_score * 0.2)

        # Make decision
        if (horizontal_count >= min_h and
            vertical_count >= min_v and
            intersections >= min_i):

            reason = f"Table detected: {horizontal_count}H, {vertical_count}V, {intersections} intersections"
            return True, confidence, reason

        else:
            reason = (f"No table: {horizontal_count}/{min_h}H, "
                     f"{vertical_count}/{min_v}V, {intersections}/{min_i} intersections")
            return False, confidence, reason

    def batch_detect(self, image_paths: list[Path]) -> list[Dict[str, Any]]:
        """
        Detect table presence in multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of detection results
        """
        results = []
        for image_path in image_paths:
            result = self.detect_table_presence(image_path)
            results.append(result)

        return results