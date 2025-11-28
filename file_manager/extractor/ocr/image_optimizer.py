"""
Image Optimization for OCR Processing

Optimizes images for faster OCR processing while maintaining accuracy.
Implements intelligent downscaling and grayscale conversion to reduce memory usage
and improve processing speed.

Key Features:
- Intelligent downscaling for large images (>2500-3000px width)
- Safe 8-bit grayscale conversion for text-heavy content
- Memory-efficient processing with quality preservation
- Content-aware optimization based on document type
"""

import logging
import io
from typing import Tuple, Optional, Union
from PIL import Image, ImageStat
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class ImageOptimizer:
    """
    Intelligent image optimizer for OCR preprocessing.

    Provides optimized image processing pipeline that reduces memory usage
    and improves OCR speed while maintaining accuracy through conservative
    downscaling and smart color space conversion.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize image optimizer with configuration.

        Args:
            config: Configuration dictionary with optimization settings
        """
        self.config = config or self._get_default_config()
        self.optimization_stats = {
            'images_processed': 0,
            'memory_saved_mb': 0,
            'total_savings_percent': 0
        }

    def _get_default_config(self) -> dict:
        """Get default configuration for image optimization."""
        return {
            'downscaling': {
                'enabled': True,
                'width_threshold': 2800,  # Scale down images > 2800px wide
                'target_width': 1800,     # Target width for downscaling
                'min_dpi': 200,           # Minimum DPI to preserve text quality
                'preserve_aspect_ratio': True,
                'resampling_method': 'LANCZOS'
            },
            'grayscale': {
                'enabled': True,
                'auto_detect': True,      # Auto-detect when grayscale is safe
                'force_for_text_docs': True,
                'preserve_color_for_ids': True  # Keep color for ID cards with photos
            },
            'quality': {
                'jpeg_quality': 85,       # Quality for JPEG compression if needed
                'min_confidence_impact': 2.0  # Max acceptable confidence loss (%)
            }
        }

    def optimize_pil_image(self, image: Image.Image, document_type: str = 'general') -> Tuple[Image.Image, dict]:
        """
        Optimize a PIL Image for OCR processing.

        Args:
            image: PIL Image object to optimize
            document_type: Type of document ('text', 'id_card', 'general')

        Returns:
            Tuple of (optimized_image, optimization_metadata)
        """
        metadata = {
            'original_size': image.size,
            'original_mode': image.mode,
            'optimizations_applied': [],
            'memory_reduction': 0
        }

        optimized_image = image.copy()
        original_memory = self._estimate_image_memory(image)

        logger.debug(f"Starting optimization for {image.size} {image.mode} image")

        # Step 1: Intelligent downscaling
        if self.config['downscaling']['enabled']:
            optimized_image, downscale_info = self._intelligent_downscale(optimized_image)
            if downscale_info['scaled']:
                metadata['optimizations_applied'].append('downscaling')
                metadata['downscale_info'] = downscale_info
                logger.info(f"Downscaled from {image.size} to {optimized_image.size}")

        # Step 2: Grayscale conversion
        if self.config['grayscale']['enabled']:
            should_grayscale = self._should_convert_to_grayscale(optimized_image, document_type)
            if should_grayscale:
                optimized_image = self._convert_to_grayscale(optimized_image)
                metadata['optimizations_applied'].append('grayscale')
                metadata['final_mode'] = optimized_image.mode
                logger.debug(f"Converted to grayscale: {image.mode} -> {optimized_image.mode}")

        # Calculate memory savings
        final_memory = self._estimate_image_memory(optimized_image)
        metadata['memory_reduction'] = original_memory - final_memory
        metadata['memory_reduction_percent'] = (metadata['memory_reduction'] / original_memory) * 100

        # Update stats
        self.optimization_stats['images_processed'] += 1
        self.optimization_stats['memory_saved_mb'] += metadata['memory_reduction'] / (1024 * 1024)

        metadata['final_size'] = optimized_image.size
        metadata['final_memory_mb'] = final_memory / (1024 * 1024)

        logger.debug(f"Optimization complete: {metadata['memory_reduction_percent']:.1f}% memory reduction")

        return optimized_image, metadata

    def optimize_pdf_page(self, page, dpi: Optional[int] = None, document_type: str = 'general') -> Tuple[Image.Image, dict]:
        """
        Extract and optimize image from PDF page directly.

        Args:
            page: PyMuPDF page object
            dpi: Target DPI (auto-calculated if None)
            document_type: Type of document for content-aware optimization

        Returns:
            Tuple of (optimized_image, optimization_metadata)
        """
        # Calculate optimal DPI based on page content and optimization settings
        if dpi is None:
            dpi = self._calculate_optimal_dpi(page, document_type)

        metadata = {
            'original_dpi': dpi,
            'document_type': document_type,
            'optimization_method': 'pdf_direct'
        }

        # Extract with optimal DPI to avoid unnecessary high resolution
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes("png")

        with Image.open(io.BytesIO(img_data)) as image:
            optimized_image, optimization_metadata = self.optimize_pil_image(image, document_type)

        # Combine metadata
        metadata.update(optimization_metadata)
        metadata['pdf_optimization'] = True

        return optimized_image, metadata

    def _intelligent_downscale(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """
        Intelligently downscale large images while preserving text quality.
        Implements two-tier scaling: aggressive for very large images, conservative for moderate.

        Args:
            image: PIL Image to potentially downscale

        Returns:
            Tuple of (possibly_resized_image, scaling_info)
        """
        width, height = image.size
        max_dimension = max(width, height)

        scaling_info = {
            'scaled': False,
            'original_size': image.size,
            'scale_factor': 1.0,
            'reason': None,
            'scaling_tier': 'none'
        }

        threshold = self.config['downscaling']['width_threshold']
        target_size = self.config['downscaling']['target_width']

        # Check if downscaling is needed
        if width > threshold or height > threshold:
            scaling_info['reason'] = f"Image size ({width}x{height}) exceeds threshold ({threshold}px)"

            # Two-tier scaling approach
            if max_dimension > 3500:
                # AGGRESSIVE TIER: Very large images (>3500px) - prioritize memory efficiency
                scaling_info['scaling_tier'] = 'aggressive'

                # More aggressive target for very large images
                aggressive_target = max(target_size, 1200)  # Ensure minimum 1200px
                scale_factor = min(aggressive_target / width, aggressive_target / height)

                # For very large images, allow more flexible DPI requirements
                min_dpi = max(150, self.config['downscaling']['min_dpi'] - 50)  # Allow 150 DPI minimum

                logger.info(f"Using aggressive scaling for very large image: {max_dimension}px > 3500px threshold")

            else:
                # CONSERVATIVE TIER: Moderately large images (2500-3500px) - prioritize quality
                scaling_info['scaling_tier'] = 'conservative'

                scale_factor = min(target_size / width, target_size / height)
                min_dpi = self.config['downscaling']['min_dpi']

                logger.info(f"Using conservative scaling for moderately large image: {max_dimension}px")

            scaling_info['scale_factor'] = scale_factor
            scaling_info['min_dpi_used'] = min_dpi

            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            new_size = (new_width, new_height)

            # Verify DPI requirements, but be more flexible for aggressive tier
            estimated_dpi = self._estimate_dpi_after_scaling(image, scale_factor)

            if estimated_dpi < min_dpi:
                if scaling_info['scaling_tier'] == 'aggressive':
                    # For aggressive tier, use graduated DPI protection
                    if max_dimension > 5000:
                        # Ultra-large images: allow lower DPI
                        adjusted_min_dpi = max(120, min_dpi - 30)
                        logger.info(f"Ultra-large image: allowing lower DPI {adjusted_min_dpi} vs {estimated_dpi:.1f}")
                        if estimated_dpi >= adjusted_min_dpi:
                            # Accept the scaling with warning
                            logger.warning(f"Accepting aggressive scaling for ultra-large image: DPI {estimated_dpi:.1f} < preferred {min_dpi}")
                        else:
                            # Still apply minimal adjustment
                            conservative_factor = adjusted_min_dpi / estimated_dpi
                            scale_factor *= conservative_factor
                            new_width = int(width * scale_factor)
                            new_height = int(height * scale_factor)
                            new_size = (new_width, new_height)
                            scaling_info['conservative_adjustment'] = True
                            scaling_info['scale_factor'] = scale_factor
                    else:
                        # Large but not ultra-large: standard conservative adjustment
                        conservative_factor = min_dpi / estimated_dpi
                        scale_factor *= conservative_factor
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        new_size = (new_width, new_height)
                        scaling_info['conservative_adjustment'] = True
                        scaling_info['scale_factor'] = scale_factor
                        logger.info(f"Aggressive tier: applied conservative DPI adjustment")
                else:
                    # Conservative tier: standard DPI protection
                    logger.warning(f"Conservative scaling would reduce DPI below minimum ({estimated_dpi:.1f} < {min_dpi})")
                    conservative_factor = min_dpi / estimated_dpi
                    scale_factor *= conservative_factor
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    new_size = (new_width, new_height)
                    scaling_info['conservative_adjustment'] = True
                    scaling_info['scale_factor'] = scale_factor

            # Additional safety check: ensure we don't exceed 4000px limit for OCR compatibility
            final_max_dimension = max(new_width, new_height)
            if final_max_dimension > 4000:
                safety_factor = 3800 / final_max_dimension  # Leave some margin
                new_width = int(new_width * safety_factor)
                new_height = int(new_height * safety_factor)
                scale_factor *= safety_factor
                scaling_info['safety_adjustment'] = True
                scaling_info['scale_factor'] = scale_factor
                logger.warning(f"Applied safety adjustment to stay under 4000px limit: {final_max_dimension}px -> {max(new_width, new_height)}px")

            # Perform high-quality downscaling
            resampling_method = getattr(Image.Resampling, self.config['downscaling']['resampling_method'])
            resized_image = image.resize((new_width, new_height), resampling_method)

            scaling_info['new_size'] = (new_width, new_height)
            scaling_info['scaled'] = True
            scaling_info['estimated_dpi'] = self._estimate_dpi_after_scaling(image, scale_factor)

            logger.info(f"Intelligently downscaled ({scaling_info['scaling_tier']} tier): {image.size} -> {(new_width, new_height)} "
                       f"(scale: {scale_factor:.3f}, est. DPI: {scaling_info['estimated_dpi']:.1f})")

            return resized_image, scaling_info

        return image, scaling_info

    def _should_convert_to_grayscale(self, image: Image.Image, document_type: str) -> bool:
        """
        Determine if image should be converted to grayscale.

        Args:
            image: PIL Image to analyze
            document_type: Type of document for context-aware decision

        Returns:
            True if grayscale conversion is recommended
        """
        # Skip if already grayscale
        if image.mode in ('L', '1'):
            return False

        # Preserve color for ID cards with photos (important for verification)
        if (document_type in ['id_card', 'aadhar', 'pan', 'passport'] and
            self.config['grayscale']['preserve_color_for_ids']):
            return False

        # Force grayscale for text-heavy documents
        if document_type in ['text', 'resume', 'document'] and self.config['grayscale']['force_for_text_docs']:
            return True

        # Auto-detect if grayscale is safe
        if self.config['grayscale']['auto_detect']:
            return self._is_text_heavy_content(image)

        return False

    def _convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """
        Convert image to 8-bit grayscale efficiently.

        Args:
            image: PIL Image to convert

        Returns:
            Grayscale PIL Image
        """
        if image.mode in ('L', '1'):
            return image

        # Use high-quality grayscale conversion
        grayscale_image = image.convert('L')

        return grayscale_image

    def _is_text_heavy_content(self, image: Image.Image) -> bool:
        """
        Analyze image to determine if it's text-heavy (safe for grayscale).

        Args:
            image: PIL Image to analyze

        Returns:
            True if image appears to be text-heavy content
        """
        # Sample-based analysis for performance
        sample_size = min(100, min(image.size) // 4)
        if sample_size < 10:
            return True  # Small images are likely text-heavy

        # Create a sample from center of image
        width, height = image.size
        left = (width - sample_size) // 2
        top = (height - sample_size) // 2
        sample = image.crop((left, top, left + sample_size, top + sample_size))

        # Convert to grayscale for analysis
        gray_sample = sample.convert('L')

        # Calculate statistics
        stat = ImageStat.Stat(gray_sample)

        # Text-heavy images typically have:
        # 1. High contrast (high standard deviation)
        # 2. Bimodal distribution (text vs background)
        std_dev = stat.stddev[0]
        mean = stat.mean[0]

        # Check for text-like characteristics
        has_high_contrast = std_dev > 30
        has_reasonable_brightness = 50 < mean < 200

        # Additional check: analyze edge density (text has many edges)
        edge_density = self._calculate_edge_density(gray_sample)
        has_text_like_edges = edge_density > 0.1

        is_text_heavy = has_high_contrast and has_reasonable_brightness and has_text_like_edges

        logger.debug(f"Text analysis: contrast={std_dev:.1f}, brightness={mean:.1f}, "
                    f"edges={edge_density:.3f}, text_heavy={is_text_heavy}")

        return is_text_heavy

    def _calculate_edge_density(self, gray_image: Image.Image) -> float:
        """
        Calculate edge density in grayscale image (simplified).

        Args:
            gray_image: Grayscale PIL Image

        Returns:
            Edge density ratio (0.0 to 1.0)
        """
        # Simple edge detection using neighboring pixel differences
        width, height = gray_image.size
        pixels = list(gray_image.getdata())

        edge_pixels = 0
        total_pixels = width * height

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                idx = y * width + x
                current = pixels[idx]

                # Check neighbors
                neighbors = [
                    pixels[idx - 1],      # left
                    pixels[idx + 1],      # right
                    pixels[idx - width],  # top
                    pixels[idx + width]   # bottom
                ]

                # Count significant differences (edges)
                for neighbor in neighbors:
                    if abs(current - neighbor) > 30:
                        edge_pixels += 1
                        break

        return edge_pixels / total_pixels

    def _calculate_optimal_dpi(self, page, document_type: str) -> int:
        """
        Calculate optimal DPI for PDF page extraction.

        Args:
            page: PyMuPDF page object
            document_type: Type of document being processed

        Returns:
            Optimal DPI value
        """
        # Get page dimensions
        rect = page.rect
        width_inches = rect.width / 72  # PDF points to inches
        height_inches = rect.height / 72

        # Base DPI recommendations by document type
        base_dpi = {
            'id_card': 250,      # Higher DPI for ID cards with photos
            'aadhar': 200,
            'pan': 200,
            'text': 150,         # Moderate for text documents
            'resume': 180,       # Slightly higher for resumes
            'general': 200       # Default for unknown content
        }

        target_dpi = base_dpi.get(document_type, 200)

        # Adjust based on page size (smaller pages may need higher DPI)
        if width_inches < 3 or height_inches < 3:
            target_dpi = int(target_dpi * 1.3)

        # Cap at reasonable maximum to avoid oversized images
        max_dpi = 300
        target_dpi = min(target_dpi, max_dpi)

        logger.debug(f"Calculated optimal DPI: {target_dpi} for {document_type} document "
                    f"({width_inches:.1f}x{height_inches:.1f} inches)")

        return target_dpi

    def _estimate_dpi_after_scaling(self, image: Image.Image, scale_factor: float) -> float:
        """
        Estimate the resulting DPI after scaling an image.

        Args:
            image: Original PIL Image
            scale_factor: Scaling factor applied

        Returns:
            Estimated DPI after scaling
        """
        # Estimate original DPI based on common document sizes
        width, height = image.size

        # Calculate aspect ratio for better document classification
        aspect_ratio = height / width if width > 0 else 1.0

        # Common document dimensions and their typical DPIs
        # Format: (width, height, dpi, description)
        document_specs = [
            # Standard documents
            (2480, 3508, 300, "A4 at 300 DPI"),
            (1654, 2339, 200, "A4 at 200 DPI"),
            (2550, 3300, 300, "US Letter at 300 DPI"),
            (1700, 2200, 200, "US Letter at 200 DPI"),

            # ID cards and small documents
            (600, 900, 200, "ID Card portrait"),
            (900, 600, 200, "ID Card landscape"),
            (1200, 800, 200, "Small document"),

            # Medium format documents
            (1200, 1800, 200, "Typical photo/document"),
            (1600, 2400, 200, "Medium document"),

            # Large format documents and high-res scans
            (2400, 3600, 300, "Large format document"),
            (3000, 4000, 300, "High-resolution scan"),
            (3120, 4160, 300, "Mobile photo high-res"),
            (3300, 5100, 300, "Very high-res scan"),
            (3600, 4800, 300, "Pro-grade scan"),
            (4000, 6000, 300, "Ultra high-res scan"),

            # Lower resolution documents
            (800, 1200, 150, "Lower resolution document"),
            (1000, 1500, 150, "Standard low-res"),
        ]

        estimated_dpi = 200  # Default assumption

        # Try exact size match first
        for doc_width, doc_height, doc_dpi, description in document_specs:
            if abs(width - doc_width) < 50 and abs(height - doc_height) < 50:
                estimated_dpi = doc_dpi
                logger.debug(f"Matched exact size: {description}")
                break

        # If no exact match, try aspect ratio + size range matching
        else:
            for doc_width, doc_height, doc_dpi, description in document_specs:
                doc_aspect = doc_height / doc_width

                # Check aspect ratio similarity (within 10%) and size range
                aspect_match = abs(aspect_ratio - doc_aspect) < 0.1
                size_range_match = (abs(width - doc_width) / doc_width < 0.2 and
                                  abs(height - doc_height) / doc_height < 0.2)

                if aspect_match and size_range_match:
                    estimated_dpi = doc_dpi
                    logger.debug(f"Matched by aspect/size: {description}")
                    break

            # If still no match, estimate based on size categories
            else:
                max_dimension = max(width, height)

                if max_dimension >= 4000:
                    estimated_dpi = 300  # Large images are likely high-res scans
                    logger.debug(f"Estimated DPI {estimated_dpi} for very large image ({max_dimension}px)")
                elif max_dimension >= 2500:
                    estimated_dpi = 250  # Medium-large images
                    logger.debug(f"Estimated DPI {estimated_dpi} for large image ({max_dimension}px)")
                elif max_dimension >= 1500:
                    estimated_dpi = 200  # Standard documents
                    logger.debug(f"Estimated DPI {estimated_dpi} for medium image ({max_dimension}px)")
                else:
                    estimated_dpi = 150  # Small images
                    logger.debug(f"Estimated DPI {estimated_dpi} for small image ({max_dimension}px)")

        # Calculate DPI after scaling
        scaled_dpi = estimated_dpi * scale_factor

        logger.debug(f"DPI estimation: original={estimated_dpi}, scale_factor={scale_factor:.3f}, scaled={scaled_dpi:.1f}")

        return scaled_dpi

    def _estimate_image_memory(self, image: Image.Image) -> int:
        """
        Estimate memory usage of PIL Image in bytes.

        Args:
            image: PIL Image

        Returns:
            Estimated memory usage in bytes
        """
        width, height = image.size

        # Bytes per pixel by mode
        bytes_per_pixel = {
            '1': 1,      # 1-bit black and white
            'L': 1,      # 8-bit grayscale
            'P': 1,      # 8-bit palette
            'RGB': 3,    # 8-bit RGB
            'RGBA': 4,   # 8-bit RGB with alpha
            'CMYK': 4,   # 8-bit CMYK
        }

        bpp = bytes_per_pixel.get(image.mode, 3)

        return width * height * bpp

    def get_optimization_stats(self) -> dict:
        """Get optimization performance statistics."""
        stats = self.optimization_stats.copy()

        if stats['images_processed'] > 0:
            stats['average_memory_saved_mb'] = stats['memory_saved_mb'] / stats['images_processed']
        else:
            stats['average_memory_saved_mb'] = 0

        return stats

    def reset_stats(self):
        """Reset optimization statistics."""
        self.optimization_stats = {
            'images_processed': 0,
            'memory_saved_mb': 0,
            'total_savings_percent': 0
        }