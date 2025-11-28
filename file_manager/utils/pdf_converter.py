"""
PDF to Image Conversion Utility

Converts PDF pages to images for OCR processing.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
import time

try:
    import fitz  # PyMuPDF
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from PIL import Image
    import io
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class PDFToImageConverter:
    """Convert PDF pages to images for OCR processing."""

    def __init__(self, dpi: int = 300, image_format: str = 'PNG'):
        """
        Initialize the PDF converter.

        Args:
            dpi: Resolution for image conversion (higher = better quality, larger files)
            image_format: Output image format ('PNG', 'JPEG', etc.)
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for PDF to image conversion")

        self.dpi = dpi
        self.image_format = image_format
        logger.info(f"PDF to Image converter initialized: {dpi} DPI, {image_format} format")

    def convert_pdf_to_images(self, pdf_path: Path,
                            temp_dir: Optional[Path] = None,
                            page_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, Path]]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file
            temp_dir: Directory for temporary images (if None, uses system temp)
            page_range: Optional tuple (start_page, end_page) for partial conversion

        Returns:
            List of tuples (page_number, image_path)
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create temp directory if not provided
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="patxtract_ocr_"))
        else:
            temp_dir.mkdir(exist_ok=True)

        logger.info(f"Converting PDF to images: {pdf_path}")
        start_time = time.time()

        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                logger.info(f"PDF has {total_pages} pages")

                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(1, start_page)
                    end_page = min(total_pages, end_page)
                else:
                    start_page, end_page = 1, total_pages

                page_images = []

                for page_num in range(start_page, end_page + 1):
                    try:
                        # Get page
                        page = doc[page_num - 1]

                        # Render page to pixmap
                        pix = page.get_pixmap(dpi=self.dpi)

                        # Convert to PIL Image
                        img_data = pix.tobytes(self.image_format.lower())
                        img = Image.open(io.BytesIO(img_data))

                        # Save image
                        image_filename = f"page_{page_num:04d}.{self.image_format.lower()}"
                        image_path = temp_dir / image_filename
                        img.save(image_path, self.image_format)

                        page_images.append((page_num, image_path))
                        logger.debug(f"Page {page_num} converted to {image_path}")

                    except Exception as e:
                        logger.error(f"Failed to convert page {page_num}: {e}")
                        continue

                conversion_time = time.time() - start_time
                logger.info(f"Converted {len(page_images)} pages to images in {conversion_time:.2f}s")

                return page_images

        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise

    def convert_single_page(self, pdf_path: Path, page_num: int,
                          temp_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Convert a single PDF page to image.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-based)
            temp_dir: Directory for temporary images

        Returns:
            Path to image file or None if failed
        """
        try:
            images = self.convert_pdf_to_images(pdf_path, temp_dir, (page_num, page_num))
            if images:
                return images[0][1]  # Return image path from first (and only) tuple
            return None
        except Exception as e:
            logger.error(f"Failed to convert page {page_num}: {e}")
            return None

    def cleanup_temp_images(self, temp_dir: Path):
        """
        Clean up temporary image files.

        Args:
            temp_dir: Directory containing temporary images
        """
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")


def convert_pdf_for_ocr(pdf_path: Path, page_num: int = 1, dpi: int = 300) -> Optional[Path]:
    """
    Convenience function to convert a single PDF page for OCR.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number to convert (1-based)
        dpi: Resolution for conversion

    Returns:
        Path to converted image file or None if failed
    """
    try:
        converter = PDFToImageConverter(dpi=dpi)
        return converter.convert_single_page(pdf_path, page_num)
    except Exception as e:
        logger.error(f"Failed to convert PDF page {page_num} for OCR: {e}")
        return None