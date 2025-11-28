"""
PDF OCR Processor

Handles OCR extraction from PDF files by converting pages to images first.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from file_manager.utils.pdf_converter import PDFToImageConverter

logger = logging.getLogger(__name__)


class PdfOcrProcessor:
    """
    OCR processor that handles PDF files by converting them to images first.
    """

    def __init__(self, ocr_engine, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PDF OCR processor.

        Args:
            ocr_engine: The actual OCR engine (PaddleOCR or Tesseract)
            config: Configuration dictionary
        """
        self.ocr_engine = ocr_engine
        self.config = config or {}

        # Configure PDF to image conversion
        self.pdf_dpi = self.config.get('pdf_dpi', 300)
        self.image_format = self.config.get('image_format', 'PNG')
        self.max_pages = self.config.get('max_pages', 1000)  # Support for very large documents (500+ pages)

        self.pdf_converter = PDFToImageConverter(
            dpi=self.pdf_dpi,
            image_format=self.image_format
        )

        logger.info(f"PDF OCR processor initialized with {self.pdf_dpi} DPI")

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text from PDF using OCR.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing extracted text and metadata
        """
        start_time = time.time()
        temp_dir = None

        try:
            if not pdf_path.exists():
                return self._create_error_result(str(pdf_path), "File not found")

            logger.info(f"Starting OCR extraction from PDF: {pdf_path}")

            result = {
                "success": False,
                "text": "",
                "pages": [],
                "total_pages": 0,
                "total_characters": 0,
                "confidence": 0.0,
                "processing_time": 0,
                "errors": [],
                "extraction_method": "ocr"
            }

            # Process pages one at a time to reduce memory usage
            with tempfile.TemporaryDirectory(prefix="patxtract_pdf_ocr_") as temp_dir_str:
                temp_dir = Path(temp_dir_str)

                # First, determine the total number of pages
                try:
                    import fitz
                    with fitz.open(pdf_path) as doc:
                        total_pages_in_pdf = min(len(doc), self.max_pages)
                except Exception as e:
                    result["errors"].append(f"Failed to read PDF page count: {e}")
                    return result

                print(f"📄 PDF contains {total_pages_in_pdf} pages (processing limit: {self.max_pages})")
                logger.info(f"PDF contains {total_pages_in_pdf} pages (max limit: {self.max_pages})")
                result["total_pages"] = total_pages_in_pdf

                # Process each page one at a time (convert -> OCR -> cleanup)
                print(f"🔄 Starting OCR processing of {total_pages_in_pdf} pages...")
                logger.info(f"Starting OCR processing of {total_pages_in_pdf} pages...")

                start_processing_time = time.time()

                for page_num in range(1, total_pages_in_pdf + 1):
                    try:
                        print(f"📖 [Page {page_num}/{total_pages_in_pdf}] Converting to image...", end=" ", flush=True)
                        logger.info(f"Processing page {page_num}/{total_pages_in_pdf}")

                        # Convert single page to image (one at a time)
                        image_path = self.pdf_converter.convert_single_page(pdf_path, page_num, temp_dir)

                        if not image_path or not image_path.exists():
                            print("❌ Failed")
                            logger.warning(f"Failed to convert page {page_num} to image")
                            result["pages"].append({
                                "page_number": page_num,
                                "success": False,
                                "text": "",
                                "text_length": 0,
                                "confidence": 0.0,
                                "processing_time": 0.0,
                                "error": "Failed to convert page to image"
                            })
                            continue

                        print("✅ ", end="", flush=True)
                        logger.debug(f"Converted page {page_num} to image: {image_path}")

                        print("Extracting text...", end=" ", flush=True)
                        # Use the OCR engine to extract text from image
                        page_result = self.ocr_engine.extract_text(image_path)

                        # Clean up the image file immediately after OCR to save memory
                        try:
                            image_path.unlink()
                            logger.debug(f"Cleaned up image file: {image_path}")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup image file {image_path}: {cleanup_error}")

                        if page_result.get("success"):
                            # Store page results (note: image_path already cleaned up)
                            page_data = {
                                "page_number": page_num,
                                "success": True,
                                "text": page_result.get("text", ""),
                                "text_length": len(page_result.get("text", "")),
                                "confidence": page_result.get("confidence", 0.0),
                                "processing_time": page_result.get("processing_time", 0.0)
                            }
                            result["pages"].append(page_data)
                            result["total_characters"] += page_data["text_length"]

                            # Calculate progress percentage and show detailed feedback
                            progress_percent = (page_num / total_pages_in_pdf) * 100
                            chars_extracted = page_data['text_length']
                            confidence_score = page_data['confidence']

                            print(f"✅ {chars_extracted} chars ({confidence_score:.1f}% confidence) [{progress_percent:.1f}%]")
                            logger.info(f"✅ Page {page_num}/{total_pages_in_pdf} completed: {page_data['text_length']} chars extracted ({page_data['confidence']:.1f}% confidence)")
                        else:
                            error_msg = page_result.get("error", "Unknown OCR error")
                            progress_percent = (page_num / total_pages_in_pdf) * 100
                            print(f"❌ Failed [{progress_percent:.1f}%] - {error_msg}")
                            logger.warning(f"❌ Page {page_num}/{total_pages_in_pdf} failed: {error_msg}")

                            result["pages"].append({
                                "page_number": page_num,
                                "success": False,
                                "text": "",
                                "text_length": 0,
                                "confidence": 0.0,
                                "errors": [error_msg]
                            })
                            result["errors"].append(f"Page {page_num}: {error_msg}")

                    except Exception as e:
                        error_msg = f"Error processing page {page_num}: {str(e)}"
                        logger.error(error_msg)

                        result["pages"].append({
                            "page_number": page_num,
                            "success": False,
                            "text": "",
                            "text_length": 0,
                            "confidence": 0.0,
                            "errors": [error_msg]
                        })
                        result["errors"].append(error_msg)

                # Processing completed - show summary
                total_processing_time = time.time() - start_processing_time
                successful_pages = sum(1 for page in result["pages"] if page.get("success", False))
                avg_time_per_page = total_processing_time / total_pages_in_pdf if total_pages_in_pdf > 0 else 0

                print(f"\n🎉 OCR Processing Complete!")
                print(f"📊 Summary: {successful_pages}/{total_pages_in_pdf} pages successful")
                print(f"📝 Total characters extracted: {result['total_characters']:,}")
                print(f"⏱️  Total time: {total_processing_time:.2f} seconds")
                print(f"⚡ Average per page: {avg_time_per_page:.2f} seconds")

                # Calculate overall results
                if result["total_characters"] > 0:
                    result["success"] = True

                    # Combine all text
                    all_text = []
                    for page in result["pages"]:
                        if page.get("success") and page.get("text"):
                            all_text.append(f"--- Page {page['page_number']} ---")
                            all_text.append(page["text"])
                    result["text"] = "\n\n".join(all_text)

                    # Calculate average confidence
                    successful_pages = [p for p in result["pages"] if p.get("success")]
                    if successful_pages:
                        avg_confidence = sum(p.get("confidence", 0) for p in successful_pages) / len(successful_pages)
                        result["confidence"] = avg_confidence

                # Processing time
                result["processing_time"] = time.time() - start_time

                logger.info(f"PDF OCR extraction completed: "
                          f"{result['total_characters']} chars, "
                          f"{result['confidence']:.1f}% confidence, "
                          f"{result['processing_time']:.2f}s")

                return result

        except Exception as e:
            error_msg = f"PDF OCR extraction failed: {str(e)}"
            logger.error(error_msg)
            return self._create_error_result(str(pdf_path), error_msg, time.time() - start_time)

    def _create_error_result(self, file_path: str, error_msg: str, processing_time: float = 0.0) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            "success": False,
            "text": "",
            "pages": [],
            "total_pages": 0,
            "total_characters": 0,
            "confidence": 0.0,
            "processing_time": processing_time,
            "errors": [error_msg],
            "file_path": file_path,
            "extraction_method": "ocr"
        }


class EnhancedPaddleOCR:
    """
    Enhanced PaddleOCR wrapper that can handle both images and PDFs.
    """

    def __init__(self, ocr_extractor, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced PaddleOCR.

        Args:
            ocr_extractor: Base PaddleOCR extractor
            config: Configuration dictionary
        """
        self.ocr_extractor = ocr_extractor
        self.pdf_processor = PdfOcrProcessor(ocr_extractor, config)
        logger.info("Enhanced PaddleOCR initialized with PDF support")

    def extract_text(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from file (image or PDF).

        Args:
            file_path: Path to file (image or PDF)

        Returns:
            Dictionary containing extracted text and metadata
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.pdf':
            # Handle PDF files
            logger.info(f"Processing PDF file with enhanced OCR: {file_path}")
            return self.pdf_processor.extract_text_from_pdf(file_path)
        else:
            # Handle image files with original extractor
            logger.debug(f"Processing image file with original OCR: {file_path}")
            return self.ocr_extractor.extract_text(file_path)