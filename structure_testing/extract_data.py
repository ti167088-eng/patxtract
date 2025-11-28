"""
Data Extraction Utility

Extracts text data from PDF files using the existing FileManager OCR system
and saves it to a reusable JSON file. This allows multiple models to use
the same OCR data without reprocessing the PDF multiple times.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path to import file_manager
sys.path.append(str(Path(__file__).parent.parent))

from file_manager import FileManager

logger = logging.getLogger(__name__)

class DataExtractor:
    """
    Utility class for extracting and caching OCR data from PDF files.
    """

    def __init__(self, cache_dir: str = "extracted_data"):
        """
        Initialize the data extractor.

        Args:
            cache_dir: Directory to store extracted data files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        # Don't initialize FileManager here - only load it when needed for extraction
        self.file_manager = None
        self.logger = logging.getLogger(__name__)

    def extract_pdf_data(self, pdf_path: Path, force_reextract: bool = False) -> Path:
        """
        Extract OCR data from PDF and save to cache.

        Args:
            pdf_path: Path to the PDF file
            force_reextract: If True, re-extract even if cached data exists

        Returns:
            Path to the cached extracted data file
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError("File must be a PDF")

        # Generate cache filename
        cache_filename = self._get_cache_filename(pdf_path)
        cache_path = self.cache_dir / cache_filename

        # Check if cached data exists and we're not forcing re-extraction
        if cache_path.exists() and not force_reextract:
            self.logger.info(f"Using cached data from: {cache_path}")
            print(f"✅ Using cached OCR data from previous extraction")
            return cache_path

        print(f"🔍 Extracting OCR data from: {pdf_path.name}")
        print("⏳ This may take a moment...")

        try:
            # Initialize FileManager only when needed for extraction
            if self.file_manager is None:
                from file_manager import FileManager
                self.file_manager = FileManager()
                print("[OCR] Initializing OCR system...")

            # Extract data using FileManager
            extraction_result = self.file_manager.process_pdf(pdf_path, method="ocr")

            if not extraction_result.get("success", False):
                errors = extraction_result.get("errors", ["Unknown error"])
                raise Exception(f"Failed to extract data: {'; '.join(errors)}")

            # Prepare extracted data structure
            extracted_data = {
                "extraction_info": {
                    "pdf_path": str(pdf_path),
                    "pdf_name": pdf_path.name,
                    "extraction_timestamp": datetime.now().isoformat(),
                    "extraction_method": extraction_result.get("extraction", {}).get("method", "ocr"),
                    "total_pages": extraction_result.get("extraction", {}).get("total_pages", 0),
                    "total_characters": extraction_result.get("extraction", {}).get("total_characters", 0),
                    "overall_confidence": extraction_result.get("extraction", {}).get("confidence", 0.0),
                    "processing_time": extraction_result.get("processing_time", 0.0)
                },
                "pages": []
            }

            # Process each page
            pages_data = extraction_result.get("extraction", {}).get("pages", [])
            successful_pages = 0

            for page_data in pages_data:
                page_info = {
                    "page_number": page_data.get("page_number", 0),
                    "text": page_data.get("text", ""),
                    "text_length": page_data.get("text_length", 0),
                    "confidence": page_data.get("confidence", 0.0),
                    "processing_time": page_data.get("processing_time", 0.0),
                    "image_path": page_data.get("image_path", ""),
                    "success": page_data.get("success", False)
                }

                extracted_data["pages"].append(page_info)

                if page_info["success"]:
                    successful_pages += 1

            # Add summary statistics
            extracted_data["extraction_info"]["successful_pages"] = successful_pages
            extracted_data["extraction_info"]["extraction_success_rate"] = (
                successful_pages / len(pages_data) if pages_data else 0.0
            )

            # Save extracted data to cache
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)

            print(f"✅ OCR extraction completed successfully!")
            print(f"📄 Processed {successful_pages}/{len(pages_data)} pages")
            print(f"⏱️  Processing time: {extraction_result.get('processing_time', 0):.2f} seconds")
            print(f"💾 Data cached to: {cache_path}")

            self.logger.info(f"Successfully extracted and cached data to: {cache_path}")
            return cache_path

        except Exception as e:
            self.logger.error(f"Error extracting data from {pdf_path}: {e}")
            print(f"❌ Extraction failed: {e}")
            raise

    def load_extracted_data(self, data_path: Path) -> dict:
        """
        Load previously extracted data from cache.

        Args:
            data_path: Path to the cached data file

        Returns:
            Dictionary containing the extracted data
        """
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Extracted data file not found: {data_path}")

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.info(f"Loaded extracted data from: {data_path}")
            return data

        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in extracted data file: {e}")
        except Exception as e:
            raise Exception(f"Error loading extracted data: {e}")

    def get_cached_data_path(self, pdf_path: Path) -> Path:
        """
        Get the path to cached data for a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Path to the cached data file (may not exist)
        """
        cache_filename = self._get_cache_filename(pdf_path)
        return self.cache_dir / cache_filename

    def _get_cache_filename(self, pdf_path: Path) -> str:
        """
        Generate a consistent cache filename for a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Cache filename string
        """
        # Use the PDF name and modification time to create a unique identifier
        pdf_stat = pdf_path.stat()
        timestamp = int(pdf_stat.st_mtime)
        pdf_name = pdf_path.stem

        return f"{pdf_name}_{timestamp}.json"

    def list_cached_files(self) -> list:
        """
        List all cached extraction files.

        Returns:
            List of cache file information
        """
        cached_files = []

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                extraction_info = data.get("extraction_info", {})
                cached_files.append({
                    "cache_file": cache_file.name,
                    "pdf_name": extraction_info.get("pdf_name", "Unknown"),
                    "pdf_path": extraction_info.get("pdf_path", "Unknown"),
                    "extraction_timestamp": extraction_info.get("extraction_timestamp", "Unknown"),
                    "total_pages": extraction_info.get("total_pages", 0),
                    "successful_pages": extraction_info.get("successful_pages", 0),
                    "overall_confidence": extraction_info.get("overall_confidence", 0.0)
                })
            except Exception as e:
                self.logger.warning(f"Error reading cache file {cache_file}: {e}")

        return cached_files

    def clear_cache(self, pdf_name: str = None) -> int:
        """
        Clear cached extraction files.

        Args:
            pdf_name: If provided, only clear cache for specific PDF name

        Returns:
            Number of files deleted
        """
        deleted_count = 0

        if pdf_name:
            # Clear specific PDF cache
            pattern = f"{pdf_name}_*.json"
            for cache_file in self.cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted cache file: {cache_file}")
                except Exception as e:
                    self.logger.error(f"Error deleting cache file {cache_file}: {e}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted cache file: {cache_file}")
                except Exception as e:
                    self.logger.error(f"Error deleting cache file {cache_file}: {e}")

        return deleted_count

    def print_cache_info(self):
        """Print information about cached files."""
        cached_files = self.list_cached_files()

        if not cached_files:
            print("📂 No cached extraction files found.")
            return

        print(f"\n📂 Found {len(cached_files)} cached extraction files:")
        print("-" * 80)

        for file_info in cached_files:
            success_rate = (file_info["successful_pages"] / file_info["total_pages"] * 100) if file_info["total_pages"] > 0 else 0
            print(f"📄 {file_info['pdf_name']}")
            print(f"   Path: {file_info['pdf_path']}")
            print(f"   Pages: {file_info['successful_pages']}/{file_info['total_pages']} ({success_rate:.1f}%)")
            print(f"   Confidence: {file_info['overall_confidence']:.1%}")
            print(f"   Extracted: {file_info['extraction_timestamp']}")
            print("-" * 80)


def main():
    """
    Main function for interactive data extraction.
    """
    extractor = DataExtractor()

    print("=== PatXtract Data Extraction Utility ===")
    print("Extract OCR data from PDF files for reuse in model testing.")

    while True:
        print("\nOptions:")
        print("1. Extract data from PDF")
        print("2. List cached files")
        print("3. Clear cache")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            pdf_path = input("Enter path to PDF file: ").strip()
            if not pdf_path:
                print("Please provide a valid PDF path.")
                continue

            try:
                cache_path = extractor.extract_pdf_data(Path(pdf_path))
                print(f"\n✅ Data extracted and cached successfully!")
                print(f"Cache file: {cache_path.name}")
            except Exception as e:
                print(f"❌ Error: {e}")

        elif choice == "2":
            extractor.print_cache_info()

        elif choice == "3":
            pdf_name = input("Enter PDF name to clear (leave empty to clear all): ").strip()
            try:
                deleted_count = extractor.clear_cache(pdf_name if pdf_name else None)
                if deleted_count > 0:
                    print(f"✅ Deleted {deleted_count} cache file(s).")
                else:
                    print("No cache files found to delete.")
            except Exception as e:
                print(f"❌ Error: {e}")

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()