"""
PatXtract File Manager Test

Interactive testing interface for the File Manager component.
Provides user-friendly prompts for testing PDF processing capabilities.
"""

import sys
import json
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from file_manager.manager.manager import FileManager
from file_manager.utils.path_utils import normalize_path_input, get_example_pdf_path


class FileManagerTest:
    """Interactive test interface for File Manager."""

    def __init__(self):
        """Initialize the test interface."""
        self.file_manager = FileManager()
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)

    def run_interactive_test(self):
        """Run interactive test with user prompts."""
        print("=" * 60)
        print("🏥 PatXtract File Manager - Interactive Test")
        print("=" * 60)
        print()

        # Show supported methods
        self._show_supported_methods()
        print()

        # Get PDF path from user
        pdf_path = self._get_pdf_path_from_user()
        if not pdf_path:
            print("❌ No valid PDF file provided. Exiting.")
            return

        # Get processing options
        method = self._get_processing_method()
        include_analysis = self._get_analysis_preference()
        save_results = self._get_save_results_preference()

        print()
        print("🚀 Starting PDF processing...")
        print("-" * 40)

        # Process the PDF
        try:
            result = self.file_manager.process_pdf(
                pdf_path=pdf_path,
                method=method,
                include_analysis=include_analysis
            )

            # Display results
            self._display_results(result)

            # Save results if requested
            if save_results and result.get("success"):
                self._save_results(result)

            # Show processing summary
            self._show_processing_summary()

        except Exception as e:
            print(f"❌ Unexpected error: {e}")

        print()
        print("✅ Test completed!")

    def _show_supported_methods(self):
        """Display supported extraction methods."""
        methods = self.file_manager.get_supported_methods()
        print("📋 Supported Extraction Methods:")
        print("   • Scanned Text: ✅ Available" if methods.get("scanned") else "   • Scanned Text: ❌ Not Available")
        print("   • OCR: ✅ Available" if methods.get("ocr") else "   • OCR: ❌ Not Available")
        if methods.get("paddleocr"):
            print("   • PaddleOCR: ✅ Available")
        if methods.get("tesseract"):
            print("   • Tesseract: ✅ Available")

    def _get_pdf_path_from_user(self) -> Optional[Path]:
        """Get PDF file path from user input."""
        print(f"💡 Example path format: {get_example_pdf_path()}")
        while True:
            pdf_path_str = input("📁 Enter the path to your PDF file (or 'quit' to exit): ").strip()

            if pdf_path_str.lower() in ['quit', 'exit', 'q']:
                return None

            try:
                pdf_path = normalize_path_input(pdf_path_str)

                if not pdf_path.exists():
                    print(f"❌ File not found: {pdf_path}")
                    continue

                if pdf_path.suffix.lower() != '.pdf':
                    print(f"❌ File is not a PDF: {pdf_path}")
                    continue

                return pdf_path

            except ValueError as e:
                print(f"❌ Invalid path: {e}")
                continue

    def _get_processing_method(self) -> Optional[str]:
        """Get processing method preference from user."""
        print("\n🔧 Processing Method:")
        print("   1. Auto-detect (Recommended)")
        print("   2. Scanned Text Extraction (Faster)")
        print("   3. OCR Extraction")

        while True:
            choice = input("   Choose method (1-3): ").strip()

            if choice == '1':
                return None  # Auto-detect
            elif choice == '2':
                return 'scanned'
            elif choice == '3':
                return 'ocr'
            else:
                print("   ❌ Invalid choice. Please enter 1, 2, or 3.")

    def _get_analysis_preference(self) -> bool:
        """Get analysis preference from user."""
        print("\n📊 Document Analysis:")
        while True:
            choice = input("   Include document analysis? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("   ❌ Please enter 'y' or 'n'.")

    def _get_save_results_preference(self) -> bool:
        """Get save results preference from user."""
        print("\n💾 Save Results:")
        while True:
            choice = input("   Save results to JSON file? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("   ❌ Please enter 'y' or 'n'.")

    def _display_results(self, result: dict):
        """Display processing results in formatted way."""
        print("\n📊 Processing Results:")
        print("=" * 50)

        if result.get("success"):
            # Basic info
            print(f"✅ Status: Success")
            print(f"📁 File: {result.get('file_name', 'unknown')}")
            print(f"⏱️  Processing Time: {result.get('processing_time', 0):.2f} seconds")

            # Analysis info
            if result.get("analysis"):
                analysis = result["analysis"]
                print(f"\n🔍 Analysis Results:")
                print(f"   • Recommended Method: {analysis.get('recommended_method', 'unknown')}")
                print(f"   • Confidence: {analysis.get('confidence', 0):.1%}")
                print(f"   • Has Selectable Text: {'Yes' if analysis.get('has_selectable_text') else 'No'}")
                print(f"   • Total Pages: {analysis.get('total_pages', 0)}")
                print(f"   • Reason: {analysis.get('reason', 'N/A')}")

            # Extraction info
            if result.get("extraction"):
                extraction = result["extraction"]
                print(f"\n📝 Extraction Results:")
                print(f"   • Method Used: {extraction.get('extraction_method', 'unknown')}")
                print(f"   • Pages Processed: {extraction.get('total_pages', 0)}")
                print(f"   • Total Characters: {extraction.get('total_characters', 0):,}")

                # Show text preview
                if extraction.get("pages"):
                    first_page = extraction["pages"][0]
                    if first_page.get("text"):
                        preview = first_page["text"][:200] + "..." if len(first_page["text"]) > 200 else first_page["text"]
                        print(f"   • Text Preview (Page 1):")
                        print(f"     {repr(preview)}")

            # Metadata info
            if result.get("metadata"):
                metadata = result["metadata"]
                print(f"\n📋 Metadata:")
                print(f"   • File Size: {metadata.get('file_info', {}).get('size_mb', 0):.2f} MB")

                perf = metadata.get("performance", {})
                if perf:
                    print(f"   • Processing Speed: {perf.get('characters_per_second', 0):,.0f} chars/sec")
                    print(f"   • Efficiency: {perf.get('processing_efficiency', 'unknown')}")

        else:
            print(f"❌ Status: Failed")
            print(f"📁 File: {result.get('file_name', 'unknown')}")
            print(f"⏱️  Processing Time: {result.get('processing_time', 0):.2f} seconds")

            if result.get("errors"):
                print(f"\n❌ Errors:")
                for error in result["errors"]:
                    print(f"   • {error}")

    def _save_results(self, result: dict):
        """Save results to JSON file."""
        try:
            # Generate filename
            file_name = result.get('file_name', 'unknown').replace('.pdf', '')
            timestamp = result.get('processing_timestamp', '').replace(':', '-')[:19]
            output_filename = f"{file_name}_{timestamp}.json"
            output_path = self.output_dir / output_filename

            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)

            print(f"\n💾 Results saved to: {output_path}")

        except Exception as e:
            print(f"\n❌ Error saving results: {e}")

    def _show_processing_summary(self):
        """Show overall processing statistics."""
        stats = self.file_manager.get_processing_stats()
        print(f"\n📈 Processing Statistics:")
        print(f"   • Total Documents Processed: {stats['total_documents']}")
        print(f"   • Successful Extractions: {stats['successful_extractions']}")
        print(f"   • Failed Extractions: {stats['failed_extractions']}")
        print(f"   • Total Processing Time: {stats['total_processing_time']:.2f} seconds")

        if stats['total_documents'] > 0:
            success_rate = (stats['successful_extractions'] / stats['total_documents']) * 100
            print(f"   • Success Rate: {success_rate:.1f}%")

    def run_batch_test(self):
        """Run batch test with multiple PDFs."""
        print("\n📂 Batch Processing Test")
        print("-" * 30)

        # Get directory from user
        while True:
            dir_path_str = input("Enter directory path containing PDFs (or 'back' for main menu): ").strip()

            if dir_path_str.lower() == 'back':
                return

            dir_path = Path(dir_path_str)
            if not dir_path.exists() or not dir_path.is_dir():
                print(f"❌ Directory not found: {dir_path}")
                continue

            # Find PDF files
            pdf_files = list(dir_path.glob("*.pdf"))
            if not pdf_files:
                print(f"❌ No PDF files found in: {dir_path}")
                continue

            break

        print(f"\n📄 Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files[:5]:  # Show first 5
            print(f"   • {pdf.name}")
        if len(pdf_files) > 5:
            print(f"   ... and {len(pdf_files) - 5} more files")

        # Get processing options
        method = self._get_processing_method()
        include_analysis = self._get_analysis_preference()
        save_results = self._get_save_results_preference()

        print(f"\n🚀 Starting batch processing of {len(pdf_files)} files...")
        print("-" * 50)

        # Process batch
        try:
            batch_result = self.file_manager.process_multiple_pdfs(
                pdf_paths=pdf_files,
                method=method,
                include_analysis=include_analysis
            )

            # Display batch results
            print(f"\n📊 Batch Processing Results:")
            print(f"✅ Successful: {batch_result['successful_files']}")
            print(f"❌ Failed: {batch_result['failed_files']}")
            print(f"⏱️  Total Time: {batch_result['total_processing_time']:.2f} seconds")
            print(f"📝 Total Characters: {batch_result['summary']['total_characters_extracted']:,}")
            print(f"📄 Total Pages: {batch_result['summary']['total_pages_processed']}")

            # Save batch results if requested
            if save_results:
                batch_output_path = self.output_dir / f"batch_results_{batch_result['batch_id']}.json"
                self.file_manager.export_results(batch_result, batch_output_path)
                print(f"💾 Batch results saved to: {batch_output_path}")

        except Exception as e:
            print(f"❌ Batch processing error: {e}")


def main():
    """Main function for interactive testing."""
    tester = FileManagerTest()

    while True:
        print("\n" + "=" * 50)
        print("🏥 PatXtract File Manager Test Menu")
        print("=" * 50)
        print("1. Process Single PDF")
        print("2. Process Multiple PDFs (Batch)")
        print("3. Show Processing Statistics")
        print("4. Reset Statistics")
        print("5. Exit")
        print("-" * 50)

        choice = input("Enter your choice (1-5): ").strip()

        if choice == '1':
            tester.run_interactive_test()
        elif choice == '2':
            tester.run_batch_test()
        elif choice == '3':
            tester._show_processing_summary()
        elif choice == '4':
            tester.file_manager.reset_stats()
            print("✅ Statistics reset successfully!")
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()