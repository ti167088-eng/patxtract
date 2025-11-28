"""
PatXtract Extraction Method Test

Tests and compares different extraction methods (scanned vs OCR)
to validate routing decisions and performance characteristics.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from file_manager.manager.manager import FileManager
from file_manager.utils.path_utils import normalize_path_input, get_example_pdf_path


class ExtractionTest:
    """Test PDF extraction methods and display results."""

    def __init__(self):
        """Initialize the test interface."""
        self.file_manager = FileManager()
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)

    def run_comparison_test(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Run comparison test using both scanned and OCR methods.

        Args:
            pdf_path: Path to PDF file for testing

        Returns:
            Comparison results dictionary
        """
        if not pdf_path.exists():
            return {"error": f"File not found: {pdf_path}"}

        print(f"🔬 Running extraction comparison test on: {pdf_path.name}")
        print("-" * 60)

        results = {
            "pdf_path": str(pdf_path),
            "extraction_results": {}
        }

        # Test both methods
        methods_to_test = ["scanned", "ocr"]
        supported_methods = self.file_manager.get_supported_methods()

        for method in methods_to_test:
            print(f"\n🔧 Testing {method.upper()} method")

            if not supported_methods.get(method, False):
                print(f"   ❌ {method.upper()} method not available")
                continue

            try:
                print(f"      🔄 Starting {method.upper()} extraction...")
                start_time = time.time()
                extraction_result = self.file_manager.extraction_router.extract_text(pdf_path, method)
                processing_time = time.time() - start_time

                # Store only essential data
                results["extraction_results"][method] = {
                    "method": method,
                    "pages": extraction_result.get("pages", []),
                    "total_characters": extraction_result.get("total_characters", 0),
                    "success": extraction_result.get("success", False)
                }

                if extraction_result.get("success"):
                    print(f"   ✅ {method.upper()} successful: {extraction_result.get('total_characters', 0):,} chars in {processing_time:.2f}s")
                else:
                    print(f"   ❌ {method.upper()} failed")

            except Exception as e:
                print(f"   ❌ {method.upper()} error: {e}")
                results["extraction_results"][method] = {
                    "method": method,
                    "pages": [],
                    "total_characters": 0,
                    "success": False
                }

        return results

    def _get_text_preview(self, extraction_result: Dict[str, Any]) -> str:
        """Extract text preview from extraction result."""
        if not extraction_result.get("success"):
            return "Extraction failed"

        # Try to get text from first page
        if extraction_result.get("pages") and len(extraction_result["pages"]) > 0:
            first_page = extraction_result["pages"][0]
            text = first_page.get("text", "")
            if text:
                return text[:200] + "..." if len(text) > 200 else text

        # Fallback to other text fields
        if extraction_result.get("full_text"):
            text = extraction_result["full_text"]
            return text[:200] + "..." if len(text) > 200 else text

        return "No text extracted"

    def _get_full_page_text(self, pages_data: List[Dict[str, Any]]) -> str:
        """Get full text from all pages."""
        if not pages_data:
            return ""

        full_text = []
        for page in pages_data:
            if page.get("success") and page.get("text"):
                full_text.append(f"--- Page {page['page_number']} ---")
                full_text.append(page["text"])

        return "\n\n".join(full_text) if full_text else ""

    def _get_text_preview_from_pages(self, pages: List[Dict[str, Any]]) -> str:
        """Get text preview from pages data."""
        if not pages:
            return ""

        # Get text from first successful page
        for page in pages:
            if page.get("success") and page.get("text"):
                text = page["text"]
                return text[:200] + "..." if len(text) > 200 else text

        return ""

    def _compare_methods(self, methods: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different extraction methods."""
        comparison = {
            "successful_methods": [],
            "failed_methods": [],
            "best_performance": None,
            "most_accurate": None,
            "speed_comparison": {},
            "character_count_comparison": {},
            "similarities": {}
        }

        successful = {}
        for method, result in methods.items():
            if result.get("success"):
                successful[method] = result
                comparison["successful_methods"].append(method)
            else:
                comparison["failed_methods"].append(method)

        if len(successful) >= 2:
            # Find best performance (fastest)
            best_method = min(successful.keys(), key=lambda m: successful[m]["processing_time"])
            comparison["best_performance"] = {
                "method": best_method,
                "time": successful[best_method]["processing_time"],
                "chars_per_second": successful[best_method]["total_characters"] / successful[best_method]["processing_time"]
            }

            # Find most accurate (most characters extracted)
            most_accurate = max(successful.keys(), key=lambda m: successful[m]["total_characters"])
            comparison["most_accurate"] = {
                "method": most_accurate,
                "characters": successful[most_accurate]["total_characters"]
            }

            # Speed comparison
            fastest_time = min(successful[m]["processing_time"] for m in successful)
            for method in successful:
                time_ratio = successful[method]["processing_time"] / fastest_time
                comparison["speed_comparison"][method] = {
                    "time": successful[method]["processing_time"],
                    "relative_speed": f"{time_ratio:.1f}x"
                }

            # Character count comparison
            for method in successful:
                comparison["character_count_comparison"][method] = successful[method]["total_characters"]

            # Text similarity (basic comparison)
            if len(successful) == 2:
                methods_list = list(successful.keys())
                # Get text from first page for comparison
                text1 = self._get_text_preview_from_pages(successful[methods_list[0]].get("pages", []))
                text2 = self._get_text_preview_from_pages(successful[methods_list[1]].get("pages", []))
                similarity = self._calculate_text_similarity(text1, text2)
                comparison["similarities"] = {
                    "similarity_score": similarity,
                    "similarity_percent": f"{similarity * 100:.1f}%"
                }

        return comparison

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using common words."""
        if not text1 or not text2:
            return 0.0

        # Simple similarity based on common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _generate_recommendation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing recommendation based on test results."""
        recommendation = {
            "recommended_method": None,
            "confidence": 0.0,
            "reason": "",
            "alternatives": [],
            "notes": []
        }

        # Check analysis recommendation first
        analysis = results.get("analysis")
        if analysis and analysis.get("recommended_method"):
            analysis_method = analysis["recommended_method"]
            analysis_confidence = analysis.get("confidence", 0.0)

            # Check if analysis method worked well
            methods = results.get("methods", {})
            if methods.get(analysis_method, {}).get("success"):
                recommendation["recommended_method"] = analysis_method
                recommendation["confidence"] = analysis_confidence
                recommendation["reason"] = f"Analysis recommended {analysis_method} with {analysis_confidence:.1%} confidence"

                # Add alternatives if available
                for method, result in methods.items():
                    if method != analysis_method and result.get("success"):
                        recommendation["alternatives"].append({
                            "method": method,
                            "reason": "Successful extraction but not recommended by analysis"
                        })
            else:
                recommendation["notes"].append(f"Analysis recommended {analysis_method} but it failed during testing")

        # Fallback to performance-based recommendation
        if not recommendation["recommended_method"]:
            successful_methods = [
                method for method, result in methods.items()
                if result.get("success")
            ]

            if successful_methods:
                # Prefer scanned text for speed and accuracy
                if "scanned" in successful_methods:
                    recommendation["recommended_method"] = "scanned"
                    recommendation["confidence"] = 0.8
                    recommendation["reason"] = "Scanned text extraction available - faster and more accurate"
                else:
                    recommendation["recommended_method"] = successful_methods[0]
                    recommendation["confidence"] = 0.6
                    recommendation["reason"] = f"Only {successful_methods[0]} method available"

                # Add other successful methods as alternatives
                for method in successful_methods:
                    if method != recommendation["recommended_method"]:
                        recommendation["alternatives"].append({
                            "method": method,
                            "reason": "Alternative extraction method"
                        })
            else:
                recommendation["recommended_method"] = "none"
                recommendation["confidence"] = 0.0
                recommendation["reason"] = "No extraction methods succeeded"

        return recommendation

    def display_comparison_results(self, results: Dict[str, Any]):
        """Display extraction results in formatted way."""
        print("\n" + "=" * 60)
        print("📊 EXTRACTION RESULTS")
        print("=" * 60)

        print(f"📁 PDF: {results.get('pdf_path', 'unknown')}")

        # Method results
        print(f"\n🔧 Extraction Results:")
        extraction_results = results.get("extraction_results", {})
        for method, result in extraction_results.items():
            status = "✅ Success" if result.get("success") else "❌ Failed"
            print(f"\n   {method.upper()} Method: {status}")
            print(f"      • Total Characters: {result.get('total_characters', 0):,}")

            # Show extracted text by page
            pages = result.get("pages", [])
            if pages:
                print(f"      • Extracted Text by Page:")
                for page in pages:
                    if page.get("success") and page.get("text"):
                        page_text = page["text"]
                        print(f"        --- Page {page['page_number']} ({page.get('text_length', 0)} chars) ---")
                        print(f"        {page_text}")
                    else:
                        print(f"        --- Page {page['page_number']} (FAILED) ---")

    def save_comparison_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save extraction results to JSON file."""
        if not filename:
            timestamp = int(time.time())
            pdf_name = Path(results.get('pdf_path', 'unknown')).stem
            filename = f"extraction_{pdf_name}_{timestamp}.json"

        output_path = self.output_dir / filename

        try:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Extraction results saved to: {output_path}")

            # Save clean text-only file
            text_output_path = output_path.with_suffix('.txt')
            with open(text_output_path, 'w', encoding='utf-8') as f:
                extraction_results = results.get('extraction_results', {})
                for method, result in extraction_results.items():
                    if result.get('success') and result.get('total_characters', 0) > 0:
                        f.write(f"Method: {method.upper()}\n")
                        f.write(f"PDF: {results.get('pdf_path', 'unknown')}\n")
                        f.write(f"Total Characters: {result.get('total_characters', 0):,}\n")
                        f.write("="*50 + "\n\n")

                        pages = result.get('pages', [])
                        for page in pages:
                            if page.get('success') and page.get('text'):
                                f.write(f"--- PAGE {page['page_number']} ---\n")
                                f.write(page['text'])
                                f.write("\n\n")
                        break  # Only save the method that actually extracted text

            print(f"📄 Extracted text saved to: {text_output_path}")

        except Exception as e:
            print(f"\n❌ Error saving results: {e}")


def main():
    """Main function for extraction testing."""
    tester = ExtractionTest()

    while True:
        print("\n" + "=" * 50)
        print("🔬 Extraction Method Test Menu")
        print("=" * 50)
        print("1. Test Single PDF (Compare Methods)")
        print("2. Back to Main Menu")
        print("-" * 50)

        choice = input("Enter your choice (1-2): ").strip()

        if choice == '1':
            # Get PDF path from user
            print(f"💡 Example path format: {get_example_pdf_path()}")
            while True:
                pdf_path_str = input("\n📁 Enter PDF path (or 'back' to return): ").strip()
                if pdf_path_str.lower() == 'back':
                    break

                try:
                    pdf_path = normalize_path_input(pdf_path_str)
                    if not pdf_path.exists():
                        print("❌ File not found. Try again.")
                        continue
                    if pdf_path.suffix.lower() != '.pdf':
                        print("❌ Not a PDF file. Try again.")
                        continue
                except ValueError as e:
                    print(f"❌ Invalid path: {e}")
                    continue

                # Run comparison test
                results = tester.run_comparison_test(pdf_path)
                tester.display_comparison_results(results)

                # Ask to save results
                save_choice = input("\n💾 Save results? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    tester.save_comparison_results(results)

                break

        elif choice == '2':
            print("👋 Returning to main menu...")
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()