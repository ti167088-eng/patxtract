"""
PatXtract OCR Demo

Demonstrates OCR capabilities and provides performance testing
for both PaddleOCR and Tesseract OCR engines.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from file_manager.extractor.ocr.paddleocr import PaddleOCRExtractor
from file_manager.extractor.ocr.tesseract import TesseractExtractor


class OCRDemo:
    """Demo and testing interface for OCR engines."""

    def __init__(self):
        """Initialize the OCR demo."""
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)

        # Initialize OCR engines
        self.engines = {}
        self._initialize_engines()

    def _initialize_engines(self):
        """Initialize available OCR engines."""
        print("🔧 Initializing OCR engines...")

        # Initialize PaddleOCR
        try:
            self.engines["paddleocr"] = PaddleOCRExtractor()
            print("   ✅ PaddleOCR initialized")
        except ImportError:
            print("   ❌ PaddleOCR not available (install with: pip install paddleocr)")
        except Exception as e:
            print(f"   ❌ PaddleOCR initialization failed: {e}")

        # Initialize Tesseract
        try:
            self.engines["tesseract"] = TesseractExtractor()
            print("   ✅ Tesseract initialized")
        except ImportError:
            print("   ❌ Tesseract not available (install with: pip install pytesseract)")
        except Exception as e:
            print(f"   ❌ Tesseract initialization failed: {e}")

        if not self.engines:
            print("⚠️  No OCR engines available!")
        else:
            print(f"✅ {len(self.engines)} OCR engine(s) ready")

    def show_engine_info(self):
        """Display information about available engines."""
        print("\n📊 OCR Engine Information:")
        print("=" * 40)

        for engine_name, engine in self.engines.items():
            print(f"\n🔧 {engine_name.upper()}:")
            print(f"   Class: {engine.__class__.__name__}")

            if hasattr(engine, 'config'):
                print(f"   Configuration: {engine.config}")

            # Run a quick test if possible
            try:
                # Create a simple test image with text (if we have PIL)
                from PIL import Image, ImageDraw, ImageFont
                import numpy as np

                # Create test image
                img = Image.new('RGB', (400, 100), color='white')
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), "Hello World! OCR Test 123", fill='black')

                # Test extraction using PIL Image directly
                start_time = time.time()
                result = engine.extract_text_from_image(img)
                processing_time = time.time() - start_time

                print(f"   ✅ Quick test successful")
                print(f"   ⏱️  Processing time: {processing_time:.3f}s")
                if result.get("success"):
                    print(f"   📝 Extracted text: {result.get('text', '')[:50]}...")

            except Exception as e:
                print(f"   ⚠️  Quick test failed: {e}")

    def demo_ocr_on_image(self, image_path: Path):
        """Demonstrate OCR on a specific image."""
        if not image_path.exists():
            print(f"❌ Image file not found: {image_path}")
            return

        print(f"\n🖼️  Running OCR demo on: {image_path.name}")
        print("=" * 50)

        results = {}

        for engine_name, engine in self.engines.items():
            print(f"\n🔧 Testing {engine_name.upper()}...")
            try:
                start_time = time.time()
                result = engine.extract_text(image_path)
                processing_time = time.time() - start_time

                results[engine_name] = {
                    "success": result.get("success", False),
                    "processing_time": processing_time,
                    "text": result.get("text", ""),
                    "confidence": result.get("average_confidence", 0),
                    "errors": result.get("errors", []),
                    "details": result
                }

                if result.get("success"):
                    text = result.get("text", "")
                    confidence = result.get("average_confidence", 0)
                    print(f"   ✅ Success ({processing_time:.3f}s)")
                    print(f"   📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                    print(f"   📊 Confidence: {confidence:.1f}%")
                else:
                    errors = result.get("errors", [])
                    print(f"   ❌ Failed: {'; '.join(errors) if errors else 'Unknown error'}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[engine_name] = {
                    "success": False,
                    "error": str(e),
                    "processing_time": 0
                }

        # Compare results
        self._compare_ocr_results(results)

    def _compare_ocr_results(self, results: Dict[str, Any]):
        """Compare results from different OCR engines."""
        if len(results) < 2:
            return

        print(f"\n📈 OCR Comparison:")
        print("-" * 30)

        successful = {name: result for name, result in results.items() if result.get("success")}

        if len(successful) >= 2:
            # Speed comparison
            fastest_engine = min(successful.keys(), key=lambda e: successful[e]["processing_time"])
            fastest_time = successful[fastest_engine]["processing_time"]

            print(f"⚡ Speed:")
            for engine_name, result in successful.items():
                relative_speed = result["processing_time"] / fastest_time
                print(f"   {engine_name}: {result['processing_time']:.3f}s ({relative_speed:.1f}x relative)")

            # Confidence comparison
            print(f"\n📊 Confidence:")
            for engine_name, result in successful.items():
                confidence = result.get("confidence", 0)
                print(f"   {engine_name}: {confidence:.1f}%")

            # Text length comparison
            print(f"\n📝 Text Length:")
            for engine_name, result in successful.items():
                text_length = len(result.get("text", ""))
                print(f"   {engine_name}: {text_length} characters")

            # Text similarity
            if len(successful) == 2:
                engines = list(successful.keys())
                text1 = successful[engines[0]]["text"]
                text2 = successful[engines[1]]["text"]
                similarity = self._calculate_similarity(text1, text2)
                print(f"\n🔗 Text Similarity: {similarity:.1%}")

        else:
            print("Only one engine succeeded, no comparison possible")

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def batch_test_images(self, image_dir: Path):
        """Run batch OCR tests on all images in a directory."""
        if not image_dir.exists() or not image_dir.is_dir():
            print(f"❌ Directory not found: {image_dir}")
            return

        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"❌ No image files found in: {image_dir}")
            return

        print(f"\n📂 Batch OCR Test on {len(image_files)} images")
        print("=" * 50)

        batch_results = {
            "directory": str(image_dir),
            "total_images": len(image_files),
            "results": {},
            "summary": {}
        }

        for image_path in image_files:
            print(f"\n🖼️  Processing: {image_path.name}")
            image_results = {}

            for engine_name, engine in self.engines.items():
                try:
                    start_time = time.time()
                    result = engine.extract_text(image_path)
                    processing_time = time.time() - start_time

                    image_results[engine_name] = {
                        "success": result.get("success", False),
                        "processing_time": processing_time,
                        "confidence": result.get("average_confidence", 0),
                        "text_length": len(result.get("text", "")),
                        "errors": result.get("errors", [])
                    }

                    status = "✅" if result.get("success") else "❌"
                    print(f"   {engine_name}: {status} ({processing_time:.3f}s)")

                except Exception as e:
                    image_results[engine_name] = {
                        "success": False,
                        "error": str(e),
                        "processing_time": 0
                    }
                    print(f"   {engine_name}: ❌ Error: {e}")

            batch_results["results"][image_path.name] = image_results

        # Generate summary
        self._generate_batch_summary(batch_results)

    def _generate_batch_summary(self, batch_results: Dict[str, Any]):
        """Generate summary statistics for batch test."""
        print(f"\n📊 Batch Test Summary:")
        print("-" * 30)

        summary = {
            "engine_stats": {},
            "best_engine": None,
            "best_avg_time": float('inf')
        }

        for engine_name in self.engines.keys():
            successful_count = 0
            total_time = 0
            total_confidence = 0
            confidence_count = 0

            for image_name, image_results in batch_results["results"].items():
                if engine_name in image_results:
                    result = image_results[engine_name]
                    if result.get("success"):
                        successful_count += 1
                        total_time += result.get("processing_time", 0)
                        if result.get("confidence"):
                            total_confidence += result["confidence"]
                            confidence_count += 1

            avg_time = total_time / successful_count if successful_count > 0 else 0
            avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
            success_rate = (successful_count / batch_results["total_images"]) * 100

            summary["engine_stats"][engine_name] = {
                "success_rate": success_rate,
                "successful_count": successful_count,
                "avg_time": avg_time,
                "avg_confidence": avg_confidence
            }

            print(f"{engine_name.upper()}:")
            print(f"   ✅ Success Rate: {success_rate:.1f}% ({successful_count}/{batch_results['total_images']})")
            print(f"   ⏱️  Avg Time: {avg_time:.3f}s")
            print(f"   📊 Avg Confidence: {avg_confidence:.1f}%")

            if successful_count > 0 and avg_time < summary["best_avg_time"]:
                summary["best_avg_time"] = avg_time
                summary["best_engine"] = engine_name

        if summary["best_engine"]:
            print(f"\n🏆 Best Performing Engine: {summary['best_engine'].upper()}")
            print(f"   Average Time: {summary['best_avg_time']:.3f}s")

        batch_results["summary"] = summary

        # Save batch results
        timestamp = int(time.time())
        filename = f"ocr_batch_results_{timestamp}.json"
        output_path = self.output_dir / filename

        try:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Batch results saved to: {output_path}")
        except Exception as e:
            print(f"\n❌ Error saving batch results: {e}")


def main():
    """Main function for OCR demo."""
    demo = OCRDemo()

    if not demo.engines:
        print("\n❌ No OCR engines available. Please install PaddleOCR and/or Tesseract.")
        print("   • PaddleOCR: pip install paddleocr")
        print("   • Tesseract: pip install pytesseract")
        return

    while True:
        print("\n" + "=" * 50)
        print("🔬 OCR Demo Menu")
        print("=" * 50)
        print("1. Show Engine Information")
        print("2. Test Single Image")
        print("3. Batch Test Directory")
        print("4. Exit")
        print("-" * 50)

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            demo.show_engine_info()

        elif choice == '2':
            image_path_str = input("\n📁 Enter image path: ").strip()
            if image_path_str:
                demo.demo_ocr_on_image(Path(image_path_str))

        elif choice == '3':
            dir_path_str = input("\n📂 Enter directory path: ").strip()
            if dir_path_str:
                demo.batch_test_images(Path(dir_path_str))

        elif choice == '4':
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()