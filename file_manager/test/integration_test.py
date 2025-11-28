"""
PatXtract Integration Test

Quick integration test to verify the complete pipeline works correctly.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")

    try:
        from file_manager.manager.manager import FileManager
        print("   OK: FileManager imported successfully")

        from file_manager.extractor.extraction_routing import ExtractionRouter
        print("   OK: ExtractionRouter imported successfully")

        from file_manager.extractor.scanned.scan import ScannedTextExtractor
        print("   OK: ScannedTextExtractor imported successfully")

        from file_manager.extractor.ocr.paddleocr import PaddleOCRExtractor
        print("   OK: PaddleOCRExtractor imported successfully")

        from file_manager.extractor.ocr.tesseract import TesseractExtractor
        print("   OK: TesseractExtractor imported successfully")

        return True

    except ImportError as e:
        print(f"   ERROR: Import failed: {e}")
        return False

def test_file_manager_initialization():
    """Test FileManager initialization."""
    print("\n🏗️  Testing FileManager initialization...")

    try:
        from file_manager.manager.manager import FileManager

        manager = FileManager()
        print("   ✅ FileManager initialized successfully")

        # Test getting supported methods
        methods = manager.get_supported_methods()
        print(f"   ✅ Supported methods: {methods}")

        # Test getting stats
        stats = manager.get_processing_stats()
        print(f"   ✅ Initial stats: {stats}")

        return manager

    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return None

def test_extraction_router():
    """Test ExtractionRouter functionality."""
    print("\n🔍 Testing ExtractionRouter...")

    try:
        from file_manager.extractor.extraction_routing import ExtractionRouter

        router = ExtractionRouter()
        print("   ✅ ExtractionRouter initialized successfully")

        # Test getting supported methods
        methods = router.get_supported_methods()
        print(f"   ✅ Available extraction methods: {methods}")

        return router

    except Exception as e:
        print(f"   ❌ ExtractionRouter test failed: {e}")
        return None

def test_scanned_text_extractor():
    """Test ScannedTextExtractor functionality."""
    print("\n📝 Testing ScannedTextExtractor...")

    try:
        from file_manager.extractor.scanned.scan import ScannedTextExtractor

        extractor = ScannedTextExtractor()
        print("   ✅ ScannedTextExtractor initialized successfully")

        return extractor

    except Exception as e:
        print(f"   ❌ ScannedTextExtractor test failed: {e}")
        return None

def test_configuration_loading():
    """Test configuration file loading."""
    print("\n⚙️  Testing configuration loading...")

    try:
        config_path = Path(__file__).parent.parent / "extractor" / "config.json"
        if config_path.exists():
            print(f"   ✅ Config file found: {config_path}")

            # Test loading with ExtractionRouter
            from file_manager.extractor.extraction_routing import ExtractionRouter
            router = ExtractionRouter(config_path)
            print("   ✅ Configuration loaded successfully")

            return True
        else:
            print(f"   ⚠️  Config file not found: {config_path}")
            print("   ✅ Using default configuration")
            return True

    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without actual files."""
    print("\n🧪 Testing basic functionality...")

    try:
        from file_manager.manager.manager import FileManager

        manager = FileManager()

        # Test with non-existent file (should handle gracefully)
        fake_path = Path("non_existent.pdf")
        result = manager.process_pdf(fake_path)

        if not result.get("success") and "File not found" in str(result.get("errors", [])):
            print("   ✅ File not found handled correctly")
            return True
        else:
            print("   ⚠️  Unexpected result for non-existent file")
            return False

    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False

def run_integration_test():
    """Run complete integration test."""
    print("=" * 60)
    print("🏥 PatXtract Integration Test")
    print("=" * 60)

    start_time = time.time()

    # Test 1: Imports
    if not test_imports():
        return False

    # Test 2: Configuration
    if not test_configuration_loading():
        return False

    # Test 3: Component initialization
    manager = test_file_manager_initialization()
    if not manager:
        return False

    router = test_extraction_router()
    if not router:
        return False

    extractor = test_scanned_text_extractor()
    if not extractor:
        return False

    # Test 4: Basic functionality
    if not test_basic_functionality():
        return False

    # Final results
    total_time = time.time() - start_time
    print(f"\n✅ Integration test completed successfully!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 All components initialized and working correctly")

    return True

if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)