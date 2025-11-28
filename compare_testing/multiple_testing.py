"""
Multiple Patient Algorithm Testing Interface

Allows testing multiple patient identification algorithms on the same extracted data file
and comparing their results. This interface mirrors the structure_testing multiple_testing.py
but works with patient identification algorithms instead of AI extraction models.
"""

import sys
import os
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add the parent directory to the path to import from config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.security import SecureConfig
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config.security import SecureConfig

from extract_data import DataExtractor


class MultipleAlgorithmTester:
    """Interface for testing multiple patient identification algorithms on the same data"""

    def __init__(self):
        # Use absolute paths based on the script location
        script_dir = Path(__file__).parent
        self.extracted_data_dir = script_dir / "extracted_data"
        self.all_algorithms_dir = script_dir / "allModels"
        self.compare_results_dir = script_dir / "compare_results"
        self.compare_results_dir.mkdir(exist_ok=True)
        self.output_dir = script_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

        # Available algorithms mapping
        self.available_algorithms = {}
        self.load_available_algorithms()

        # Data extractor
        self.data_extractor = DataExtractor(str(self.extracted_data_dir))

    def load_available_algorithms(self):
        """Load all available algorithms from the allModels directory"""
        algorithm_files = list(self.all_algorithms_dir.glob("*.py"))

        # Exclude base_patient_identifier.py and __init__.py
        algorithm_files = [f for f in algorithm_files if f.name not in ["base_patient_identifier.py", "__init__.py"]]

        for i, algorithm_file in enumerate(algorithm_files, 1):
            self.available_algorithms[i] = {
                "file": algorithm_file,
                "name": algorithm_file.stem,
                "module_name": algorithm_file.stem
            }

    def display_available_data_files(self) -> List[Path]:
        """Display list of available extracted data files"""
        if not self.extracted_data_dir.exists():
            print("ERROR: No 'extracted_data' directory found!")
            print("Please run extract_data.py first to extract data from PDF files.")
            return []

        data_files = list(self.extracted_data_dir.glob("*.json"))
        if not data_files:
            print("ERROR: No extracted data files found!")
            print("Please run extract_data.py first to extract data from PDF files.")
            return []

        print("\nAvailable extracted data files:")
        print("=" * 50)

        for i, file in enumerate(data_files, 1):
            print(f"{i}. {file.name}")

        print("=" * 50)
        return data_files

    def get_selected_data_file(self, data_files: List[Path]) -> Path:
        """Get user selection for data file"""
        while True:
            try:
                choice = input(f"\nSelect data file (1-{len(data_files)}) or 'q' to quit: ").strip().lower()

                if choice == 'q':
                    print("Exiting...")
                    sys.exit(0)

                choice_num = int(choice)
                if 1 <= choice_num <= len(data_files):
                    selected_file = data_files[choice_num - 1]
                    print(f"Selected: {selected_file.name}")
                    return selected_file
                else:
                    print(f"ERROR: Please enter a number between 1 and {len(data_files)}")
            except ValueError:
                print("ERROR: Please enter a valid number")

    def display_available_algorithms(self):
        """Display list of available algorithms"""
        if not self.available_algorithms:
            print("ERROR: No algorithms found in allModels directory!")
            return False

        print("\nAvailable patient identification algorithms:")
        print("=" * 60)

        for num, algorithm_info in self.available_algorithms.items():
            print(f"{num}. {algorithm_info['name']}")

        print("=" * 60)
        print("TIP: You can select multiple algorithms (e.g., 1,3,5 or 1-5)")
        return True

    def get_selected_algorithms(self) -> List[Dict[str, Any]]:
        """Get user selection for algorithms"""
        while True:
            try:
                choice = input(f"\nSelect algorithms (1-{len(self.available_algorithms)}) or 'q' to quit: ").strip().lower()

                if choice == 'q':
                    print("Exiting...")
                    sys.exit(0)

                selected_algorithms = []

                # Parse comma-separated numbers and ranges
                parts = choice.split(',')
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        # Handle range (e.g., 1-3)
                        start, end = part.split('-')
                        start_num = int(start.strip())
                        end_num = int(end.strip())
                        for num in range(start_num, end_num + 1):
                            if num in self.available_algorithms:
                                selected_algorithms.append(self.available_algorithms[num])
                    else:
                        # Handle single number
                        num = int(part)
                        if num in self.available_algorithms:
                            selected_algorithms.append(self.available_algorithms[num])

                if not selected_algorithms:
                    print("ERROR: No valid algorithms selected. Please try again.")
                    continue

                print(f"Selected {len(selected_algorithms)} algorithm(s):")
                for algorithm in selected_algorithms:
                    print(f"   - {algorithm['name']}")

                return selected_algorithms

            except ValueError:
                print("ERROR: Please enter valid numbers separated by commas")
            except Exception as e:
                print(f"ERROR: Error parsing selection: {e}")

    def load_algorithm_module(self, algorithm_info: Dict[str, Any]):
        """Load an algorithm module dynamically"""
        algorithm_path = algorithm_info["file"]

        # Load the module
        spec = importlib.util.spec_from_file_location(algorithm_info["module_name"], algorithm_path)
        module = importlib.util.module_from_spec(spec)

        # Add the allModels directory to sys.path for imports
        sys.path.insert(0, str(self.all_algorithms_dir))

        try:
            spec.loader.exec_module(module)

            # Get the algorithm class using a mapping of file names to class names
            class_name_mapping = {
                'name_dob_matcher': 'NameDobMatcher',
                'gpt_oss_20b': 'GPT_OSS_20B_PatientIdentifier',
                'qwen_2_5_7b_instruct': 'Qwen_2_5_7B_Instruct_PatientIdentifier',
                'qwen3_vl_8b_instruct': 'Qwen3_VL_8B_Instruct_PatientIdentifier',
                'gemma_3_4b_it': 'Gemma_3_4B_IT_PatientIdentifier',
                'mistral_7b_instruct': 'Mistral_7B_Instruct_PatientIdentifier',
                'name_address_matcher': 'NameAddressMatcher',  # Future
                'comprehensive_matcher': 'ComprehensiveMatcher',  # Future
                'fuzzy_matcher': 'FuzzyMatcher'  # Future
            }

            class_name = class_name_mapping.get(algorithm_info["name"])
            if not class_name:
                # Fallback to snake_case to PascalCase conversion
                class_name = ''.join(word.capitalize() for word in algorithm_info["name"].split('_'))

            algorithm_class = getattr(module, class_name)

            return algorithm_class()
        except Exception as e:
            print(f"ERROR: Error loading algorithm {algorithm_info['name']}: {e}")
            return None
        finally:
            # Remove from sys.path
            if str(self.all_algorithms_dir) in sys.path:
                sys.path.remove(str(self.all_algorithms_dir))

    def run_algorithm(self, algorithm, data_file: Path) -> Dict[str, Any]:
        """Run a single algorithm on the selected data"""
        print(f"\nRunning {algorithm.algorithm_name}...")

        try:
            # Load the extracted data
            print(f"📂 Loading data from {data_file.name}...", end=" ", flush=True)
            with open(data_file, 'r', encoding='utf-8') as f:
                extracted_data = json.load(f)
            print("✅ Loaded")

            # Get pages data
            pages = extracted_data.get('pages', [])
            total_pages = len(pages)

            print(f"📄 Processing {total_pages} pages with {algorithm.algorithm_name}")
            print(f"⏳ This may take several minutes for large documents...")
            print(f"🔄 Starting patient identification...", flush=True)

            # Process the data using the algorithm
            import time
            start_time = time.time()

            result = algorithm.identify_patients(pages)

            processing_time = time.time() - start_time

            if result:
                # Extract summary information
                summary = result.get('summary', {})
                total_patients = summary.get('total_patients', 0)
                total_entries = summary.get('total_entries', 0)
                assigned_pages = summary.get('assigned_pages', 0)
                avg_confidence = summary.get('average_confidence', 0.0)

                # Check if chunking was used
                chunking_info = result.get('chunking_info', {})
                processed_in_chunks = chunking_info.get('processed_in_chunks', False)
                num_chunks = chunking_info.get('num_chunks', 1)

                print(f"✅ {algorithm.algorithm_name} completed successfully!")
                print(f"⏱️  Processing time: {processing_time:.2f} seconds")
                print(f"👥 Patients found: {total_patients}")
                print(f"📋 Total entries: {total_entries}")
                print(f"📄 Pages assigned: {assigned_pages}/{total_pages}")
                print(f"🎯 Average confidence: {avg_confidence:.2%}")

                if processed_in_chunks:
                    print(f"🔀 Processed in {num_chunks} chunks (large document mode)")

                # Add vision processing info if applicable
                if result.get('vision_enabled', False):
                    images_processed = result.get('images_processed', 0)
                    print(f"👁️  Vision mode: {images_processed} images analyzed")
            else:
                print(f"❌ {algorithm.algorithm_name} returned no results")

            return result

        except Exception as e:
            print(f"❌ ERROR: Error running {algorithm.algorithm_name}: {e}")
            # Print more detailed error information for debugging
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return None

    def save_algorithm_result(self, algorithm_name: str, result: Dict[str, Any], data_file: Path):
        """Save the algorithm result to the output folder"""
        if result is None:
            print(f"WARNING: Skipping save for {algorithm_name} due to processing error")
            return

        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{algorithm_name}_{timestamp}.json"
        output_path = self.output_dir / output_filename

        # Prepare the data to save
        save_data = {
            **result,
            "source_data_file": data_file.name,
            "algorithm_name": algorithm_name
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"Saved result to: {output_path}")
            return output_path
        except Exception as e:
            print(f"ERROR: Error saving result for {algorithm_name}: {e}")
            return None

    def create_comparison_analysis(self, algorithm_results: List[Dict[str, Any]], data_file: Path) -> Dict[str, Any]:
        """Create comparison analysis between different algorithm results"""
        if not algorithm_results:
            return {}

        print("\nCreating comparison analysis...")

        analysis = {
            "comparison_timestamp": datetime.now().isoformat(),
            "source_data_file": data_file.name,
            "algorithms_tested": [result.get('algorithm_name', 'Unknown') for result in algorithm_results],
            "algorithm_results": {},
            "patient_count_agreement": {},
            "page_assignment_differences": [],
            "summary_statistics": {}
        }

        # Collect algorithm statistics
        patient_counts = []
        total_entries = []
        processing_times = []
        avg_confidences = []

        for result in algorithm_results:
            if result:
                algorithm_name = result.get('algorithm_name', 'Unknown')
                summary = result.get('summary', {})

                analysis["algorithm_results"][algorithm_name] = {
                    "result_file": f"{algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "total_patients": summary.get('total_patients', 0),
                    "total_entries": summary.get('total_entries', 0),
                    "assigned_pages": summary.get('assigned_pages', 0),
                    "unassigned_pages": summary.get('unassigned_pages', 0),
                    "average_confidence": summary.get('average_confidence', 0.0),
                    "processing_time_seconds": result.get('processing_time_seconds', 0.0)
                }

                patient_counts.append(summary.get('total_patients', 0))
                total_entries.append(summary.get('total_entries', 0))
                processing_times.append(result.get('processing_time_seconds', 0.0))
                avg_confidences.append(summary.get('average_confidence', 0.0))

        # Patient count agreement analysis
        if patient_counts:
            analysis["patient_count_agreement"] = {
                "consensus_patients": max(set(patient_counts), key=patient_counts.count) if patient_counts else 0,
                "max_patients": max(patient_counts),
                "min_patients": min(patient_counts),
                "agreement_percentage": (patient_counts.count(max(set(patient_counts), key=patient_counts.count)) / len(patient_counts) * 100) if patient_counts else 0
            }

        # Summary statistics
        if len(algorithm_results) > 1:
            analysis["summary_statistics"] = {
                "average_patients_per_algorithm": sum(patient_counts) / len(patient_counts),
                "average_entries_per_algorithm": sum(total_entries) / len(total_entries),
                "average_processing_time": sum(processing_times) / len(processing_times),
                "average_confidence_across_algorithms": sum(avg_confidences) / len(avg_confidences),
                "processing_time_variance": max(processing_times) - min(processing_times) if processing_times else 0
            }

        # Note: Page assignment differences would require more complex analysis
        # This could be implemented in a future version

        return analysis

    def save_comparison_analysis(self, analysis: Dict[str, Any], data_file: Path):
        """Save the comparison analysis to the compare_results folder"""
        if not analysis:
            print("WARNING: No comparison analysis to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"comparison_analysis_{timestamp}.json"
        output_path = self.compare_results_dir / output_filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"\nComparison analysis saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"ERROR: Error saving comparison analysis: {e}")
            return None

    def run(self):
        """Main execution method"""
        print("Multiple Patient Algorithm Testing Interface")
        print("=" * 50)

        # Step 1: Show data files and get selection
        data_files = self.display_available_data_files()
        if not data_files:
            return 1

        selected_data_file = self.get_selected_data_file(data_files)

        # Step 2: Show algorithms and get selection
        if not self.display_available_algorithms():
            return 1

        selected_algorithms = self.get_selected_algorithms()

        # Step 3: Run selected algorithms
        print(f"\nStarting testing with {len(selected_algorithms)} algorithm(s) on {selected_data_file.name}")
        print("=" * 70)

        successful_runs = 0
        total_runs = len(selected_algorithms)
        algorithm_results = []

        for i, algorithm_info in enumerate(selected_algorithms, 1):
            print(f"\n[{i}/{total_runs}] Processing algorithm: {algorithm_info['name']}")
            print("-" * 50)

            # Load and run the algorithm
            algorithm = self.load_algorithm_module(algorithm_info)
            if algorithm is None:
                continue

            result = self.run_algorithm(algorithm, selected_data_file)

            # Save the result
            if result:
                saved_path = self.save_algorithm_result(algorithm.algorithm_name, result, selected_data_file)
                algorithm_results.append(result)
                successful_runs += 1

        # Step 4: Create comparison analysis
        if len(algorithm_results) > 1:
            print("\n" + "=" * 70)
            print("Creating comparison analysis...")
            comparison_analysis = self.create_comparison_analysis(algorithm_results, selected_data_file)
            self.save_comparison_analysis(comparison_analysis, selected_data_file)

        # Summary
        print("\n" + "=" * 70)
        print(f"Testing completed!")
        print(f"Successful runs: {successful_runs}/{total_runs}")
        print(f"Individual results saved in: {self.output_dir}")
        if len(algorithm_results) > 1:
            print(f"Comparison analysis saved in: {self.compare_results_dir}")
        print("=" * 70)

        return 0 if successful_runs > 0 else 1


def main():
    """Main function"""
    try:
        tester = MultipleAlgorithmTester()
        return tester.run()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())