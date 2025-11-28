"""
Multiple model testing interface
Allows testing multiple models on the same selected data file
"""
import sys
import os
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, List

# Add the parent directory to the path to import from config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.security import SecureConfig
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config.security import SecureConfig


class MultipleModelTester:
    """Interface for testing multiple models on the same data"""

    def __init__(self):
        # Use absolute paths based on the script location
        script_dir = Path(__file__).parent
        self.extracted_data_dir = script_dir.parent / "extracted_data"  # Go up one level to extracted_data
        self.all_models_dir = script_dir / "allModels"
        self.multiple_output_dir = script_dir / "multiple"
        self.multiple_output_dir.mkdir(exist_ok=True)

        # Available models mapping
        self.available_models = {}
        self.load_available_models()

    def load_available_models(self):
        """Load all available models from the allModels directory"""
        model_files = list(self.all_models_dir.glob("*.py"))

        # Exclude base_model.py and __init__.py
        model_files = [f for f in model_files if f.name not in ["base_model.py", "__init__.py"]]

        for i, model_file in enumerate(model_files, 1):
            self.available_models[i] = {
                "file": model_file,
                "name": model_file.stem,
                "module_name": model_file.stem
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

    def display_available_models(self):
        """Display list of available models"""
        if not self.available_models:
            print("ERROR: No models found in allModels directory!")
            return False

        print("\nAvailable models:")
        print("=" * 60)

        for num, model_info in self.available_models.items():
            print(f"{num}. {model_info['name']}")

        print("=" * 60)
        print("TIP: You can select multiple models (e.g., 1,3,5 or 1-5)")
        return True

    def get_selected_models(self) -> List[Dict[str, Any]]:
        """Get user selection for models"""
        while True:
            try:
                choice = input(f"\nSelect models (1-{len(self.available_models)}) or 'q' to quit: ").strip().lower()

                if choice == 'q':
                    print("Exiting...")
                    sys.exit(0)

                selected_models = []

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
                            if num in self.available_models:
                                selected_models.append(self.available_models[num])
                    else:
                        # Handle single number
                        num = int(part)
                        if num in self.available_models:
                            selected_models.append(self.available_models[num])

                if not selected_models:
                    print("ERROR: No valid models selected. Please try again.")
                    continue

                print(f"Selected {len(selected_models)} model(s):")
                for model in selected_models:
                    print(f"   - {model['name']}")

                return selected_models

            except ValueError:
                print("ERROR: Please enter valid numbers separated by commas")
            except Exception as e:
                print(f"ERROR: Error parsing selection: {e}")

    def load_model_module(self, model_info: Dict[str, Any]):
        """Load a model module dynamically"""
        model_path = model_info["file"]

        # Load the module
        spec = importlib.util.spec_from_file_location(model_info["module_name"], model_path)
        module = importlib.util.module_from_spec(spec)

        # Add the allModels directory to sys.path for imports
        sys.path.insert(0, str(self.all_models_dir))

        try:
            spec.loader.exec_module(module)

            # Get the model class using a mapping of file names to class names
            class_name_mapping = {
                'phi_3_small_8k_instruct': 'Phi_3_Small_8K_Instruct',
                'meta_llama_3_1_8b_instruct': 'MetaLlama31_8B_Instruct',
                'mistral_7b_instruct': 'Mistral_7B_Instruct',
                'gpt_oss_20b': 'GPT_OSS_20B_Model',
                'qwen_2_5_7b_instruct': 'Qwen_2_5_7B_Instruct_Model',
                'gemma_3_4b_it': 'Gemma_3_4B_IT'
            }

            class_name = class_name_mapping.get(model_info["name"])
            if not class_name:
                # Fallback to snake_case to PascalCase conversion
                class_name = ''.join(word.capitalize() for word in model_info["name"].split('_'))

            model_class = getattr(module, class_name)

            return model_class()
        except Exception as e:
            print(f"ERROR: Error loading model {model_info['name']}: {e}")
            return None
        finally:
            # Remove from sys.path
            if str(self.all_models_dir) in sys.path:
                sys.path.remove(str(self.all_models_dir))

    def run_model(self, model, data_file: Path) -> Dict[str, Any]:
        """Run a single model on the selected data"""
        print(f"\nRunning {model.model_name}...")

        try:
            # Load the extracted data
            with open(data_file, 'r', encoding='utf-8') as f:
                extracted_data = json.load(f)

            # Process the data using the model
            result = model.process_entire_document(extracted_data)

            print(f"{model.model_name} completed successfully")
            return result

        except Exception as e:
            print(f"ERROR: Error running {model.model_name}: {e}")
            return None

    def save_model_result(self, model_name: str, result: Dict[str, Any], data_file: Path):
        """Save the model result to the multiple folder"""
        if result is None:
            print(f"WARNING: Skipping save for {model_name} due to processing error")
            return

        # Create output filename
        output_filename = f"{model_name}.json"
        output_path = self.multiple_output_dir / output_filename

        # Prepare the data to save (only the model's response)
        # Handle both old format and new base class format
        # First check document_result for combined approach, then structured_data
        if "document_result" in result:
            response_data = result["document_result"].get("structured_data", {})
        else:
            response_data = result.get("data", result.get("structured_data", {}))

        # Also extract from document_summary if available for better data
        if "document_summary" in result:
            summary = result["document_summary"]
            # Merge summary data into response if it has more complete information
            if summary.get("patient_summary"):
                response_data["patient"] = summary["patient_summary"]
            if summary.get("doctor_npi"):
                response_data.setdefault("doctor", {})["npi"] = summary["doctor_npi"]
            if summary.get("insurance_summary"):
                response_data["insurance"] = summary["insurance_summary"]
            if summary.get("dme_summary"):
                response_data["dme"] = summary["dme_summary"]

        output_data = {
            "model": model_name,
            "source_data_file": data_file.name,
            "extraction_date": result.get("processing_timestamp", ""),
            "response": response_data
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Saved result to: {output_path}")
        except Exception as e:
            print(f"ERROR: Error saving result for {model_name}: {e}")

    def run(self):
        """Main execution method"""
        print("Multiple Model Testing Interface")
        print("=" * 50)

        # Step 1: Show data files and get selection
        data_files = self.display_available_data_files()
        if not data_files:
            return 1

        selected_data_file = self.get_selected_data_file(data_files)

        # Step 2: Show models and get selection
        if not self.display_available_models():
            return 1

        selected_models = self.get_selected_models()

        # Step 3: Run selected models
        print(f"\nStarting testing with {len(selected_models)} model(s) on {selected_data_file.name}")
        print("=" * 70)

        successful_runs = 0
        total_runs = len(selected_models)

        for i, model_info in enumerate(selected_models, 1):
            print(f"\n[{i}/{total_runs}] Processing model: {model_info['name']}")
            print("-" * 50)

            # Load and run the model
            model = self.load_model_module(model_info)
            if model is None:
                continue

            result = self.run_model(model, selected_data_file)

            # Save the result
            self.save_model_result(model.model_name, result, selected_data_file)

            if result is not None:
                successful_runs += 1

                # Display extracted field summary
                summary = result.get("document_summary", {})
                if summary:
                    print(f"   [INFO] Extracted fields for {model.model_name}:")
                    patient = summary.get("patient_summary", {})
                    if patient.get("full_name"):
                        print(f"   - Patient: {patient['full_name']}")
                    if patient.get("dob"):
                        print(f"   - DOB: {patient['dob']}")
                    if summary.get("doctor_npi"):
                        print(f"   - Doctor NPI: {summary['doctor_npi']}")

                    insurance = summary.get("insurance_summary", {})
                    if insurance.get("primary_insurance"):
                        print(f"   - Primary Insurance: {insurance['primary_insurance']} (ID: {insurance.get('primary_insurance_id', 'N/A')})")
                    if insurance.get("secondary_insurance"):
                        print(f"   - Secondary Insurance: {insurance['secondary_insurance']} (ID: {insurance.get('secondary_insurance_id', 'N/A')})")

                    dme = summary.get("dme_summary", {})
                    if dme.get("dme_id"):
                        print(f"   - DME ID: {dme['dme_id']}")
                    if dme.get("items"):
                        print(f"   - DME Items: {len(dme['items'])} item(s)")
                        for item in dme["items"][:3]:  # Show first 3 items
                            print(f"     • {item.get('item_name', 'Unknown')} (Qty: {item.get('item_quantity', 'N/A')})")

        # Summary
        print("\n" + "=" * 70)
        print(f"Testing completed!")
        print(f"Successful runs: {successful_runs}/{total_runs}")
        print(f"Results saved in: {self.multiple_output_dir}")
        print("=" * 70)

        return 0 if successful_runs > 0 else 1


def main():
    """Main function"""
    try:
        tester = MultipleModelTester()
        return tester.run()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())