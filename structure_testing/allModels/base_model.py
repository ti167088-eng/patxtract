"""
Base Model Class

Abstract base class defining the interface for all AI model implementations
in the structure testing framework.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging
import time
from datetime import datetime
import sys

# We'll import FileManager only when needed to avoid PaddleOCR initialization
try:
    from structure_testing.config.security import SecureConfig
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(Path(__file__).parent.parent))
    from config.security import SecureConfig

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all AI model implementations.

    This class provides a common interface and functionality for all models
    used in the structure testing framework.
    """

    def __init__(self, model_name: str, model_id: Optional[str] = None):
        """
        Initialize the base model.

        Args:
            model_name: Human-readable name for the model
            model_id: Optional model identifier used in API calls
        """
        self.model_name = model_name
        self.model_id = model_id or model_name

        # Initialize secure configuration
        self.config = SecureConfig()

        # Validate configuration
        if not self.config.validate_config():
            raise ValueError("Invalid configuration - check environment variables")

        # Get API key and base URL
        self.api_key = self.config.get_openrouter_key()
        self.base_url = self.config.get_base_url()

        # Get model-specific configuration
        self.model_config = self.config.get_model_config(model_name)

        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{model_name}")

        # Don't initialize FileManager here - we'll only load it when needed
        self.file_manager = None

        self.logger.info(f"Initialized model: {model_name}")

    @abstractmethod
    def call_api(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Abstract method for making API calls to the specific model.

        Args:
            messages: List of message dictionaries for the API call
            **kwargs: Additional keyword arguments for the API call

        Returns:
            Dictionary containing the API response

        Raises:
            Exception: If API call fails
        """
        pass

    @abstractmethod
    def extract_structure(self, content: str) -> Dict[str, Any]:
        """
        Abstract method for extracting structured data from model response.

        Args:
            content: Raw text content from the model response

        Returns:
            Dictionary containing structured data with patient and doctor information
        """
        pass

    def get_extraction_prompt(self, page_text: str) -> str:
        """
        Generate a structured extraction prompt for the given text.

        Args:
            page_text: Text content from a PDF page

        Returns:
            Formatted prompt string for structured extraction
        """
        prompt = f"""
You are a medical document extractor. Extract the following information from the provided text and return it as a JSON object.

Required fields to extract:
Patient Information:
- full_name: Complete patient name
- first_name: Patient's first/given name
- middle_name: Patient's middle name/initial
- last_name: Patient's last/family name
- dob: Date of birth (MM/DD/YYYY format preferred)
- address_full: Complete street address
- city: City name
- state: Full state name
- country: Country name
- postal_code: Postal code
- state_code: 2-letter state abbreviation
- phone: Primary phone number of patient
- mobile_landline: Secondary/landline phone of patient
- email: Email address of patient
- fax: Fax number of patient
- account_number: Patient account number

Doctor Information:
- npi: National Provider Identifier (10-digit number)

Insurance Information:
- primary_insurance: Name of primary insurance provider
- primary_insurance_id: Primary insurance policy/member ID
- secondary_insurance: Name of secondary insurance provider (if any)
- secondary_insurance_id: Secondary insurance policy/member ID (if any)
- tertiary_insurance: Name of tertiary insurance provider (if any)
- tertiary_insurance_id: Tertiary insurance policy/member ID (if any)

DME (Durable Medical Equipment) Information:
- dme_id: DME supplier or order ID
- items: List of DME items, each containing:
  - item_name: Name/description of the DME item
  - item_quantity: Quantity of the item

Instructions:
1. Extract all available information from the text
2. If a field is not found, use an empty string ("") or null
3. For items array, if no items found, use an empty array []
4. Normalize phone numbers to standard format (e.g., "123-456-7890")
5. Ensure NPI is exactly 10 digits if found
6. Return only valid JSON object with the structure below

Expected JSON structure:
{{
    "patient": {{
        "full_name": "",
        "first_name": "",
        "middle_name": "",
        "last_name": "",
        "dob": "",
        "address_full": "",
        "city": "",
        "state": "",
        "country": "",
        "postal_code": "",
        "state_code": "",
        "phone": "",
        "mobile_landline": "",
        "email": "",
        "fax": "",
        "account_number": ""
    }},
    "doctor": {{
        "npi": ""
    }},
    "insurance": {{
        "primary_insurance": "",
        "primary_insurance_id": "",
        "secondary_insurance": "",
        "secondary_insurance_id": "",
        "tertiary_insurance": "",
        "tertiary_insurance_id": ""
    }},
    "dme": {{
        "dme_id": "",
        "items": []
    }}
}}

Document text to extract from:
{page_text}

Extract the information and return the JSON object:
"""
        return prompt.strip()

    def process_pdf(self, pdf_path: Path, use_cached_data: bool = True) -> Dict[str, Any]:
        """
        Process a PDF file page-wise and extract structured information.

        Args:
            pdf_path: Path to the PDF file
            use_cached_data: If True, try to use previously extracted OCR data

        Returns:
            Dictionary containing the complete extraction results
        """
        start_time = time.time()

        self.logger.info(f"Processing PDF: {pdf_path}")

        try:
            # Step 1: Get OCR data (from cache or fresh extraction)
            if use_cached_data:
                extracted_data = self._get_cached_data(pdf_path)
            else:
                extracted_data = self._extract_fresh_data(pdf_path)

            # Step 2: Process each page through the model
            pages = extracted_data.get('pages', [])
            structured_results = []
            successful_pages = 0

            for page_data in pages:
                page_result = self.process_page(page_data)
                structured_results.append(page_result)

                if page_result.get("success", False):
                    successful_pages += 1

            # Step 3: Create final result
            final_result = self._create_final_result(
                pdf_path,
                structured_results,
                extracted_data,
                time.time() - start_time,
                successful_pages,
                len(pages)
            )

            # Step 4: Save results
            self._save_results(final_result)

            self.logger.info(f"Successfully processed {successful_pages}/{len(pages)} pages")
            return final_result

        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {e}")
            return self._create_error_result(pdf_path, str(e), time.time() - start_time)

    def process_page(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process individual page data through the model.

        Args:
            page_data: Dictionary containing page information and text

        Returns:
            Dictionary containing the page processing results
        """
        page_number = page_data.get('page_number', 0)
        page_text = page_data.get('text', '')

        if not page_text.strip():
            return {
                "page_number": page_number,
                "success": False,
                "error": "No text found on page",
                "raw_text": page_text
            }

        try:
            # Generate extraction prompt
            prompt = self.get_extraction_prompt(page_text)
            messages = [{"role": "user", "content": prompt}]

            # Call the model API
            api_start = time.time()
            response = self.call_api(messages, temperature=0.1)
            api_time = time.time() - api_start

            # Extract structured data from response
            content = self._extract_content_from_response(response)
            structured_data = self.extract_structure(content)

            # Calculate confidence scores (placeholder - could be enhanced)
            confidence_scores = self._calculate_confidence(structured_data, content)

            return {
                "page_number": page_number,
                "success": True,
                "raw_text": page_text,
                "structured_data": structured_data,
                "api_response": response,
                "confidence_scores": confidence_scores,
                "processing_metadata": {
                    "api_response_time": api_time,
                    "tokens_used": response.get("usage", {}).get("total_tokens", 0),
                    "model_used": self.model_name
                }
            }

        except Exception as e:
            self.logger.error(f"Error processing page {page_number}: {e}")
            return {
                "page_number": page_number,
                "success": False,
                "error": str(e),
                "raw_text": page_text
            }

    def _extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """
        Extract content from API response.

        Args:
            response: API response dictionary

        Returns:
            Extracted content string
        """
        try:
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except (KeyError, IndexError, TypeError):
            raise ValueError("Invalid API response format")

    def _calculate_confidence(self, structured_data: Dict[str, Any], raw_content: str) -> Dict[str, float]:
        """
        Calculate confidence scores for the extraction.

        Args:
            structured_data: Extracted structured data
            raw_content: Raw content from model response

        Returns:
            Dictionary with confidence scores
        """
        # Simple confidence calculation - can be enhanced
        patient_confidence = 0.0
        doctor_confidence = 0.0
        insurance_confidence = 0.0
        dme_confidence = 0.0

        patient_data = structured_data.get("patient", {})
        doctor_data = structured_data.get("doctor", {})
        insurance_data = structured_data.get("insurance", {})
        dme_data = structured_data.get("dme", {})

        # Calculate patient confidence based on filled fields
        patient_fields = [
            "full_name", "first_name", "last_name", "dob", "address_full",
            "city", "state", "postal_code", "phone"
        ]
        filled_patient_fields = sum(1 for field in patient_fields if patient_data.get(field, "").strip())
        patient_confidence = filled_patient_fields / len(patient_fields) if patient_fields else 0.0

        # Calculate doctor confidence
        npi = doctor_data.get("npi", "").strip()
        if npi and len(npi) == 10 and npi.isdigit():
            doctor_confidence = 1.0
        elif npi:
            doctor_confidence = 0.5

        # Calculate insurance confidence
        insurance_fields = [
            "primary_insurance", "primary_insurance_id",
            "secondary_insurance", "secondary_insurance_id",
            "tertiary_insurance", "tertiary_insurance_id"
        ]
        filled_insurance_fields = sum(1 for field in insurance_fields if insurance_data.get(field, "").strip())
        insurance_confidence = filled_insurance_fields / len(insurance_fields) if insurance_fields else 0.0

        # Calculate DME confidence
        dme_id = dme_data.get("dme_id", "").strip()
        items = dme_data.get("items", [])
        if dme_id and items:
            dme_confidence = 1.0
        elif dme_id or items:
            dme_confidence = 0.5

        # Combine confidences (weighted average or simple average)
        overall_confidence = (patient_confidence + doctor_confidence + insurance_confidence + dme_confidence) / 4.0

        return {
            "overall": overall_confidence,
            "patient": patient_confidence,
            "doctor": doctor_confidence,
            "insurance": insurance_confidence,
            "dme": dme_confidence
        }

    def _create_final_result(self, pdf_path: Path, structured_results: List[Dict[str, Any]],
                           extraction_result: Dict[str, Any], processing_time: float,
                           successful_pages: int, total_pages: int) -> Dict[str, Any]:
        """
        Create the final result dictionary.

        Args:
            pdf_path: Path to the processed PDF
            structured_results: List of page processing results
            extraction_result: Original FileManager extraction result
            processing_time: Total processing time in seconds
            successful_pages: Number of successfully processed pages
            total_pages: Total number of pages

        Returns:
            Complete result dictionary
        """
        # Create document summary
        document_summary = self._create_document_summary(structured_results)

        return {
            "model_name": self.model_name,
            "processing_timestamp": datetime.now().isoformat(),
            "pdf_path": str(pdf_path),
            "pdf_name": pdf_path.name,
            "total_pages": total_pages,
            "successful_pages": successful_pages,
            "processing_time_seconds": round(processing_time, 2),
            "extraction_method": extraction_result.get("extraction", {}).get("method", "ocr"),
            "pages": structured_results,
            "document_summary": document_summary,
            "metadata": {
                "file_info": extraction_result.get("metadata", {}).get("file_info", {}),
                "extraction_confidence": round(sum(page.get("confidence_scores", {}).get("overall", 0)
                                                 for page in structured_results if page.get("success")) / successful_pages, 2) if successful_pages else 0.0
            }
        }

    def _create_document_summary(self, structured_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a document-level summary of the extraction results.

        Args:
            structured_results: List of page processing results

        Returns:
            Document summary dictionary
        """
        # Aggregate patient, doctor, insurance, and DME information from all pages
        all_patient_data = []
        all_doctor_data = []
        all_insurance_data = []
        all_dme_data = []

        for page_result in structured_results:
            if page_result.get("success", False):
                structured_data = page_result.get("structured_data", {})
                all_patient_data.append(structured_data.get("patient", {}))
                all_doctor_data.append(structured_data.get("doctor", {}))
                all_insurance_data.append(structured_data.get("insurance", {}))
                all_dme_data.append(structured_data.get("dme", {}))

        # Select best patient data (prioritize complete information)
        best_patient_data = {}
        if all_patient_data:
            best_patient_data = max(all_patient_data,
                                   key=lambda x: sum(1 for v in x.values() if v.strip()))

        # Get unique NPI numbers
        unique_npis = list(set(data.get("npi") for data in all_doctor_data if data.get("npi", "").strip()))
        doctor_npi = unique_npis[0] if unique_npis else ""

        # Consolidate insurance information
        consolidated_insurance = {
            "primary_insurance": "",
            "primary_insurance_id": "",
            "secondary_insurance": "",
            "secondary_insurance_id": "",
            "tertiary_insurance": "",
            "tertiary_insurance_id": ""
        }
        for data in all_insurance_data:
            for key, value in data.items():
                if value and not consolidated_insurance.get(key):
                    consolidated_insurance[key] = value

        # Consolidate DME information
        consolidated_dme = {"dme_id": "", "items": []}
        dme_items_set = set() # Use a set to avoid duplicate items

        for data in all_dme_data:
            if data.get("dme_id") and not consolidated_dme["dme_id"]:
                consolidated_dme["dme_id"] = data["dme_id"]

            for item in data.get("items", []):
                item_tuple = (item.get("item_name", ""), item.get("item_quantity", ""))
                if item_tuple not in dme_items_set:
                    consolidated_dme["items"].append(item)
                    dme_items_set.add(item_tuple)

        # Calculate overall confidence
        confidences = [page.get("confidence_scores", {}).get("overall", 0.0)
                       for page in structured_results if page.get("success")]
        overall_extraction_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

        return {
            "patient_summary": best_patient_data,
            "doctor_npi": doctor_npi,
            "insurance_summary": consolidated_insurance,
            "dme_summary": consolidated_dme,
            "extraction_confidence": overall_extraction_confidence,
            "document_type": "medical_form"
        }

    def _create_error_result(self, pdf_path: Path, error_message: str, processing_time: float) -> Dict[str, Any]:
        """
        Create an error result dictionary.

        Args:
            pdf_path: Path to the PDF that failed
            error_message: Error description
            processing_time: Processing time before failure

        Returns:
            Error result dictionary
        """
        return {
            "model_name": self.model_name,
            "processing_timestamp": datetime.now().isoformat(),
            "pdf_path": str(pdf_path),
            "pdf_name": pdf_path.name,
            "success": False,
            "error": error_message,
            "processing_time_seconds": round(processing_time, 2),
            "pages": [],
            "document_summary": {}
        }

    def _get_cached_data(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Get OCR data from cache or extract if not available.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing extracted OCR data
        """
        # Import DataExtractor here to avoid circular imports
        try:
            from structure_testing.extract_data import DataExtractor
        except ImportError:
            from extract_data import DataExtractor

        extractor = DataExtractor()
        cache_path = extractor.get_cached_data_path(pdf_path)

        if cache_path.exists():
            self.logger.info(f"Using cached OCR data: {cache_path}")
            print(f"[CACHE] Using cached OCR data from previous extraction")
            return extractor.load_extracted_data(cache_path)
        else:
            print(f"[EXTRACT] No cached data found, extracting fresh OCR data...")
            cache_path = extractor.extract_pdf_data(pdf_path)
            return extractor.load_extracted_data(cache_path)

    def _extract_fresh_data(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract fresh OCR data from PDF (ignoring cache).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing extracted OCR data
        """
        try:
            from structure_testing.extract_data import DataExtractor
        except ImportError:
            from extract_data import DataExtractor

        extractor = DataExtractor()
        cache_path = extractor.extract_pdf_data(pdf_path, force_reextract=True)
        return extractor.load_extracted_data(cache_path)

    def _save_results(self, result: Dict[str, Any]) -> None:
        """
        Save the extraction results to a JSON file.

        Args:
            result: Result dictionary to save
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(exist_ok=True)

            # Generate output filename
            pdf_name = Path(result["pdf_name"]).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{pdf_name}_{self.model_name}_{timestamp}.json"
            output_path = output_dir / output_filename

            # Save result to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Results saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def run_interactive(self) -> None:
        """
        Run the model interactively, asking user to choose from extracted data files.
        """
        print(f"\n=== {self.model_name} Interactive Mode ===")
        print("Choose an extracted data file to process, or 'quit' to exit.")

        while True:
            # Get list of available extracted data files
            extracted_files = self._get_available_extracted_files()

            if not extracted_files:
                print("\nNo extracted data files found!")
                print("Please run 'extract_data.py' first to extract OCR data from PDF files.")
                choice = input("\nPress Enter to refresh, or 'quit' to exit: ").strip()
                if choice.lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break
                continue

            # Display available files
            print(f"\nAvailable extracted data files:")
            print("-" * 60)
            for i, file_info in enumerate(extracted_files, 1):
                print(f"{i}. {file_info['name']}")
                print(f"   PDF: {file_info['pdf_name']}")
                print(f"   Pages: {file_info['pages']}")
                print(f"   Extracted: {file_info['timestamp']}")
                if i < len(extracted_files):
                    print()
            print("-" * 60)

            # Get user choice
            choice_input = input(f"\nEnter file number (1-{len(extracted_files)}) or 'quit': ").strip()

            if choice_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            try:
                choice = int(choice_input)
                if choice < 1 or choice > len(extracted_files):
                    print(f"Invalid choice. Please enter a number between 1 and {len(extracted_files)}.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            # Process the chosen file
            selected_file = extracted_files[choice - 1]
            print(f"\nProcessing {selected_file['name']} with {self.model_name}...")

            try:
                result = self._process_extracted_data(selected_file['path'])

                if result.get("success", True):
                    print(f"[OK] Successfully processed {result.get('successful_pages', 0)}/{result.get('total_pages', 0)} pages")
                    print(f"[TIME] Processing time: {result.get('processing_time_seconds', 0):.2f} seconds")
                    print(f"[SAVE] Results saved to output directory")

                    # Show summary
                    summary = result.get("document_summary", {})
                    patient = summary.get("patient_summary", {})
                    if patient.get("full_name"):
                        print(f"[PATIENT] Name: {patient['full_name']}")
                    if patient.get("dob"):
                        print(f"[PATIENT] DOB: {patient['dob']}")
                    if summary.get("doctor_npi"):
                        print(f"[DOCTOR] NPI: {summary['doctor_npi']}")
                    if summary.get("insurance_summary", {}).get("primary_insurance"):
                        print(f"[INSURANCE] Primary: {summary['insurance_summary']['primary_insurance']}")
                    if summary.get("dme_summary", {}).get("dme_id"):
                        print(f"[DME] ID: {summary['dme_summary']['dme_id']}")
                        print(f"[DME] Items: {len(summary['dme_summary'].get('items', []))}")
                else:
                    print(f"[FAIL] Processing failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"[ERROR] {e}")

    def _get_available_extracted_files(self) -> List[Dict[str, Any]]:
        """
        Get list of available extracted data files.

        Returns:
            List of dictionaries containing file information
        """
        try:
            from structure_testing.extract_data import DataExtractor
        except ImportError:
            from extract_data import DataExtractor

        extractor = DataExtractor()
        cache_files = list(extractor.cache_dir.glob("*.json"))

        extracted_files = []
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                extraction_info = data.get("extraction_info", {})
                extracted_files.append({
                    'name': cache_file.name,
                    'path': cache_file,
                    'pdf_name': extraction_info.get("pdf_name", "Unknown"),
                    'pages': extraction_info.get("total_pages", 0),
                    'successful_pages': extraction_info.get("successful_pages", 0),
                    'timestamp': extraction_info.get("extraction_timestamp", "Unknown")
                })
            except Exception as e:
                self.logger.warning(f"Error reading cache file {cache_file}: {e}")
                continue

        # Sort by timestamp (newest first)
        extracted_files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return extracted_files

    def _process_extracted_data(self, data_path: Path) -> Dict[str, Any]:
        """
        Process pre-extracted OCR data with the model using combined text for better context.

        Args:
            data_path: Path to the extracted data JSON file

        Returns:
            Dictionary containing the complete extraction results
        """
        start_time = time.time()

        self.logger.info(f"Processing extracted data: {data_path}")

        try:
            # Load extracted data
            try:
                from structure_testing.extract_data import DataExtractor
            except ImportError:
                from extract_data import DataExtractor

            extractor = DataExtractor()
            extracted_data = extractor.load_extracted_data(data_path)

            # Process all pages together for better context
            document_result = self.process_entire_document(extracted_data)

            # Create final result
            final_result = self._create_document_final_result(
                Path(extracted_data.get("extraction_info", {}).get("pdf_path", data_path.name)),
                document_result,
                extracted_data,
                time.time() - start_time
            )

            # Add success flag
            final_result["success"] = True
            final_result["data_file"] = str(data_path.name)

            # Save results
            self._save_results(final_result)

            self.logger.info(f"Successfully processed document with {len(extracted_data.get('pages', []))} pages")
            return final_result

        except Exception as e:
            self.logger.error(f"Error processing extracted data {data_path}: {e}")
            return self._create_error_result(data_path, str(e), time.time() - start_time)

    def process_entire_document(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the entire document as one combined text for better context.

        Args:
            extracted_data: Dictionary containing all page data

        Returns:
            Dictionary containing the structured extraction result
        """
        pages = extracted_data.get('pages', [])

        if not pages:
            return {
                "success": False,
                "error": "No pages found in extracted data",
                "structured_data": self._get_empty_structure()
            }

        # Combine all page text with page separators for context
        combined_text = ""
        for i, page_data in enumerate(pages, 1):
            page_text = page_data.get('text', '').strip()
            if page_text:
                combined_text += f"\n--- PAGE {i} ---\n{page_text}\n"

        if not combined_text.strip():
            return {
                "success": False,
                "error": "No text content found in any pages",
                "structured_data": self._get_empty_structure()
            }

        print(f"[CONTEXT] Processing {len(pages)} pages as single document ({len(combined_text)} characters)")

        try:
            # Generate comprehensive extraction prompt
            prompt = self.get_document_extraction_prompt(combined_text)
            messages = [{"role": "user", "content": prompt}]

            # Call the model API
            api_start = time.time()
            response = self.call_api(messages, temperature=0.1)
            api_time = time.time() - api_start

            # Extract structured data from response
            content = self._extract_content_from_response(response)
            structured_data = self.extract_structure(content)

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence(structured_data, content)

            return {
                "success": True,
                "combined_text": combined_text,
                "structured_data": structured_data,
                "confidence_scores": confidence_scores,
                "processing_metadata": {
                    "api_response_time": api_time,
                    "tokens_used": response.get("usage", {}).get("total_tokens", 0),
                    "pages_processed": len(pages),
                    "total_characters": len(combined_text),
                    "model_used": self.model_name
                }
            }

        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return {
                "success": False,
                "error": str(e),
                "combined_text": combined_text,
                "structured_data": self._get_empty_structure()
            }

    def get_document_extraction_prompt(self, combined_text: str) -> str:
        """
        Generate a comprehensive extraction prompt for the entire document.

        Args:
            combined_text: Combined text from all pages of the document

        Returns:
            Formatted prompt string for comprehensive extraction
        """
        prompt = f"""
You are a medical document extractor. This document contains multiple pages that ALL belong to the SAME patient. Extract the following comprehensive information from the ENTIRE document and return it as a single JSON object.

Required fields to extract:
Patient Information:
- full_name: Complete patient name (may appear on different pages)
- first_name: Patient's first/given name
- middle_name: Patient's middle name/initial
- last_name: Patient's last/family name
- dob: Date of birth (MM/DD/YYYY format preferred)
- address_full: Complete street address
- city: City name
- state: Full state name
- country: Country name
- postal_code: ZIP/Postal code
- state_code: 2-letter state abbreviation
- phone: Primary phone number
- mobile_landline: Secondary/landline phone
- email: Email address
- fax: Fax number
- account_number: Patient account number

Doctor Information:
- npi: National Provider Identifier (10-digit number)

Insurance Information:
- primary_insurance: Name of primary insurance provider
- primary_insurance_id: Primary insurance policy/member ID
- secondary_insurance: Name of secondary insurance provider (if any)
- secondary_insurance_id: Secondary insurance policy/member ID (if any)
- tertiary_insurance: Name of tertiary insurance provider (if any)
- tertiary_insurance_id: Tertiary insurance policy/member ID (if any)

DME (Durable Medical Equipment) Information:
- dme_id: DME supplier or order ID
- items: List of ALL DME items found across all pages, each containing:
  - item_name: Name/description of the DME item
  - item_quantity: Quantity of the item

IMPORTANT INSTRUCTIONS:
1. This is a SINGLE patient document - look across ALL pages for complete information
2. Combine information from different pages to create a complete patient profile
3. For DME items, collect ALL items mentioned across all pages into a single items array
4. If the same field appears on multiple pages, use the most complete/clear version
5. If a field is not found anywhere in the document, use an empty string ("")
6. For items array, if no items found, use an empty array []
7. Normalize phone numbers to standard format (e.g., "123-456-7890")
8. Ensure NPI is exactly 10 digits if found
9. Return only valid JSON object with the structure below

Expected JSON structure:
{{
    "patient": {{
        "full_name": "",
        "first_name": "",
        "middle_name": "",
        "last_name": "",
        "dob": "",
        "address_full": "",
        "city": "",
        "state": "",
        "country": "",
        "postal_code": "",
        "state_code": "",
        "phone": "",
        "mobile_landline": "",
        "email": "",
        "fax": "",
        "account_number": ""
    }},
    "doctor": {{
        "npi": ""
    }},
    "insurance": {{
        "primary_insurance": "",
        "primary_insurance_id": "",
        "secondary_insurance": "",
        "secondary_insurance_id": "",
        "tertiary_insurance": "",
        "tertiary_insurance_id": ""
    }},
    "dme": {{
        "dme_id": "",
        "items": []
    }}
}}

Complete document text to extract from:
{combined_text}

Extract the comprehensive information from all pages and return the JSON object:
"""
        return prompt.strip()

    def _get_empty_structure(self) -> Dict[str, Any]:
        """
        Get empty structure for cases where extraction fails.

        Returns:
            Empty structured data template
        """
        return {
            "patient": {
                "full_name": "",
                "first_name": "",
                "middle_name": "",
                "last_name": "",
                "dob": "",
                "address_full": "",
                "city": "",
                "state": "",
                "country": "",
                "postal_code": "",
                "state_code": "",
                "phone": "",
                "mobile_landline": "",
                "email": "",
                "fax": "",
                "account_number": ""
            },
            "doctor": {
                "npi": ""
            },
            "insurance": {
                "primary_insurance": "",
                "primary_insurance_id": "",
                "secondary_insurance": "",
                "secondary_insurance_id": "",
                "tertiary_insurance": "",
                "tertiary_insurance_id": ""
            },
            "dme": {
                "dme_id": "",
                "items": []
            }
        }

    def _create_document_final_result(self, pdf_path: Path, document_result: Dict[str, Any],
                                    extracted_data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """
        Create the final result dictionary for document-level processing.

        Args:
            pdf_path: Path to the processed PDF
            document_result: Result from document processing
            extracted_data: Original extracted OCR data
            processing_time: Total processing time in seconds

        Returns:
            Complete result dictionary
        """
        pages = extracted_data.get('pages', [])

        # Create document summary from the single comprehensive result
        structured_data = document_result.get("structured_data", self._get_empty_structure())

        document_summary = {
            "patient_summary": structured_data.get("patient", {}),
            "doctor_npi": structured_data.get("doctor", {}).get("npi", ""),
            "insurance_summary": structured_data.get("insurance", {}),
            "dme_summary": structured_data.get("dme", {}),
            "extraction_confidence": document_result.get("confidence_scores", {}).get("overall", 0.0),
            "document_type": "medical_form",
            "processing_approach": "combined_document"
        }

        return {
            "model_name": self.model_name,
            "processing_timestamp": datetime.now().isoformat(),
            "pdf_path": str(pdf_path),
            "pdf_name": pdf_path.name,
            "total_pages": len(pages),
            "successful_pages": len(pages) if document_result.get("success", False) else 0,
            "processing_time_seconds": round(processing_time, 2),
            "extraction_method": extracted_data.get("extraction_info", {}).get("extraction_method", "ocr"),
            "processing_approach": "combined_document",  # New field indicating approach
            "document_result": document_result,  # Single document result instead of page-by-page
            "pages": pages,  # Keep original page data for reference
            "document_summary": document_summary,
            "metadata": {
                "file_info": extracted_data.get("extraction_info", {}),
                "extraction_confidence": document_result.get("confidence_scores", {}).get("overall", 0.0),
                "total_characters": document_result.get("processing_metadata", {}).get("total_characters", 0),
                "api_response_time": document_result.get("processing_metadata", {}).get("api_response_time", 0)
            }
        }