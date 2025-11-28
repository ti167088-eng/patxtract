"""
Base Patient Identifier Module

Provides the abstract base class for all patient identification algorithms
in the compare_testing module. All patient identifier implementations
must inherit from this class and implement the required methods.
"""

import abc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.security import SecureConfig
except ImportError:
    from structure_testing.config.security import SecureConfig


class BasePatientIdentifier(abc.ABC):
    """
    Abstract base class for all patient identification algorithms.

    This class provides the common interface and functionality for identifying
    patients across PDF pages and grouping them appropriately. All patient
    identification algorithms must inherit from this class.
    """

    def __init__(self, algorithm_name: str):
        """
        Initialize the patient identifier.

        Args:
            algorithm_name: Name of the algorithm for logging and output
        """
        self.algorithm_name = algorithm_name
        self.logger = logging.getLogger(f"{__name__}.{algorithm_name}")

        # Configuration management
        self.config = SecureConfig()
        self.algorithm_config = self.config.get_model_config(algorithm_name.lower())

        # Initialize output directory
        script_dir = Path(__file__).parent.parent
        self.output_dir = script_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

        self.logger.info(f"Initialized {algorithm_name} patient identifier")

        # Validate that all required abstract methods are implemented
        try:
            self.validate_implementation()
        except NotImplementedError as e:
            self.logger.error(f"Invalid implementation: {e}")
            raise

    def validate_implementation(self) -> bool:
        """
        Validate that all required abstract methods are implemented.

        Returns:
            True if valid, raises exception if invalid
        """
        required_methods = [
            'identify_patients',
            'calculate_confidence',
            'prepare_pages_text'
        ]

        for method_name in required_methods:
            if not hasattr(self, method_name):
                raise NotImplementedError(
                    f"Required method '{method_name}' not implemented in {self.__class__.__name__}"
                )

            if not callable(getattr(self, method_name)):
                raise NotImplementedError(
                    f"Method '{method_name}' is not callable in {self.__class__.__name__}"
                )

        return True

    @abc.abstractmethod
    def identify_patients(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify patients across pages and group them.

        This is the core method that each algorithm must implement to analyze
        the extracted PDF page data and identify patient groupings.

        Args:
            pages: List of page data dictionaries with keys:
                - page_number: int
                - text: str (OCR extracted text)
                - confidence: float (OCR confidence)
                - text_length: int
                - success: bool

        Returns:
            Dictionary containing patient grouping results with structure:
            {
                "algorithm_name": str,
                "processing_timestamp": str,
                "source_data_file": str,
                "total_pages": int,
                "processing_time_seconds": float,
                "patient_groups": {
                    "patient_001": {
                        "patient_info": {...},
                        "entries": [
                            {
                                "entry_id": str,
                                "pages": List[int],
                                "page_range": str,
                                "confidence": float,
                                "supporting_data": {...}
                            }
                        ],
                        "total_entries": int,
                        "total_pages": int,
                        "all_pages": List[int],
                        "group_confidence": float
                    }
                },
                "unassigned_pages": {
                    "pages": List[int],
                    "reasons": List[str]
                },
                "summary": {
                    "total_patients": int,
                    "total_entries": int,
                    "assigned_pages": int,
                    "unassigned_pages": int,
                    "average_confidence": float
                }
            }
        """
        pass

    @abc.abstractmethod
    def calculate_confidence(self, patient_data: Dict[str, Any], pages_data: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for patient identification.

        Args:
            patient_data: Patient identification data dictionary
            pages_data: List of page data used for this patient identification

        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass

    @abc.abstractmethod
    def prepare_pages_text(self, pages: List[Dict[str, Any]]) -> str:
        """
        Prepare pages text for model analysis.

        Args:
            pages: List of page data dictionaries with keys:
                - page_number: int
                - text: str
                - confidence: float (optional)

        Returns:
            Formatted text string for model processing
        """
        pass

    def create_patient_signature(self, patient_info: Dict[str, Any]) -> str:
        """
        Create a unique signature for patient identification.

        Default implementation uses name + DOB + address. Override this method
        in subclasses to implement algorithm-specific signature creation.

        Args:
            patient_info: Dictionary containing patient information

        Returns:
            Unique signature string for patient identification
        """
        name = self.normalize_name(patient_info.get('full_name', ''))
        dob = self.normalize_dob(patient_info.get('dob', ''))
        address = self.normalize_address(patient_info.get('address_full', ''))

        return f"{name}|{dob}|{address}"

    def normalize_name(self, name: str) -> str:
        """
        Normalize patient name for consistent matching.

        Args:
            name: Raw patient name string

        Returns:
            Normalized name string
        """
        if not name:
            return ""

        # Remove extra whitespace and convert to lowercase
        normalized = ' '.join(name.lower().split())

        # Remove common title prefixes
        prefixes = ['mr.', 'mrs.', 'ms.', 'dr.', 'mr', 'mrs', 'ms', 'dr']
        for prefix in prefixes:
            if normalized.startswith(prefix + ' '):
                normalized = normalized[len(prefix + ' '):]
                break

        return normalized.strip()

    def normalize_dob(self, dob: str) -> str:
        """
        Normalize date of birth to YYYY-MM-DD format.

        Args:
            dob: Raw date of birth string

        Returns:
            Normalized date string in YYYY-MM-DD format
        """
        if not dob:
            return ""

        # Remove non-digit characters
        digits_only = ''.join(c for c in dob if c.isdigit())

        # Handle different date formats
        if len(digits_only) == 8:
            # MMDDYYYY format
            mm = digits_only[0:2]
            dd = digits_only[2:4]
            yyyy = digits_only[4:8]

            # Validate month and day
            if 1 <= int(mm) <= 12 and 1 <= int(dd) <= 31:
                return f"{yyyy}-{mm}-{dd}"

        return dob.lower().strip()

    def normalize_address(self, address: str) -> str:
        """
        Normalize address for consistent matching.

        Args:
            address: Raw address string

        Returns:
            Normalized address string
        """
        if not address:
            return ""

        # Remove extra whitespace and newlines, convert to lowercase
        normalized = ' '.join(address.lower().split())

        # Common address abbreviations
        abbreviations = {
            'street': 'st',
            'avenue': 'ave',
            'boulevard': 'blvd',
            'lane': 'ln',
            'drive': 'dr',
            'road': 'rd',
            'court': 'ct',
            'suite': 'ste',
            'apartment': 'apt'
        }

        # Replace common abbreviations
        for full_form, abbrev in abbreviations.items():
            normalized = normalized.replace(full_form, abbrev)

        return normalized.strip()

    def create_patient_id(self, patient_groups: Dict[str, Any]) -> str:
        """
        Create a unique patient ID.

        Args:
            patient_groups: Existing patient groups to avoid ID conflicts

        Returns:
            Unique patient ID string
        """
        if not patient_groups:
            return "patient_001"

        # Find the highest existing patient number
        max_num = 0
        for patient_id in patient_groups.keys():
            if patient_id.startswith("patient_"):
                try:
                    num = int(patient_id.split("_")[1])
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    continue

        return f"patient_{max_num + 1:03d}"

    def create_entry_id(self, patient_id: str, entry_num: int) -> str:
        """
        Create a unique entry ID for a patient.

        Args:
            patient_id: Patient ID
            entry_num: Entry number for this patient

        Returns:
            Unique entry ID string
        """
        return f"{patient_id}_entry_{entry_num}"

    def save_results(self, results: Dict[str, Any], source_file: str) -> Path:
        """
        Save algorithm results to output file.

        Args:
            results: Algorithm results dictionary
            source_file: Source data file name for reference

        Returns:
            Path to the saved results file
        """
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.algorithm_name}_{timestamp}.json"
        output_path = self.output_dir / output_filename

        # Prepare data for saving
        save_data = {
            **results,
            "source_data_file": source_file,
            "algorithm_name": self.algorithm_name
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Results saved to: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise

    def validate_pages_data(self, pages: List[Dict[str, Any]]) -> bool:
        """
        Validate that pages data has the required structure.

        Args:
            pages: List of page data dictionaries

        Returns:
            True if data is valid, False otherwise
        """
        if not pages or not isinstance(pages, list):
            self.logger.error("Pages data must be a non-empty list")
            return False

        required_fields = ['page_number', 'text', 'confidence', 'success']

        for i, page in enumerate(pages):
            if not isinstance(page, dict):
                self.logger.error(f"Page {i} must be a dictionary")
                return False

            for field in required_fields:
                if field not in page:
                    self.logger.error(f"Page {i} missing required field: {field}")
                    return False

        return True

    def get_processing_summary(self, patient_groups: Dict[str, Any], unassigned_pages: Dict[str, Any],
                              total_pages: int, processing_time: float) -> Dict[str, Any]:
        """
        Create processing summary from results.

        Args:
            patient_groups: Patient groups dictionary
            unassigned_pages: Unassigned pages dictionary
            total_pages: Total number of pages processed
            processing_time: Processing time in seconds

        Returns:
            Summary dictionary
        """
        total_patients = len(patient_groups)
        total_entries = sum(
            patient_data.get('total_entries', 1)
            for patient_data in patient_groups.values()
        )
        assigned_pages = sum(
            patient_data.get('total_pages', 0)
            for patient_data in patient_groups.values()
        )
        unassigned_count = len(unassigned_pages.get('pages', []))

        # Calculate average confidence
        all_confidences = [
            patient_data.get('group_confidence', 0.0)
            for patient_data in patient_groups.values()
        ]
        average_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        return {
            "total_patients": total_patients,
            "total_entries": total_entries,
            "assigned_pages": assigned_pages,
            "unassigned_pages": unassigned_count,
            "average_confidence": average_confidence,
            "processing_time_seconds": processing_time
        }

    def process_large_document_with_chunking(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process large documents using intelligent chunking.

        This method should be overridden by AI model implementations that need
        to handle documents larger than their context window.

        Args:
            pages: List of page data dictionaries

        Returns:
            Patient identification results with chunking applied
        """
        # Default implementation - override in AI models
        self.logger.info("Chunking not implemented for this algorithm, using standard processing")
        return self.identify_patients(pages)

    def call_model_with_chunking(self, pages: List[Dict[str, Any]],
                                create_prompt_func, call_api_func,
                                extract_structure_func, create_empty_result_func) -> Dict[str, Any]:
        """
        Generic method for AI models to handle chunking.

        Args:
            pages: List of page data dictionaries
            create_prompt_func: Function to create prompt from pages text
            call_api_func: Function to call the model API
            extract_structure_func: Function to extract structure from API response
            create_empty_result_func: Function to create empty result on failure

        Returns:
            Consolidated patient identification results
        """
        try:
            # Try relative imports first
            from ..utils.chunk_manager import ChunkManager
            from ..utils.token_estimator import TokenEstimator
        except ImportError:
            try:
                # Fallback to absolute imports
                from compare_testing.utils.chunk_manager import ChunkManager
                from compare_testing.utils.token_estimator import TokenEstimator
            except ImportError:
                # Final fallback - direct import with sys.path manipulation
                import sys
                from pathlib import Path
                utils_path = Path(__file__).parent.parent / "utils"
                if str(utils_path) not in sys.path:
                    sys.path.insert(0, str(utils_path))

                from chunk_manager import ChunkManager
                from token_estimator import TokenEstimator

        chunk_manager = ChunkManager(self.algorithm_name.lower())
        analysis = chunk_manager.analyze_document(pages)

        if not analysis['needs_chunking']:
            self.logger.info(f"Document fits in model context: {analysis['total_tokens']} tokens")
            print(f"📊 Document size: {analysis['total_tokens']:,} tokens (fits in single request)")
            return self.process_single_chunk(pages, create_prompt_func, call_api_func,
                                           extract_structure_func, create_empty_result_func)

        self.logger.info(f"Document requires chunking: {analysis['total_tokens']} tokens, "
                        f"splitting into {analysis['estimated_chunks']} chunks")

        # Add user-friendly progress reporting
        print(f"📊 Document size: {analysis['total_tokens']:,} tokens (exceeds model limit)")
        print(f"🔪 Splitting into {analysis['estimated_chunks']} chunks of ~{analysis.get('pages_per_chunk', 'N/A')} pages each")
        print(f"🔄 Processing chunks sequentially...")

        chunks, chunk_info = chunk_manager.create_chunks(pages)
        chunk_results = []

        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} pages)")

            chunk_result = self.process_single_chunk(chunk, create_prompt_func, call_api_func,
                                                  extract_structure_func, create_empty_result_func)

            if chunk_result:
                chunk_results.append(chunk_result)
            else:
                self.logger.warning(f"Chunk {i+1} failed to produce results")

        if not chunk_results:
            self.logger.error("All chunks failed to produce results")
            total_pages = len(pages)
            return create_empty_result_func(total_pages, 0)

        # Merge results from all chunks
        consolidated_result = chunk_manager.merge_chunk_results(chunk_results)

        # Add chunking metadata
        consolidated_result['chunking_metadata'] = {
            'total_chunks': len(chunks),
            'successful_chunks': len(chunk_results),
            'chunk_sizes': [len(chunk) for chunk in chunks],
            'total_tokens_processed': analysis['total_tokens'],
            'model_context_limit': analysis['model_context_limit']
        }

        self.logger.info(f"Successfully processed and merged {len(chunk_results)} chunks")
        return consolidated_result

    def process_single_chunk(self, pages: List[Dict[str, Any]],
                           create_prompt_func, call_api_func,
                           extract_structure_func, create_empty_result_func) -> Dict[str, Any]:
        """
        Process a single chunk of pages.

        Args:
            pages: List of page data dictionaries for the chunk
            create_prompt_func: Function to create prompt
            call_api_func: Function to call API
            extract_structure_func: Function to extract structure
            create_empty_result_func: Function to create empty result

        Returns:
            Patient identification results for the chunk
        """
        import time
        start_time = time.time()

        if not self.validate_pages_data(pages):
            raise ValueError("Invalid pages data format")

        # Prepare text for analysis
        try:
            from ..utils.token_estimator import TokenEstimator
        except ImportError:
            try:
                from compare_testing.utils.token_estimator import TokenEstimator
            except ImportError:
                import sys
                from pathlib import Path
                utils_path = Path(__file__).parent.parent / "utils"
                if str(utils_path) not in sys.path:
                    sys.path.insert(0, str(utils_path))
                from token_estimator import TokenEstimator

        token_estimator = TokenEstimator()

        # Prepare text for analysis with robust error handling
        try:
            if not hasattr(self, 'prepare_pages_text'):
                raise NotImplementedError(
                    f"prepare_pages_text method not implemented in {self.__class__.__name__}. "
                    "This method must be implemented by all patient identifier subclasses."
                )
            pages_text = self.prepare_pages_text(pages)
        except (NotImplementedError, AttributeError) as e:
            self.logger.error(f"Missing required method: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error preparing pages text: {e}")
            raise ValueError(f"Failed to prepare pages text: {e}")

        total_tokens = token_estimator.estimate_tokens_from_text(pages_text)

        self.logger.info(f"Processing chunk: {len(pages)} pages, ~{total_tokens} tokens")

        # Create prompt
        messages = create_prompt_func(pages_text, len(pages))

        try:
            # Call the API
            response = call_api_func(messages)

            # Extract and parse the response
            response_text = response['choices'][0]['message']['content']
            parsed_result = extract_structure_func(response_text)

            if not parsed_result:
                self.logger.error("Failed to parse patient identification response")
                processing_time = time.time() - start_time
                return create_empty_result_func(len(pages), processing_time)

            # Enhance the result with additional metadata
            processing_time = time.time() - start_time
            result = {
                'algorithm_name': self.algorithm_name,
                'processing_timestamp': self.get_current_timestamp(),
                'total_pages': len(pages),
                'processing_time_seconds': processing_time,
                'chunk_info': {
                    'is_chunk': True,
                    'chunk_tokens': total_tokens,
                    'chunk_pages': len(pages)
                },
                **parsed_result
            }

            # Calculate and add summary
            if 'summary' not in result:
                unassigned_pages = result.get('unassigned_pages', {}).get('pages', [])
                summary = self.get_processing_summary(
                    result.get('patient_groups', {}),
                    {'pages': unassigned_pages, 'reasons': []},
                    len(pages),
                    processing_time
                )
                result['summary'] = summary

            self.logger.info(f"Chunk processing completed: {result.get('summary', {}).get('total_patients', 0)} patients")
            return result

        except Exception as e:
            self.logger.error(f"Error in chunk processing: {e}")
            processing_time = time.time() - start_time
            return create_empty_result_func(len(pages), processing_time)

    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()