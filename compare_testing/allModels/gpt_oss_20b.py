"""
OpenAI GPT-OSS-20B Patient Identification Model

Uses the OpenAI GPT-OSS-20B model via OpenRouter to identify patients
and group pages within multi-patient documents.
"""

import sys
import os
import json
from typing import Dict, Any, List
import time
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from base_patient_identifier import BasePatientIdentifier
except ImportError:
    # Fallback for different import patterns
    sys.path.append(str(Path(__file__).parent.parent))
    from allModels.base_patient_identifier import BasePatientIdentifier


class GPT_OSS_20B_PatientIdentifier(BasePatientIdentifier):
    """OpenAI GPT-OSS-20B patient identification model via OpenRouter"""

    def __init__(self):
        """Initialize the GPT-OSS-20B patient identifier."""
        super().__init__("GPT-OSS-20B_Patient")

        # Model configuration
        self.model_name = "GPT-OSS-20B_Patient"
        self.model_id = "openai/gpt-4o-2024-08-06"  # Use GPT-4o for patient identification
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Load API configuration
        self.api_key = self.config.get_openrouter_key()
        if not self.api_key:
            raise ValueError("OpenRouter API key not found in configuration")

        # Algorithm-specific configuration
        self.temperature = self.algorithm_config.get('temperature', 0.1)
        self.max_tokens = self.algorithm_config.get('max_tokens', 4000)
        self.timeout = self.algorithm_config.get('timeout', 30)

        self.logger.info(f"GPT-OSS-20B Patient Identifier initialized with model: {self.model_id}")

    def call_api(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Make API call to OpenAI GPT-OSS-20B via OpenRouter.

        Args:
            messages: List of message dictionaries for the API call
            **kwargs: Additional keyword arguments (temperature, max_tokens, etc.)

        Returns:
            Dictionary containing the API response

        Raises:
            Exception: If API call fails
        """
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "PatXtract Patient Identification - GPT-OSS-20B",
            "Content-Type": "application/json"
        }

        # Prepare request payload
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": 0.9,
            "stream": False
        }

        self.logger.debug(f"Making API call to {self.api_url}")

        try:
            import requests
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            # Check for API-specific errors
            if "error" in result:
                error_msg = result["error"].get("message", "Unknown API error")
                error_type = result["error"].get("type", "unknown")
                self.logger.error(f"API Error ({error_type}): {error_msg}")
                raise Exception(f"API Error: {error_msg}")

            self.logger.debug(f"API call successful, tokens used: {result.get('usage', {}).get('total_tokens', 'unknown')}")
            return result

        except requests.exceptions.Timeout:
            raise Exception("API request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to API endpoint")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise Exception("Rate limit exceeded - please try again later")
            elif e.response.status_code == 401:
                raise Exception("Invalid API key or unauthorized access")
            elif e.response.status_code == 400:
                raise Exception("Bad request - invalid parameters")
            else:
                raise Exception(f"HTTP Error {e.response.status_code}: {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}")

    def create_patient_identification_prompt(self, pages_text: str, total_pages: int) -> List[Dict[str, str]]:
        """
        Create a comprehensive prompt for patient identification and page grouping.

        Args:
            pages_text: Combined text from all pages
            total_pages: Total number of pages in the document

        Returns:
            List of message dictionaries for the API call
        """
        system_prompt = """You are an expert medical document analyst specializing in patient identification and page grouping. Your task is to:

1. Identify all unique patients mentioned in the document
2. Determine which pages belong to each patient
3. Handle non-consecutive page assignments properly
4. Group pages into continuous entries for each patient

KEY REQUIREMENTS:
- Each patient should have a unique patient_id (patient_001, patient_002, etc.)
- Track separate entries when the same patient appears on non-consecutive pages
- Provide confidence scores for each identification
- Handle OCR errors and formatting variations

OUTPUT FORMAT:
Return a JSON object with the following structure:

{
  "patient_groups": {
    "patient_001": {
      "patient_info": {
        "full_name": "Complete patient name",
        "first_name": "First name",
        "last_name": "Last name",
        "middle_name": "Middle name/initial",
        "dob": "YYYY-MM-DD format",
        "address_full": "Full address",
        "city": "City",
        "state": "State",
        "postal_code": "ZIP code",
        "phone": "Primary phone number"
      },
      "entries": [
        {
          "entry_id": "patient_001_entry_1",
          "pages": [1, 2],
          "page_range": "1-2",
          "confidence": 0.95,
          "supporting_data": {
            "evidence_summary": "Brief summary of why pages belong to this patient"
          }
        }
      ],
      "total_entries": 1,
      "total_pages": 2,
      "all_pages": [1, 2],
      "group_confidence": 0.95
    }
  },
  "unassigned_pages": {
    "pages": [5],
    "reasons": ["No patient information found", "Insufficient data"]
  }
}

CRITICAL RULES:
1. Create separate entries for non-consecutive appearances of the same patient
2. Use high confidence scores (0.8-1.0) for clear matches
3. Include all available patient information in patient_info
4. Handle missing fields gracefully (use empty strings)
5. Group pages by examining name, DOB, address, and other identifying information
6. Be careful with similar names - look for multiple identifiers to confirm matches

NON-CONSECUTIVE PAGE HANDLING EXAMPLE:
If pages 1-2 belong to Patient A, page 3 belongs to Patient B, and page 4 belongs to Patient A again:
- Patient A should have 2 entries: pages [1,2] and [4]
- Patient B should have 1 entry: pages [3]
- Track total_entries = 2 for Patient A"""

        user_prompt = f"""Please analyze the following multi-page medical document and identify all patients, grouping pages appropriately.

Document Information:
- Total pages: {total_pages}
- Page numbering starts from 1

Document Text (pages are separated by "=== PAGE X ==="):
{pages_text}

Please analyze each page, identify all patients, and group pages according to the patient identification rules provided in the system prompt. Focus on accuracy in patient matching and proper handling of non-consecutive page assignments.

Return ONLY the JSON response, no additional text or explanations."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def identify_patients(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify patients across pages using GPT-OSS-20B analysis with chunking support.

        Args:
            pages: List of page data dictionaries

        Returns:
            Dictionary with patient grouping results
        """
        self.logger.info(f"Starting GPT-OSS-20B patient identification for {len(pages)} pages")

        # Use chunking for large documents
        return self.call_model_with_chunking(
            pages=pages,
            create_prompt_func=self.create_patient_identification_prompt,
            call_api_func=self.call_api,
            extract_structure_func=self.extract_structure,
            create_empty_result_func=self.create_empty_result
        )

    def prepare_pages_text(self, pages: List[Dict[str, Any]]) -> str:
        """
        Prepare the pages text for API analysis.

        Args:
            pages: List of page data dictionaries

        Returns:
            Formatted text string for analysis
        """
        pages_text_parts = []

        for page in pages:
            page_number = page['page_number']
            text = page['text']
            confidence = page.get('confidence', 0.0)

            pages_text_parts.append(f"=== PAGE {page_number} (Confidence: {confidence:.2f}) ===")
            pages_text_parts.append(text.strip())
            pages_text_parts.append("")  # Empty line for separation

        return "\n".join(pages_text_parts)

    def extract_structure(self, content: str) -> Dict[str, Any]:
        """
        Extract structured patient identification data from model response.

        Args:
            content: Raw text content from the model response

        Returns:
            Dictionary containing structured patient grouping information
        """
        try:
            # Try to extract JSON from the content
            import re

            # Look for JSON object in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group())
                    return extracted_data
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON from model response: {e}")

            # If JSON parsing fails, try to extract information manually
            self.logger.warning("Failed to extract JSON, attempting manual extraction")
            return self.manual_patient_extraction(content)

        except Exception as e:
            self.logger.error(f"Error extracting structure: {e}")
            return {}

    def manual_patient_extraction(self, content: str) -> Dict[str, Any]:
        """
        Fallback method to manually extract patient information from text.

        Args:
            content: Model response text

        Returns:
            Dictionary with basic patient grouping structure
        """
        # This is a fallback - return empty structure
        self.logger.warning("Manual extraction not implemented, returning empty structure")
        return {
            "patient_groups": {},
            "unassigned_pages": {
                "pages": [],
                "reasons": ["JSON parsing failed"]
            }
        }

    def calculate_confidence(self, patient_data: Dict[str, Any], pages_data: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for patient identification.

        Args:
            patient_data: Patient identification data
            pages_data: List of page data used for identification

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # If confidence is already provided in the data, use it
        if 'group_confidence' in patient_data:
            return float(patient_data['group_confidence'])

        # Otherwise, calculate based on field completeness
        patient_info = patient_data.get('patient_info', {})
        required_fields = ['full_name', 'dob']
        filled_fields = sum(1 for field in required_fields if patient_info.get(field, '').strip())

        base_confidence = filled_fields / len(required_fields)

        # Factor in OCR confidence from pages
        if pages_data:
            ocr_confidence = sum(page.get('confidence', 0.0) for page in pages_data) / len(pages_data)
            final_confidence = (base_confidence * 0.7) + (ocr_confidence * 0.3)
        else:
            final_confidence = base_confidence

        return min(1.0, max(0.0, final_confidence))

    def create_empty_result(self, total_pages: int, processing_time: float) -> Dict[str, Any]:
        """
        Create an empty result when no patient information is found.

        Args:
            total_pages: Total number of pages processed
            processing_time: Processing time in seconds

        Returns:
            Empty result dictionary
        """
        return {
            'algorithm_name': self.algorithm_name,
            'processing_timestamp': self.get_current_timestamp(),
            'total_pages': total_pages,
            'processing_time_seconds': processing_time,
            'model_used': self.model_id,
            'patient_groups': {},
            'unassigned_pages': {
                'pages': list(range(1, total_pages + 1)),
                'reasons': ['No patient information identified by GPT model']
            },
            'summary': {
                'total_patients': 0,
                'total_entries': 0,
                'assigned_pages': 0,
                'unassigned_pages': total_pages,
                'average_confidence': 0.0
            }
        }


def main():
    """Main function to run the GPT-OSS-20B patient identifier"""
    try:
        identifier = GPT_OSS_20B_PatientIdentifier()
        print(f"{identifier.algorithm_name} ready for use")
        print(f"Model: {identifier.model_id}")
        print(f"API Endpoint: {identifier.api_url}")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())