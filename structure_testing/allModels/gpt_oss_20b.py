"""
GPT-OSS-20B Model Implementation

Implementation of the OpenAI GPT-OSS-20B model via OpenRouter API
for structured medical document extraction.
"""

import json
import requests
import logging
import time
import re
from typing import Dict, Any, List, Optional

# Handle both direct execution and module imports
try:
    from .base_model import BaseModel
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from allModels.base_model import BaseModel

logger = logging.getLogger(__name__)

class GPT_OSS_20B_Model(BaseModel):
    """
    GPT-OSS-20B model implementation using OpenRouter API.

    This model uses the same model identifier as in the original test.py file
    for consistency with existing testing.
    """

    def __init__(self):
        """Initialize the GPT-OSS-20B model."""
        super().__init__(
            model_name="gpt-oss-20b",
            model_id="openai/gpt-oss-20b"
        )
        self.api_url = f"{self.base_url}/chat/completions"
        self.logger.info("GPT-OSS-20B model initialized")

    def call_api(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Make API call to GPT-OSS-20B via OpenRouter.

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
            "X-Title": "PatXtract Medical Document Processing",
            "Content-Type": "application/json"
        }

        # Prepare request payload
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.model_config.get("temperature", 0.1)),
            "max_tokens": kwargs.get("max_tokens", self.model_config.get("max_tokens", 4000)),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": False
        }

        self.logger.debug(f"Making API call to {self.api_url}")

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.model_config.get("timeout", 30)
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

    def extract_structure(self, content: str) -> Dict[str, Any]:
        """
        Extract structured data from GPT-OSS-20B response.

        Args:
            content: Raw text content from the model response

        Returns:
            Dictionary containing structured patient and doctor information
        """
        # Initialize the result structure
        result = {
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

        if not content or not content.strip():
            self.logger.warning("Empty content provided for structure extraction")
            return result

        try:
            # Try to extract JSON from the response
            json_content = self._extract_json_from_response(content)

            if json_content:
                parsed_data = json.loads(json_content)

                # Merge with result structure
                result = self._merge_structured_data(result, parsed_data)

            else:
                # Fallback to text-based extraction
                self.logger.warning("Could not extract JSON from response, using fallback parsing")
                result = self._extract_from_text(content, result)

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {e}, using fallback parsing")
            result = self._extract_from_text(content, result)
        except Exception as e:
            self.logger.error(f"Error in structure extraction: {e}")
            # Return empty result on error

        # Post-process and normalize the extracted data
        result = self._normalize_data(result)

        return result

    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """
        Extract JSON content from the model response.

        Args:
            content: Raw response content

        Returns:
            JSON string if found, None otherwise
        """
        # Look for JSON code blocks
        if "```json" in content:
            start_idx = content.find("```json") + 7
            end_idx = content.find("```", start_idx)
            if end_idx != -1:
                return content[start_idx:end_idx].strip()

        # Look for regular JSON objects
        if content.strip().startswith("{"):
            # Find the matching closing brace
            brace_count = 0
            start_idx = content.find("{")
            for i, char in enumerate(content[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return content[start_idx:i + 1].strip()

        return None

    def _merge_structured_data(self, result: Dict[str, Any], parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge parsed data into the result structure.

        Args:
            result: Base result structure
            parsed_data: Data parsed from JSON response

        Returns:
            Merged result structure
        """
        try:
            # Merge patient information
            if "patient" in parsed_data:
                patient_data = parsed_data["patient"]
                if isinstance(patient_data, dict):
                    for key in result["patient"]:
                        if key in patient_data and patient_data[key] is not None:
                            result["patient"][key] = str(patient_data[key]).strip()

            # Merge doctor information
            if "doctor" in parsed_data:
                doctor_data = parsed_data["doctor"]
                if isinstance(doctor_data, dict):
                    for key in result["doctor"]:
                        if key in doctor_data and doctor_data[key] is not None:
                            result["doctor"][key] = str(doctor_data[key]).strip()

            # Merge insurance information
            if "insurance" in parsed_data:
                insurance_data = parsed_data["insurance"]
                if isinstance(insurance_data, dict):
                    for key in result["insurance"]:
                        if key in insurance_data and insurance_data[key] is not None:
                            result["insurance"][key] = str(insurance_data[key]).strip()

            # Merge DME information
            if "dme" in parsed_data:
                dme_data = parsed_data["dme"]
                if isinstance(dme_data, dict):
                    if "dme_id" in dme_data and dme_data["dme_id"] is not None:
                        result["dme"]["dme_id"] = str(dme_data["dme_id"]).strip()
                    if "items" in dme_data and isinstance(dme_data["items"], list):
                        result["dme"]["items"] = dme_data["items"]

        except Exception as e:
            self.logger.warning(f"Error merging structured data: {e}")

        return result

    def _extract_from_text(self, content: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback method to extract information from text without JSON.

        Args:
            content: Text content to parse
            result: Base result structure to fill

        Returns:
            Result structure with extracted information
        """
        # Simple text-based extraction patterns
        content_lower = content.lower()

        # Extract phone numbers
        import re
        phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
        phones = re.findall(phone_pattern, content)
        if phones:
            result["patient"]["phone"] = phones[0].replace('.', '-').replace(' ', '-')
            if len(phones) > 1:
                result["patient"]["mobile_landline"] = phones[1].replace('.', '-').replace(' ', '-')

        # Extract NPI (10-digit numbers)
        npi_pattern = r'\b(\d{10})\b'
        npi_matches = re.findall(npi_pattern, content)
        for npi in npi_matches:
            # Validate NPI (basic check - real validation is more complex)
            if npi.startswith(('1', '2')):  # NPIs typically start with 1 or 2
                result["doctor"]["npi"] = npi
                break

        # Extract date of birth patterns
        dob_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # MM/DD/YYYY, MM-DD-YYYY
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'   # YYYY/MM/DD
        ]

        for pattern in dob_patterns:
            dob_matches = re.findall(pattern, content)
            if dob_matches:
                result["patient"]["dob"] = dob_matches[0]
                break

        return result

    def _normalize_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and clean the extracted data.

        Args:
            result: Result structure to normalize

        Returns:
            Normalized result structure
        """
        # Normalize patient data
        patient = result["patient"]

        # Normalize phone numbers
        for phone_field in ["phone", "mobile_landline", "fax"]:
            if patient[phone_field]:
                # Remove all non-digit characters except for hyphens
                phone = re.sub(r'[^\d-]', '', patient[phone_field])
                # Add dashes if missing
                if re.match(r'^\d{10}$', phone):
                    phone = f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"
                patient[phone_field] = phone

        # Normalize state code
        if patient["state_code"]:
            patient["state_code"] = patient["state_code"].upper().strip()
        elif patient["state"]:
            # Try to derive state code from state name
            state_mapping = {
                "georgia": "GA", "california": "CA", "texas": "TX",
                "florida": "FL", "new york": "NY", "pennsylvania": "PA"
            }
            state_lower = patient["state"].lower()
            if state_lower in state_mapping:
                patient["state_code"] = state_mapping[state_lower]

        # Normalize NPI
        if result["doctor"]["npi"]:
            # Remove all non-digit characters
            npi = re.sub(r'[^\d]', '', result["doctor"]["npi"])
            if len(npi) == 10:
                result["doctor"]["npi"] = npi
            else:
                result["doctor"]["npi"] = ""

        # Clean up names
        for name_field in ["first_name", "middle_name", "last_name"]:
            if patient[name_field]:
                patient[name_field] = patient[name_field].strip().title()

        # Create full name if not present but components are available
        if not patient["full_name"]:
            name_parts = [patient["first_name"], patient["middle_name"], patient["last_name"]]
            name_parts = [part for part in name_parts if part.strip()]
            if name_parts:
                patient["full_name"] = " ".join(name_parts)

        return result

def main():
    """
    Main function for running the GPT-OSS-20B model interactively.
    """
    try:
        model = GPT_OSS_20B_Model()
        model.run_interactive()
    except Exception as e:
        print(f"Error initializing model: {e}")

if __name__ == "__main__":
    main()