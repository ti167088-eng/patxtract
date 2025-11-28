"""
Google Gemma 3 4B IT model implementation for structure extraction
"""
import sys
import os
import json
from typing import Dict, Any, List

# Add the parent directory to the path to import from config and base_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.security import SecureConfig
    from base_model import BaseModel
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.security import SecureConfig
    from base_model import BaseModel


class Gemma_3_4B_IT(BaseModel):
    """Google Gemma 3 4B IT model implementation via OpenRouter"""

    def __init__(self):
        """Initialize the Google Gemma 3 4B IT model."""
        super().__init__(
            model_name="Gemma_3_4B_IT",
            model_id="google/gemma-3-4b-it"
        )
        self.api_url = f"{self.base_url}/chat/completions"
        self.logger.info("Google Gemma 3 4B IT model initialized")

    def call_api(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Make API call to Google Gemma 3 4B IT via OpenRouter.

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
            "temperature": kwargs.get("temperature", self.model_config.get("temperature", 0.2)),
            "max_tokens": kwargs.get("max_tokens", self.model_config.get("max_tokens", 4000)),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": False
        }

        self.logger.debug(f"Making API call to {self.api_url}")

        try:
            import requests
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
        Extract structured data from Google Gemma 3 4B IT response.

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

        try:
            # Try to extract JSON from the content
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group())
                    # Merge extracted data with result structure
                    if "patient" in extracted_data:
                        result["patient"].update(extracted_data["patient"])
                    if "doctor" in extracted_data:
                        result["doctor"].update(extracted_data["doctor"])
                    if "insurance" in extracted_data:
                        result["insurance"].update(extracted_data["insurance"])
                    if "dme" in extracted_data:
                        result["dme"].update(extracted_data["dme"])
                    return result
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse JSON from model response")

            # If JSON parsing fails, return the default structure
            self.logger.warning("Failed to extract JSON, returning default structure")

        except Exception as e:
            self.logger.error(f"Error extracting structure: {e}")
            return result


def main():
    """Main function to run the Google Gemma 3 4B IT model"""
    try:
        extractor = Gemma_3_4B_IT()
        extractor.run_interactive()
    except Exception as e:
        print(f"ERROR: Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())