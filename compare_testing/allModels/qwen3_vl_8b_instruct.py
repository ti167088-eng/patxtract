"""
Qwen3-VL 8B Instruct Patient Identification Model

Uses the Qwen3-VL 8B Instruct model via OpenRouter to identify patients
and group pages within multi-patient documents. This model can process
both text and images, providing enhanced OCR error handling and visual
document analysis capabilities.
"""

import sys
import os
import json
import base64
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


class Qwen3_VL_8B_Instruct_PatientIdentifier(BasePatientIdentifier):
    """Qwen3-VL 8B Instruct patient identification model via OpenRouter"""

    def __init__(self):
        """Initialize the Qwen3-VL 8B Instruct patient identifier."""
        super().__init__("Qwen3_VL_8B_Instruct_Patient")

        # Model configuration
        self.model_name = "Qwen3_VL_8B_Instruct_Patient"
        self.model_id = "qwen/qwen3-vl-8b-instruct"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.supports_vision = True

        # Load API configuration
        self.api_key = self.config.get_openrouter_key()
        if not self.api_key:
            raise ValueError("OpenRouter API key not found in configuration")

        # Algorithm-specific configuration
        self.temperature = self.algorithm_config.get('temperature', 0.1)
        self.max_tokens = self.algorithm_config.get('max_tokens', 4000)
        self.timeout = self.algorithm_config.get('timeout', 45)  # Longer timeout for vision processing
        self.max_image_size = self.algorithm_config.get('max_image_size', 20 * 1024 * 1024)  # 20MB
        self.image_quality = self.algorithm_config.get('image_quality', 0.8)  # JPEG quality

        self.logger.info(f"Qwen3-VL 8B Instruct Patient Identifier initialized with model: {self.model_id}")
        self.logger.info(f"Vision support: {self.supports_vision}")

    def call_api(self, messages: List[Dict[str, str]], images: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Make API call to Qwen3-VL 8B Instruct via OpenRouter with vision support.

        Args:
            messages: List of message dictionaries for the API call
            images: List of base64-encoded image strings (optional)
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
            "X-Title": "PatXtract Patient Identification - Qwen3-VL 8B",
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

        # Add images if provided and model supports vision
        if images and self.supports_vision:
            # Add images to the last user message
            for message in messages:
                if message.get("role") == "user":
                    if "content" not in message:
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [{"type": "text", "text": message["content"]}]

                    # Add image content
                    for img_data in images:
                        message["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}",
                                "detail": "high"
                            }
                        })
                    break

        self.logger.debug(f"Making API call to {self.api_url} with {len(images) if images else 0} images")

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

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Convert an image file to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded image string

        Raises:
            Exception: If image encoding fails
        """
        try:
            from PIL import Image
            import io

            # Open and optimize the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if too large
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Compress to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=int(self.image_quality * 100))
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Check size
                if len(img_str) > self.max_image_size:
                    self.logger.warning(f"Image size ({len(img_str)} bytes) exceeds limit")

                return img_str

        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {e}")
            raise Exception(f"Image encoding failed: {str(e)}")

    def create_patient_identification_prompt(self, pages_text: str, total_pages: int) -> List[Dict[str, str]]:
        """
        Create a comprehensive prompt for patient identification and page grouping.
        Enhanced for vision-language model with emphasis on visual document analysis.

        Args:
            pages_text: Combined text from all pages
            total_pages: Total number of pages in the document

        Returns:
            List of message dictionaries for the API call
        """
        system_prompt = """You are an expert medical document analyst with advanced vision and language capabilities. Your task is to:

1. **VISUAL ANALYSIS**: Examine document images for patient identifiers, headers, forms, and visual patterns
2. **PATIENT IDENTIFICATION**: Identify all unique patients mentioned in the document
3. **PAGE GROUPING**: Determine which pages belong to each patient
4. **NON-CONSECUTIVE HANDLING**: Properly handle separate entries when the same patient appears on non-consecutive pages
5. **OCR ERROR RECOVERY**: Use visual information to correct OCR errors and ambiguities

**VISION CAPABILITIES**:
- Analyze document layouts, form structures, and visual patient identifiers
- Read patient information from headers, forms, labels, and medical records
- Detect visual separators between different patient sections
- Verify patient identity through visual document features

**KEY REQUIREMENTS**:
- Each patient should have a unique patient_id (patient_001, patient_002, etc.)
- Track separate entries when the same patient appears on non-consecutive pages
- Provide confidence scores based on both textual and visual evidence
- Use visual analysis to resolve OCR errors and ambiguities
- Be thorough in extracting all available patient information

**OUTPUT FORMAT**:
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
            "evidence_summary": "Brief summary of visual and textual evidence",
            "visual_indicators": ["Header with patient name", "Form fields", "Document layout"]
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
    "reasons": ["No patient information found", "Insufficient visual or textual data"]
  }
}

**CRITICAL RULES**:
1. Create separate entries for non-consecutive appearances of the same patient
2. Use both visual and textual evidence for patient identification
3. Provide higher confidence scores when multiple evidence types agree
4. Include visual indicators in supporting_data when available
5. Handle OCR errors by cross-referencing visual information
6. Be precise with patient matching - use visual verification when possible

**NON-CONSECUTIVE PAGE HANDLING EXAMPLE**:
If pages 1-2 belong to Patient A, page 3 belongs to Patient B, and page 4 belongs to Patient A again:
- Patient A should have 2 entries: pages [1,2] and [4]
- Patient B should have 1 entry: pages [3]
- Track total_entries = 2 for Patient A

**VISUAL ANALYSIS INSTRUCTIONS**:
- Look for patient names in document headers, form titles, and section headers
- Identify patient information from form layouts and structured fields
- Use visual separators (lines, boxes, spacing) to distinguish patient sections
- Verify patient information consistency across visual and textual elements
- Pay attention to document stamps, signatures, and official visual markers"""

        user_prompt = f"""Please analyze the following multi-page medical document using both text and visual information to identify all patients and group pages appropriately.

**Document Information**:
- Total pages: {total_pages}
- Page numbering starts from 1
- Visual analysis: Document images provided for each page
- Analysis approach: Combined visual and textual patient identification

**Document Text (pages are separated by "=== PAGE X ===")**:
{pages_text}

**Instructions**:
1. Analyze both the provided document images and extracted text
2. Use visual information to identify patients, form layouts, and document structure
3. Cross-reference visual and textual evidence for accurate patient identification
4. Group pages by patient based on comprehensive visual and textual analysis
5. Handle non-consecutive page assignments properly
6. Include visual indicators in your evidence when available

Focus on leveraging the vision capabilities to:
- Extract patient names from headers and form layouts
- Identify patient sections through visual document structure
- Verify and correct OCR errors using visual information
- Detect visual patient identifiers that might be missed in text extraction

Return ONLY the JSON response, no additional text or explanations."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def prepare_pages_with_images(self, pages: List[Dict[str, Any]], source_data_path: str) -> tuple:
        """
        Prepare pages text and collect base64 images for vision analysis.

        Args:
            pages: List of page data dictionaries
            source_data_path: Path to the source JSON file (for finding images)

        Returns:
            Tuple of (formatted_text, list_of_base64_images)
        """
        pages_text_parts = []
        images = []

        # Try to find corresponding image files
        image_dir = None
        if source_data_path:
            source_path = Path(source_data_path)
            possible_image_dirs = [
                source_path.parent / "images",
                source_path.parent.parent / "images",
                source_path.parent.parent.parent / "file_manager" / "extractor" / "ocr" / "output"
            ]

            for possible_dir in possible_image_dirs:
                if possible_dir.exists():
                    image_dir = possible_dir
                    break

        for page in pages:
            page_number = page['page_number']
            text = page['text']
            confidence = page.get('confidence', 0.0)

            pages_text_parts.append(f"=== PAGE {page_number} (Confidence: {confidence:.2f}) ===")
            pages_text_parts.append(text.strip())
            pages_text_parts.append("")  # Empty line for separation

            # Try to find and encode corresponding image
            if image_dir:
                possible_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
                for ext in possible_extensions:
                    image_path = image_dir / f"page_{page_number}{ext}"
                    if image_path.exists():
                        try:
                            img_base64 = self.encode_image_to_base64(str(image_path))
                            images.append(img_base64)
                            self.logger.debug(f"Added image for page {page_number}: {image_path}")
                            break
                        except Exception as e:
                            self.logger.warning(f"Failed to encode image {image_path}: {e}")
                    else:
                        # Try with different naming patterns
                        image_path = image_dir / f"page_{page_number:03d}{ext}"
                        if image_path.exists():
                            try:
                                img_base64 = self.encode_image_to_base64(str(image_path))
                                images.append(img_base64)
                                self.logger.debug(f"Added image for page {page_number}: {image_path}")
                                break
                            except Exception as e:
                                self.logger.warning(f"Failed to encode image {image_path}: {e}")

        formatted_text = "\n".join(pages_text_parts)
        return formatted_text, images

    def identify_patients(self, pages: List[Dict[str, Any]], source_data_path: str = None) -> Dict[str, Any]:
        """
        Identify patients across pages using Qwen3-VL 8B Instruct analysis with vision support.

        Args:
            pages: List of page data dictionaries
            source_data_path: Path to source data for finding images

        Returns:
            Dictionary with patient grouping results
        """
        self.logger.info(f"Starting Qwen3-VL 8B Instruct patient identification for {len(pages)} pages")

        # Prepare text and images
        pages_text, images = self.prepare_pages_with_images(pages, source_data_path)
        self.logger.info(f"Prepared {len(images)} images for vision analysis")

        # Check if document needs chunking (based on text length)
        total_chars = len(pages_text)
        estimated_tokens = total_chars * 0.3  # Rough estimation for vision + text

        model_limit = 32000  # Qwen models typically have 32K context
        if estimated_tokens > model_limit * 0.8:
            # Use chunking for large documents
            self.logger.info("Document requires chunking due to size")
            return self.call_model_with_chunking(
                pages=pages,
                create_prompt_func=self.create_patient_identification_prompt,
                call_api_func=self.call_api,
                extract_structure_func=self.extract_structure,
                create_empty_result_func=self.create_empty_result
            )
        else:
            # Process as single chunk with vision
            return self.call_model_with_vision_and_chunking(pages, source_data_path)

    def call_model_with_vision_and_chunking(self, pages: List[Dict[str, Any]], source_data_path: str = None) -> Dict[str, Any]:
        """
        Process pages with vision capabilities, handling chunking if needed.

        Args:
            pages: List of page data dictionaries
            source_data_path: Path to source data for finding images

        Returns:
            Dictionary with patient identification results
        """
        start_time = time.time()
        total_pages = len(pages)

        try:
            # Prepare text and images
            pages_text, images = self.prepare_pages_with_images(pages, source_data_path)

            # Create prompt
            messages = self.create_patient_identification_prompt(pages_text, total_pages)

            # Make API call with images
            api_result = self.call_api(messages, images=images)

            # Extract structure from response
            if 'choices' in api_result and len(api_result['choices']) > 0:
                content = api_result['choices'][0]['message']['content']
                extracted_data = self.extract_structure(content)

                # Add metadata
                processing_time = time.time() - start_time
                result = {
                    'algorithm_name': self.algorithm_name,
                    'processing_timestamp': self.get_current_timestamp(),
                    'total_pages': total_pages,
                    'processing_time_seconds': processing_time,
                    'model_used': self.model_id,
                    'vision_enabled': True,
                    'images_processed': len(images),
                    'tokens_used': api_result.get('usage', {}).get('total_tokens', 0),
                    **extracted_data
                }

                self.logger.info(f"Vision-based patient identification completed in {processing_time:.2f}s")
                return result
            else:
                raise Exception("No valid response from API")

        except Exception as e:
            self.logger.error(f"Error in vision-based patient identification: {e}")
            processing_time = time.time() - start_time
            return self.create_empty_result(total_pages, processing_time)

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
            # Vision models get higher base confidence due to visual verification
            final_confidence = (base_confidence * 0.6) + (ocr_confidence * 0.4)
        else:
            final_confidence = base_confidence

        return min(1.0, max(0.0, final_confidence))

    def prepare_pages_text(self, pages: List[Dict[str, Any]]) -> str:
        """
        Prepare pages text for model analysis (compatibility fallback).

        Args:
            pages: List of page data dictionaries

        Returns:
            Formatted text string for model processing
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
            'vision_enabled': True,
            'images_processed': 0,
            'patient_groups': {},
            'unassigned_pages': {
                'pages': list(range(1, total_pages + 1)),
                'reasons': ['No patient information identified by Qwen3-VL model']
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
    """Main function to run the Qwen3-VL 8B Instruct patient identifier"""
    try:
        identifier = Qwen3_VL_8B_Instruct_PatientIdentifier()
        print(f"{identifier.algorithm_name} ready for use")
        print(f"Model: {identifier.model_id}")
        print(f"API Endpoint: {identifier.api_url}")
        print(f"Vision Support: {identifier.supports_vision}")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())