# PatXtract Structure Testing Framework

A comprehensive framework for comparing structured outputs from different AI models when processing medical documents using the PatXtract OCR system.

## Overview

This framework allows you to test and compare how different AI models extract structured information from medical documents. It integrates with the existing PatXtract file_manager OCR system and uses OpenRouter API for model access.

## Features

- **Reusable OCR Data**: Extract OCR data once and reuse across multiple models
- **Multiple Model Support**: Easy to add new AI models via OpenRouter
- **Structured Output**: Consistent JSON format for patient and doctor information
- **Interactive Mode**: Simple command-line interface for testing
- **Caching System**: Efficient caching of OCR data to save processing time
- **Security**: Secure API key management using environment variables

## Required Structured Fields

### Patient Information
- `full_name` - Complete patient name
- `first_name` - Patient's first/given name
- `middle_name` - Patient's middle name/initial
- `last_name` - Patient's last/family name
- `dob` - Date of birth
- `address_full` - Complete street address
- `city` - City name
- `state` - Full state name
- `country` - Country name
- `postal_code` - ZIP/Postal code
- `state_code` - 2-letter state abbreviation
- `phone` - Primary phone number
- `mobile_landline` - Secondary/landline phone
- `email` - Email address
- `fax` - Fax number
- `account_number` - Patient account number

### Doctor Information
- `npi` - National Provider Identifier

## Directory Structure

```
structure_testing/
├── __init__.py
├── README.md
├── extract_data.py          # OCR data extraction utility
├── test_framework.py        # Main testing script
├── allModels/              # Model implementations
│   ├── __init__.py
│   ├── base_model.py       # Abstract base class
│   └── gpt_oss_20b.py      # GPT-OSS-20B implementation
├── config/                 # Configuration files
│   ├── __init__.py
│   └── security.py         # Secure API key management
├── utils/                  # Utility functions
│   └── __init__.py
├── output/                 # Model results (auto-created)
└── extracted_data/         # OCR data cache (auto-created)
```

## Setup

### 1. Environment Configuration

Make sure your `.env` file contains your OpenRouter API key:

```env
API_KEY=your_openrouter_api_key_here
# or specifically:
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 2. Dependencies

Ensure you have all required dependencies installed:

```bash
pip install requests python-dotenv pathlib logging json re
```

## Usage

### Method 1: Interactive Data Extraction

First, extract OCR data from your PDF:

```bash
cd structure_testing
python extract_data.py
```

This will give you options to:
1. Extract data from PDF
2. List cached files
3. Clear cache
4. Exit

### Method 2: Test a Specific Model

Run a model directly (example with GPT-OSS-20B):

```bash
cd structure_testing
python allModels/gpt_oss_20b.py
```

The model will:
1. Ask for PDF path
2. Use cached OCR data or extract fresh data
3. Process each page with the AI model
4. Save results to `output/` directory

### Method 3: Use the Test Framework

```bash
cd structure_testing
python test_framework.py
```

This provides a simple interactive interface for testing the framework.

## Output Format

Results are saved as JSON files in the `output/` directory with the naming format:
`{pdf_name}_{model_name}_{timestamp}.json`

### Example Output Structure

```json
{
  "model_name": "gpt-oss-20b",
  "processing_timestamp": "2025-01-XX...",
  "pdf_path": "/path/to/file.pdf",
  "pdf_name": "document.pdf",
  "total_pages": 2,
  "successful_pages": 2,
  "processing_time_seconds": 45.2,
  "extraction_method": "ocr",
  "pages": [
    {
      "page_number": 1,
      "success": true,
      "raw_text": "Extracted OCR text...",
      "structured_data": {
        "patient": {
          "full_name": "John Doe Smith",
          "first_name": "John",
          "middle_name": "Doe",
          "last_name": "Smith",
          "dob": "08/29/1942",
          "address_full": "1151 Creekwood CV",
          "city": "LAWRENCEVILLE",
          "state": "Georgia",
          "country": "USA",
          "postal_code": "30046",
          "state_code": "GA",
          "phone": "678-349-2069",
          "mobile_landline": "",
          "email": "",
          "fax": "678-546-2844",
          "account_number": ""
        },
        "doctor": {
          "npi": "1013189364"
        }
      },
      "confidence_scores": {
        "overall": 0.85,
        "patient": 0.90,
        "doctor": 0.95
      }
    }
  ],
  "document_summary": {
    "patient_summary": {
      "full_name": "John Doe Smith",
      "dob": "08/29/1942"
    },
    "doctor_npi": "1013189364",
    "extraction_confidence": 0.85,
    "document_type": "medical_form"
  }
}
```

## Adding New Models

To add a new model:

1. Create a new file in `allModels/` (e.g., `claude.py`)
2. Inherit from `BaseModel` class
3. Implement the required abstract methods:
   - `call_api()` - Make API calls to your model
   - `extract_structure()` - Parse structured data from responses

Example structure:

```python
from .base_model import BaseModel

class YourModel(BaseModel):
    def __init__(self):
        super().__init__(model_name="your-model-name", model_id="provider/model-id")

    def call_api(self, messages, **kwargs):
        # Your API call implementation
        pass

    def extract_structure(self, content):
        # Your structured data extraction implementation
        pass
```

## Caching System

The framework automatically caches OCR data to avoid reprocessing the same PDF multiple times:

- OCR data is stored in `extracted_data/` directory
- Cache files are named using PDF filename and modification time
- Use `extract_data.py` to manage cached files
- Models automatically use cached data when available

## Security

- API keys are loaded from environment variables (never hardcoded)
- Use the `SecureConfig` class for secure credential management
- API keys are not logged or exposed in output files

## Error Handling

- Comprehensive error handling with detailed logging
- Graceful fallbacks for JSON parsing
- API retry logic with exponential backoff
- Detailed error messages in output files

## Performance

- OCR data is cached to avoid reprocessing
- Page-wise processing for large documents
- Configurable timeouts and retry logic
- Processing time tracking and reporting

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Check your `.env` file contains the correct API key
   - Ensure the environment variable name matches what's expected

2. **PDF Processing Fails**
   - Verify the PDF file exists and is readable
   - Check that the file is actually a PDF (has .pdf extension)

3. **No OCR Data Found**
   - Run `extract_data.py` to extract OCR data first
   - Check if the PDF contains text or requires OCR

4. **Model API Errors**
   - Check internet connection
   - Verify API key is valid and has sufficient quota
   - Review model-specific error messages

### Debug Mode

Enable detailed logging by setting the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

1. **Test with your PDF documents** using the GPT-OSS-20B model
2. **Compare results** across different models
3. **Add new models** as needed
4. **Enhance extraction prompts** for better accuracy
5. **Implement comparison utilities** for analyzing model performance

## Contributing

To contribute to this framework:

1. Follow the existing code patterns and structure
2. Add proper error handling and logging
3. Document new features and models
4. Test with various PDF types and medical documents