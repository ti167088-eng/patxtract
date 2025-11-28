"""
Path utilities for handling file paths in PatXtract.
"""

import os
from pathlib import Path
from typing import Optional, Union


def normalize_path_input(path_input: str) -> Path:
    """
    Normalize user input path by handling quotes and path formats.

    Args:
        path_input: Raw path string from user input or source

    Returns:
        Normalized Path object
    """
    if not path_input:
        raise ValueError("Path input cannot be empty")

    # Remove surrounding quotes if present
    path_input = path_input.strip()
    if (path_input.startswith('"') and path_input.endswith('"')) or \
       (path_input.startswith("'") and path_input.endswith("'")):
        path_input = path_input[1:-1]

    # Handle escape sequences in Windows paths
    path_input = path_input.replace('\\', os.sep)

    return Path(path_input)


def validate_pdf_path(pdf_path: Union[str, Path]) -> Optional[Path]:
    """
    Validate that a path points to an existing PDF file.

    Args:
        pdf_path: Path to validate

    Returns:
        Validated Path object or None if invalid
    """
    try:
        path = normalize_path_input(str(pdf_path)) if isinstance(pdf_path, str) else Path(pdf_path)

        if not path.exists():
            return None

        if path.suffix.lower() != '.pdf':
            return None

        return path

    except (ValueError, OSError):
        return None


def get_example_pdf_path() -> str:
    """
    Get an example of a properly quoted PDF path.

    Returns:
        Example path string with quotes
    """
    return '"C:\\Users\\pc\\Desktop\\data\\patient wise pdf\\Ali Haider -20250917T091038Z-1-002\\Ali Haider\\Medical Records\\Alphonso_ Faulkner\\Alphonso Faulkner DO (1).pdf"'