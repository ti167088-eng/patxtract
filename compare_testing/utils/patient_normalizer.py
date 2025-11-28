"""
Patient Data Normalizer

Provides utilities for normalizing patient information including names,
dates of birth, addresses, and other demographic data for consistent
matching and comparison across different documents.
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class PatientNormalizer:
    """
    Utility class for normalizing patient data to enable consistent matching
    and comparison across different documents and OCR variations.
    """

    def __init__(self):
        self.logger = __import__('logging').getLogger(__name__)

        # Common name prefixes and suffixes
        self.name_prefixes = {
            'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'sir', 'madam',
            'mr.', 'mrs.', 'ms.', 'miss.', 'dr.', 'prof.', 'sir.', 'madam.'
        }

        self.name_suffixes = {
            'jr', 'sr', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
            'jr.', 'sr.', 'ii.', 'iii.', 'iv.', 'v.', 'vi.', 'vii.', 'viii.', 'ix.', 'x.'
        }

        # Common name variations/nicknames mapping
        self.name_variations = {
            'william': ['bill', 'will', 'willie'],
            'robert': ['bob', 'rob', 'bobby'],
            'richard': ['dick', 'rick', 'ricky'],
            'james': ['jim', 'jimmy'],
            'john': ['jon', 'johnny'],
            'michael': ['mike', 'mikey'],
            'david': ['dave'],
            'joseph': ['joe', 'joey'],
            'thomas': ['tom', 'tommy'],
            'charles': ['charlie', 'chuck'],
            'elizabeth': ['beth', 'liz', 'lizzie', 'betty'],
            'jennifer': ['jen', 'jenny'],
            'patricia': ['pat', 'patty', 'tricia'],
            'susan': ['sue', 'suzie'],
            'margaret': ['maggie', 'peggy', 'meg'],
            'sarah': ['sally']
        }

        # Address abbreviations
        self.address_abbreviations = {
            'street': 'st',
            'avenue': 'ave',
            'boulevard': 'blvd',
            'lane': 'ln',
            'drive': 'dr',
            'road': 'rd',
            'court': 'ct',
            'place': 'pl',
            'square': 'sq',
            'terrace': 'terr',
            'highway': 'hwy',
            'suite': 'ste',
            'apartment': 'apt',
            'department': 'dept',
            'building': 'bldg',
            'floor': 'fl'
        }

        # State name abbreviations
        self.state_abbreviations = {
            'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
            'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
            'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
            'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
            'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
            'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
            'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
            'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
            'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
            'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
            'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
            'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
            'wisconsin': 'WI', 'wyoming': 'WY'
        }

    def normalize_name(self, name: str) -> str:
        """
        Normalize patient name for consistent matching.

        Args:
            name: Raw patient name string

        Returns:
            Normalized name string
        """
        if not name or not isinstance(name, str):
            return ""

        # Convert to lowercase and remove extra whitespace
        normalized = ' '.join(name.lower().split())

        # Remove common prefixes
        for prefix in sorted(self.name_prefixes, key=len, reverse=True):
            if normalized.startswith(prefix + ' '):
                normalized = normalized[len(prefix + ' '):]
                break

        # Handle and remove suffixes
        suffix = ""
        for suffix_candidate in sorted(self.name_suffixes, key=len, reverse=True):
            if normalized.endswith(' ' + suffix_candidate):
                suffix = suffix_candidate
                normalized = normalized[:-len(' ' + suffix_candidate)]
                break
            elif normalized.endswith(',' + suffix_candidate):
                suffix = suffix_candidate
                normalized = normalized[:-len(',' + suffix_candidate)]
                break

        # Remove special characters (except hyphens in names)
        normalized = re.sub(r'[^\w\s\-]', ' ', normalized)

        # Remove extra spaces and hyphens at boundaries
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        normalized = re.sub(r'\s*\-\s*', '-', normalized)
        normalized = re.sub(r'\-$', '', normalized)
        normalized = re.sub(r'^\-', '', normalized)

        # Add back suffix if it was found
        if suffix:
            normalized = f"{normalized} {suffix}"

        return normalized

    def normalize_dob(self, dob: str) -> str:
        """
        Normalize date of birth to YYYY-MM-DD format.

        Args:
            dob: Raw date of birth string

        Returns:
            Normalized date string in YYYY-MM-DD format
        """
        if not dob or not isinstance(dob, str):
            return ""

        # Remove non-digit characters first
        digits_only = re.sub(r'[^\d]', '', dob)

        if len(digits_only) == 8:
            # Try to determine format based on separators
            if re.search(r'[/\-\.]', dob):
                separators = re.findall(r'[/\-\.]', dob)
                if len(separators) >= 2:
                    # Split based on first separator
                    parts = re.split(r'[/\-\.]', dob)
                    if len(parts) == 3:
                        # Determine if MM/DD/YYYY or DD/MM/YYYY based on context
                        try:
                            mm, dd, yyyy = parts[:3]
                            mm = int(re.sub(r'[^\d]', '', mm))
                            dd = int(re.sub(r'[^\d]', '', dd))
                            yyyy = int(re.sub(r'[^\d]', '', yyyy))

                            # Validate and normalize year
                            if yyyy < 100:
                                # Assume 1900s for years less than 100
                                yyyy += 1900

                            # Validate month and day
                            if 1 <= mm <= 12 and 1 <= dd <= 31:
                                return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
                        except (ValueError, IndexError):
                            pass

            # Fallback: assume MMDDYYYY format
            try:
                mm = int(digits_only[0:2])
                dd = int(digits_only[2:4])
                yyyy = int(digits_only[4:8])

                if yyyy < 100:
                    yyyy += 1900

                if 1 <= mm <= 12 and 1 <= dd <= 31:
                    return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
            except ValueError:
                pass

        elif len(digits_only) == 6:
            # Try MMDDYY format
            try:
                mm = int(digits_only[0:2])
                dd = int(digits_only[2:4])
                yy = int(digits_only[4:6])

                # Assume 1900s for years 50-99, 2000s for 00-49
                yyyy = 1900 + yy if yy >= 50 else 2000 + yy

                if 1 <= mm <= 12 and 1 <= dd <= 31:
                    return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
            except ValueError:
                pass

        # Return original normalized string if format parsing fails
        return dob.lower().strip()

    def normalize_address(self, address: str) -> str:
        """
        Normalize address for consistent matching.

        Args:
            address: Raw address string

        Returns:
            Normalized address string
        """
        if not address or not isinstance(address, str):
            return ""

        # Remove newlines and extra whitespace, convert to lowercase
        normalized = ' '.join(address.lower().split())

        # Replace common words with abbreviations
        for full_form, abbrev in self.address_abbreviations.items():
            normalized = re.sub(r'\b' + re.escape(full_form) + r'\b', abbrev, normalized)

        # Expand common abbreviations
        expanded = {
            'st': 'street', 'ave': 'avenue', 'rd': 'road',
            'ln': 'lane', 'dr': 'drive', 'ct': 'court'
        }
        # Only expand if it's likely a standalone abbreviation, not part of a word
        for abbrev, full_form in expanded.items():
            normalized = re.sub(r'\b' + abbrev + r'\b', full_form, normalized)

        # Normalize state names
        for state_name, state_abbr in self.state_abbreviations.items():
            normalized = re.sub(r'\b' + re.escape(state_name) + r'\b', state_abbr, normalized)

        # Remove special characters except for common address symbols
        normalized = re.sub(r'[^\w\s\#\-\,\.]', ' ', normalized)

        # Remove extra spaces and normalize punctuation
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'\s*,\s*', ', ', normalized)
        normalized = re.sub(r'\s*#\s*', ' #', normalized)

        return normalized.strip()

    def normalize_phone(self, phone: str) -> str:
        """
        Normalize phone number to XXX-XXX-XXXX format.

        Args:
            phone: Raw phone number string

        Returns:
            Normalized phone string
        """
        if not phone or not isinstance(phone, str):
            return ""

        # Remove all non-digit characters
        digits_only = re.sub(r'[^\d]', '', phone)

        # Handle different phone number lengths
        if len(digits_only) == 10:
            # Standard US phone number
            return f"{digits_only[0:3]}-{digits_only[3:6]}-{digits_only[6:10]}"
        elif len(digits_only) == 11 and digits_only[0] == '1':
            # US phone with country code
            return f"{digits_only[1:4]}-{digits_only[4:7]}-{digits_only[7:11]}"
        elif len(digits_only) == 7:
            # Local number
            return f"{digits_only[0:3]}-{digits_only[3:7]}"

        # Return normalized original if format doesn't match expected patterns
        return phone.strip()

    def get_name_variants(self, name: str) -> List[str]:
        """
        Get possible variants of a name for fuzzy matching.

        Args:
            name: Normalized name

        Returns:
            List of possible name variants
        """
        variants = [name]
        base_name = self.normalize_name(name)

        # Split into components
        parts = base_name.split()
        if not parts:
            return variants

        # Try to find full name variations
        for part in parts:
            part_clean = part.strip('.,-_')
            for full_name, nicknames in self.name_variations.items():
                if part_clean == full_name:
                    for nickname in nicknames:
                        variant_name = base_name.replace(part_clean, nickname)
                        if variant_name not in variants:
                            variants.append(variant_name)
                elif part_clean in nicknames:
                    variant_name = base_name.replace(part_clean, full_name)
                    if variant_name not in variants:
                        variants.append(variant_name)

        return variants

    def create_name_signature(self, name: str) -> str:
        """
        Create a signature for name matching that accounts for variations.

        Args:
            name: Patient name

        Returns:
            Name signature string
        """
        normalized = self.normalize_name(name)
        if not normalized:
            return ""

        # Split into first and last name parts
        parts = normalized.split()
        if len(parts) < 2:
            return normalized

        # Handle compound last names
        first_name = parts[0]
        last_name = ' '.join(parts[1:])

        # Create signature using first initial + last name
        signature = f"{first_name[0]}{last_name}"

        # Add name variants to signature
        variants = self.get_name_variants(name)
        if variants:
            signature += "|" + "|".join(sorted(set(variants)))

        return signature

    def normalize_mrn(self, mrn: str) -> str:
        """
        Normalize medical record number.

        Args:
            mrn: Raw medical record number

        Returns:
            Normalized MRN string
        """
        if not mrn or not isinstance(mrn, str):
            return ""

        # Remove common prefixes and whitespace
        normalized = re.sub(r'(?i)(mrn|medical record|patient id|account)\s*[:#]?\s*', '', mrn)
        normalized = re.sub(r'[^\w\-]', '', normalized.strip())

        return normalized.upper()

    def extract_dob_from_text(self, text: str) -> List[str]:
        """
        Extract potential dates of birth from text.

        Args:
            text: Text to search for DOB patterns

        Returns:
            List of potential DOB strings
        """
        if not text:
            return []

        # Common DOB patterns
        patterns = [
            r'\bDOB[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b',  # DOB: MM/DD/YYYY
            r'\bDate of Birth[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b',
            r'\bBorn[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b',
            r'\bAge[:\s]*(\d{1,2})[yo]\b',  # Age 50yo - convert to estimated DOB
            r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b'  # Generic date format
        ]

        found_dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in found_dates:
                    found_dates.append(match)

        return found_dates