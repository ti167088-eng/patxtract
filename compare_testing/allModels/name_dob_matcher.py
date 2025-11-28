"""
Name + DOB Matcher Algorithm

Primary patient identification algorithm that uses name and date of birth
as the primary identifier for grouping pages by patient.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Set
import re

# Add parent directories to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from base_patient_identifier import BasePatientIdentifier
from utils.patient_normalizer import PatientNormalizer
from utils.similarity_calculator import SimilarityCalculator


class NameDobMatcher(BasePatientIdentifier):
    """
    Patient identification algorithm using name + DOB as primary identifier.

    This algorithm extracts patient names and dates of birth from each page,
    creates patient signatures, and groups pages with matching signatures.
    It handles OCR variations, name formatting differences, and various DOB formats.
    """

    def __init__(self):
        super().__init__("Name_DOB_Matcher")

        # Initialize utility classes
        self.normalizer = PatientNormalizer()
        self.similarity_calculator = SimilarityCalculator()

        # Configuration
        self.name_similarity_threshold = self.algorithm_config.get('name_similarity_threshold', 0.8)
        self.dob_match_required = self.algorithm_config.get('dob_match_required', True)
        self.min_confidence_score = self.algorithm_config.get('min_confidence_score', 0.6)

        # Regex patterns for extracting patient information
        self.name_patterns = [
            r'Patient Name[:\s]*([A-Za-z\-\,\.\s]+?)(?:\n|$|[A-Z])',
            r'Name[:\s]*([A-Za-z\-\,\.\s]+?)(?:\n|$|[A-Z])',
            r'(?:Patient|Name)[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)*)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)*)',
            r'(?:[Pp]atient|Name|Name of Patient)[:\s]*([A-Za-z\s\,\.\-]+?)(?:\n|Sex|DOB|Age|$)'
        ]

        self.dob_patterns = [
            r'DOB[:\s]*(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})',
            r'Date of Birth[:\s]*(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})',
            r'Birth(?:\s+Date)?[:\s]*(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})',
            r'(?:(?:Sex|Gender)[:\s]*[MFmf][,\s]*)?DOB[:\s]*(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})',
            r'Age[:\s]*(\d{1,2})\s*yo(?:\s*\(DOB:\s*(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})\))?'
        ]

        self.logger.info("Name + DOB Matcher algorithm initialized")

    def identify_patients(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify patients across pages using name and DOB matching.

        Args:
            pages: List of page data dictionaries

        Returns:
            Dictionary with patient grouping results
        """
        start_time = time.time()

        if not self.validate_pages_data(pages):
            raise ValueError("Invalid pages data format")

        self.logger.info(f"Starting patient identification for {len(pages)} pages")

        # Step 1: Extract patient information from each page
        page_patient_data = []
        for page in pages:
            patient_info = self.extract_patient_info(page['text'])
            if patient_info:
                patient_info['page_number'] = page['page_number']
                patient_info['confidence'] = page.get('confidence', 0.0)
                page_patient_data.append(patient_info)

        if not page_patient_data:
            self.logger.warning("No patient information found in any pages")
            return self.create_empty_result(len(pages), time.time() - start_time)

        # Step 2: Create patient signatures and group pages
        patient_groups = self.group_pages_by_patient(page_patient_data)

        # Step 3: Calculate confidence scores for each group
        for patient_id, group_data in patient_groups.items():
            confidence = self.calculate_group_confidence(group_data['entries'], pages)
            group_data['group_confidence'] = confidence

        # Step 4: Handle unassigned pages
        assigned_pages = set()
        for group_data in patient_groups.values():
            for entry in group_data['entries']:
                assigned_pages.update(entry['pages'])

        unassigned_pages = [
            page['page_number'] for page in pages
            if page['page_number'] not in assigned_pages
        ]

        # Step 5: Create summary
        processing_time = time.time() - start_time
        summary = self.get_processing_summary(
            patient_groups,
            {'pages': unassigned_pages, 'reasons': ['No patient information found']},
            len(pages),
            processing_time
        )

        result = {
            'algorithm_name': self.algorithm_name,
            'processing_timestamp': datetime.now().isoformat(),
            'total_pages': len(pages),
            'processing_time_seconds': processing_time,
            'patient_groups': patient_groups,
            'unassigned_pages': {
                'pages': unassigned_pages,
                'reasons': ['No patient information found'] if unassigned_pages else []
            },
            'summary': summary
        }

        self.logger.info(f"Patient identification completed: {summary['total_patients']} patients, {summary['total_entries']} entries")
        return result

    def extract_patient_info(self, text: str) -> Dict[str, str]:
        """
        Extract patient name and DOB from page text.

        Args:
            text: OCR-extracted text from a page

        Returns:
            Dictionary with extracted patient information
        """
        patient_info = {
            'full_name': '',
            'first_name': '',
            'last_name': '',
            'middle_name': '',
            'dob': '',
            'extracted_name': '',
            'extracted_dob': ''
        }

        # Extract name
        name_candidates = []
        for pattern in self.name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                name_str = match.strip()
                if name_str and len(name_str) > 2:  # Minimum length check
                    name_candidates.append(name_str)

        if name_candidates:
            # Select the best name candidate
            best_name = self.select_best_name_candidate(name_candidates, text)
            patient_info['extracted_name'] = best_name
            patient_info['full_name'] = best_name

            # Parse name components
            name_parts = self.parse_name_components(best_name)
            patient_info.update(name_parts)

        # Extract DOB
        dob_candidates = []
        for pattern in self.dob_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle patterns with multiple capture groups
                    for group in match:
                        if group and self.is_valid_date_format(group):
                            dob_candidates.append(group.strip())
                else:
                    if match and self.is_valid_date_format(match):
                        dob_candidates.append(match.strip())

        if dob_candidates:
            # Select the best DOB candidate
            best_dob = self.select_best_dob_candidate(dob_candidates)
            patient_info['extracted_dob'] = best_dob
            patient_info['dob'] = self.normalizer.normalize_dob(best_dob)

        return patient_info

    def select_best_name_candidate(self, candidates: List[str], text: str) -> str:
        """
        Select the best name candidate from a list of potential names.

        Args:
            candidates: List of potential name strings
            text: Original text for context

        Returns:
            Best name candidate
        """
        if not candidates:
            return ""

        # Score each candidate
        scored_candidates = []
        for name in candidates:
            score = 0

            # Prefer names with proper capitalization
            if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', name):
                score += 3
            elif re.match(r'^[A-Z]+\s+[A-Z]+$', name):  # All caps
                score += 2

            # Prefer names with spaces (full names)
            if ' ' in name:
                score += 2

            # Prefer names that look like patient names (not headers, etc.)
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in ['patient', 'name', 'mr', 'mrs', 'dr']):
                score -= 1

            # Prefer names with reasonable length
            if 5 <= len(name) <= 50:
                score += 1

            # Prefer names that appear with DOB patterns nearby
            if re.search(r'dob|birth|age', text[text.find(name):text.find(name) + 200], re.IGNORECASE):
                score += 2

            scored_candidates.append((name, score))

        # Return the highest scoring candidate
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

    def select_best_dob_candidate(self, candidates: List[str]) -> str:
        """
        Select the best DOB candidate from a list of potential dates.

        Args:
            candidates: List of potential DOB strings

        Returns:
            Best DOB candidate
        """
        if not candidates:
            return ""

        # Score each candidate
        scored_candidates = []
        for dob in candidates:
            score = 0

            # Prefer dates with separators
            if re.search(r'[\/\-\.\s]', dob):
                score += 2

            # Prefer 8-digit dates
            digits = re.sub(r'[^\d]', '', dob)
            if len(digits) == 8:
                score += 1

            # Prefer reasonable years (1900-2024)
            year_match = re.search(r'(19|20)\d{2}', dob)
            if year_match:
                year = int(year_match.group())
                if 1900 <= year <= 2024:
                    score += 2

            scored_candidates.append((dob, score))

        # Return the highest scoring candidate
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

    def is_valid_date_format(self, date_str: str) -> bool:
        """
        Check if a string looks like a valid date.

        Args:
            date_str: Date string to validate

        Returns:
            True if valid date format
        """
        # Check for common date patterns
        patterns = [
            r'^\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4}$',  # MM/DD/YYYY
            r'^\d{8}$',  # MMDDYYYY
            r'^\d{6}$'   # MMDDYY
        ]

        return any(re.match(pattern, date_str) for pattern in patterns)

    def parse_name_components(self, full_name: str) -> Dict[str, str]:
        """
        Parse full name into components.

        Args:
            full_name: Full patient name

        Returns:
            Dictionary with name components
        """
        name_parts = {
            'first_name': '',
            'last_name': '',
            'middle_name': ''
        }

        if not full_name:
            return name_parts

        # Clean up the name
        cleaned_name = ' '.join(full_name.strip().split())

        # Split into parts
        parts = cleaned_name.split()

        if len(parts) == 1:
            name_parts['first_name'] = parts[0]
        elif len(parts) == 2:
            name_parts['first_name'] = parts[0]
            name_parts['last_name'] = parts[1]
        elif len(parts) == 3:
            name_parts['first_name'] = parts[0]
            name_parts['middle_name'] = parts[1]
            name_parts['last_name'] = parts[2]
        else:
            # For longer names, assume first is first, last is last, rest are middle
            name_parts['first_name'] = parts[0]
            name_parts['middle_name'] = ' '.join(parts[1:-1])
            name_parts['last_name'] = parts[-1]

        return name_parts

    def group_pages_by_patient(self, page_patient_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Group pages by patient using name and DOB signatures.

        Args:
            page_patient_data: List of patient information from each page

        Returns:
            Dictionary of patient groups
        """
        patient_groups = {}
        patient_signatures = {}  # Track which patient has which signature

        # First pass: create initial groups based on exact signature matches
        for page_data in page_patient_data:
            signature = self.create_patient_signature(page_data)

            if not signature:
                continue

            # Find existing patient with similar signature
            matched_patient_id = None
            for patient_id, patient_sig in patient_signatures.items():
                if self.is_signature_match(signature, patient_sig):
                    matched_patient_id = patient_id
                    break

            if matched_patient_id:
                # Add to existing patient group
                patient_id = matched_patient_id
            else:
                # Create new patient group
                patient_id = self.create_patient_id(patient_groups)
                patient_groups[patient_id] = {
                    'patient_info': page_data,
                    'entries': [],
                    'total_entries': 0,
                    'total_pages': 0,
                    'all_pages': []
                }
                patient_signatures[patient_id] = signature

            # Add page to patient entry (handle non-consecutive pages)
            self.add_page_to_patient_entry(patient_groups[patient_id], page_data, signature)

        # Second pass: handle fuzzy matches and merge similar patients
        patient_groups = self.merge_similar_patients(patient_groups, patient_signatures)

        return patient_groups

    def create_patient_signature(self, patient_data: Dict[str, Any]) -> str:
        """
        Create a unique signature for patient identification.

        Args:
            patient_data: Patient information dictionary

        Returns:
            Patient signature string
        """
        name = self.normalizer.normalize_name(patient_data.get('full_name', ''))
        dob = self.normalizer.normalize_dob(patient_data.get('dob', ''))

        if not name:
            return ""

        return f"{name}|{dob}" if dob else f"{name}|"

    def is_signature_match(self, sig1: str, sig2: str) -> bool:
        """
        Check if two patient signatures match.

        Args:
            sig1: First patient signature
            sig2: Second patient signature

        Returns:
            True if signatures match
        """
        if not sig1 or not sig2:
            return False

        parts1 = sig1.split('|')
        parts2 = sig2.split('|')

        # Check name similarity
        name1 = parts1[0] if len(parts1) > 0 else ""
        name2 = parts2[0] if len(parts2) > 0 else ""

        name_similarity = self.similarity_calculator.name_similarity(name1, name2)
        if name_similarity < self.name_similarity_threshold:
            return False

        # Check DOB if both have it
        dob1 = parts1[1] if len(parts1) > 1 else ""
        dob2 = parts2[1] if len(parts2) > 1 else ""

        if self.dob_match_required and dob1 and dob2:
            return dob1 == dob2
        elif self.dob_match_required and dob1:
            return False  # Both should have DOB for confident match
        else:
            return True

    def add_page_to_patient_entry(self, patient_group: Dict[str, Any], page_data: Dict[str, Any], signature: str):
        """
        Add a page to the appropriate patient entry.

        Args:
            patient_group: Patient group dictionary
            page_data: Page patient data
            signature: Patient signature
        """
        page_number = page_data['page_number']

        # Check if this page continues a previous entry or starts a new one
        if patient_group['entries']:
            last_entry = patient_group['entries'][-1]
            last_pages = last_entry['pages']

            # Check if this page is consecutive to the last page in the last entry
            if page_number == max(last_pages) + 1:
                # Add to existing entry (consecutive pages)
                last_entry['pages'].append(page_number)
                last_entry['pages'].sort()
                last_entry['page_range'] = self.format_page_range(last_entry['pages'])
            else:
                # Create new entry (non-consecutive pages)
                self.create_new_patient_entry(patient_group, page_data, signature)
        else:
            # First entry for this patient
            self.create_new_patient_entry(patient_group, page_data, signature)

        # Update patient group totals
        patient_group['all_pages'].append(page_number)
        patient_group['all_pages'].sort()
        patient_group['total_pages'] += 1

    def create_new_patient_entry(self, patient_group: Dict[str, Any], page_data: Dict[str, Any], signature: str):
        """
        Create a new patient entry.

        Args:
            patient_group: Patient group dictionary
            page_data: Page patient data
            signature: Patient signature
        """
        entry_id = self.create_entry_id(
            list(patient_group.keys())[0] if patient_group else "patient_001",
            len(patient_group['entries']) + 1
        )

        new_entry = {
            'entry_id': entry_id,
            'pages': [page_data['page_number']],
            'page_range': str(page_data['page_number']),
            'confidence': page_data.get('confidence', 0.0),
            'supporting_data': {
                'extracted_name': page_data.get('extracted_name', ''),
                'extracted_dob': page_data.get('extracted_dob', ''),
                'signature': signature
            }
        }

        patient_group['entries'].append(new_entry)
        patient_group['total_entries'] += 1

    def format_page_range(self, pages: List[int]) -> str:
        """
        Format a list of page numbers as a range string.

        Args:
            pages: List of page numbers

        Returns:
            Formatted page range string
        """
        if not pages:
            return ""

        pages = sorted(pages)
        if len(pages) == 1:
            return str(pages[0])

        ranges = []
        start = pages[0]
        end = pages[0]

        for i in range(1, len(pages)):
            if pages[i] == end + 1:
                end = pages[i]
            else:
                ranges.append(f"{start}-{end}" if start != end else str(start))
                start = end = pages[i]

        ranges.append(f"{start}-{end}" if start != end else str(start))
        return ", ".join(ranges)

    def merge_similar_patients(self, patient_groups: Dict[str, Any], patient_signatures: Dict[str, str]) -> Dict[str, Any]:
        """
        Merge patients that are likely the same person.

        Args:
            patient_groups: Dictionary of patient groups
            patient_signatures: Dictionary of patient signatures

        Returns:
            Merged patient groups
        """
        patient_ids = list(patient_groups.keys())
        merged_groups = dict(patient_groups)
        merged_signatures = dict(patient_signatures)

        i = 0
        while i < len(patient_ids):
            j = i + 1
            while j < len(patient_ids):
                id1 = patient_ids[i]
                id2 = patient_ids[j]

                # Check if these two patients should be merged
                if self.should_merge_patients(merged_groups[id1], merged_groups[id2]):
                    # Merge id2 into id1
                    self.merge_patient_data(merged_groups[id1], merged_groups[id2])
                    del merged_groups[id2]
                    del merged_signatures[id2]
                    patient_ids.pop(j)
                else:
                    j += 1
            i += 1

        return merged_groups

    def should_merge_patients(self, patient1: Dict[str, Any], patient2: Dict[str, Any]) -> bool:
        """
        Determine if two patient groups should be merged.

        Args:
            patient1: First patient group
            patient2: Second patient group

        Returns:
            True if patients should be merged
        """
        info1 = patient1['patient_info']
        info2 = patient2['patient_info']

        # High similarity threshold for merging
        similarity = self.similarity_calculator.calculate_patient_similarity(info1, info2)
        return similarity >= 0.95

    def merge_patient_data(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Merge source patient data into target.

        Args:
            target: Target patient group
            source: Source patient group to merge
        """
        # Merge entries
        target['entries'].extend(source['entries'])
        target['total_entries'] += source['total_entries']
        target['total_pages'] += source['total_pages']
        target['all_pages'].extend(source['all_pages'])
        target['all_pages'].sort()

        # Use the patient info with more complete data
        info1 = target['patient_info']
        info2 = source['patient_info']

        # Count non-empty fields for each
        count1 = sum(1 for v in info1.values() if v and v.strip())
        count2 = sum(1 for v in info2.values() if v and v.strip())

        if count2 > count1:
            target['patient_info'] = info2

    def calculate_confidence(self, patient_data: Dict[str, Any], pages_data: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for patient identification.

        Args:
            patient_data: Patient identification data
            pages_data: List of page data used for identification

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence on field completeness
        required_fields = ['full_name', 'dob']
        filled_fields = sum(1 for field in required_fields if patient_data.get(field, '').strip())

        base_confidence = filled_fields / len(required_fields)

        # Factor in OCR confidence from pages
        ocr_confidence = sum(page.get('confidence', 0.0) for page in pages_data) / len(pages_data) if pages_data else 0.0

        # Combine scores
        final_confidence = (base_confidence * 0.7) + (ocr_confidence * 0.3)

        return min(1.0, max(0.0, final_confidence))

    def calculate_group_confidence(self, entries: List[Dict[str, Any]], pages: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for a patient group.

        Args:
            entries: List of patient entries
            pages: All pages data

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not entries:
            return 0.0

        # Calculate confidence for each entry
        entry_confidences = []
        for entry in entries:
            # Get OCR confidence for this entry's pages
            entry_pages = [p for p in pages if p['page_number'] in entry['pages']]
            if entry_pages:
                ocr_confidence = sum(p.get('confidence', 0.0) for p in entry_pages) / len(entry_pages)
                entry_confidences.append(ocr_confidence)

        # Average entry confidences
        avg_confidence = sum(entry_confidences) / len(entry_confidences) if entry_confidences else 0.0

        # Bonus for multiple entries (stronger evidence)
        multi_entry_bonus = min(0.1, (len(entries) - 1) * 0.05)

        return min(1.0, avg_confidence + multi_entry_bonus)

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
            'processing_timestamp': datetime.now().isoformat(),
            'total_pages': total_pages,
            'processing_time_seconds': processing_time,
            'patient_groups': {},
            'unassigned_pages': {
                'pages': list(range(1, total_pages + 1)),
                'reasons': ['No patient information found']
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
    """Main function for testing the algorithm"""
    try:
        matcher = NameDobMatcher()
        # Test with sample data
        print(f"{matcher.algorithm_name} algorithm ready for use")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())