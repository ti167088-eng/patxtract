"""
Similarity Calculator

Provides various algorithms for calculating similarity scores between patient data,
including string similarity, numeric similarity, and weighted scoring for comprehensive
patient matching across different documents.
"""

import math
import re
from typing import Dict, List, Tuple, Optional, Any
from difflib import SequenceMatcher
from collections import defaultdict

class SimilarityCalculator:
    """
    Utility class for calculating similarity scores between patient data
    using various algorithms and weighting strategies.
    """

    def __init__(self):
        self.logger = __import__('logging').getLogger(__name__)

        # Character similarity mapping for OCR errors
        self.char_similarity = {
            '0': ['o', 'O'],
            '1': ['l', 'i', 'I', '|'],
            '2': ['z', 'Z'],
            '5': ['s', 'S'],
            '8': ['b'],
            'o': ['0'],
            'l': ['1', 'i', 'I'],
            'i': ['1', 'l', '|'],
            'z': ['2'],
            's': ['5']
        }

        # Field weights for patient matching
        self.field_weights = {
            'full_name': 40,
            'first_name': 20,
            'last_name': 20,
            'dob': 30,
            'address_full': 20,
            'city': 5,
            'state': 3,
            'postal_code': 5,
            'phone': 10,
            'mobile_landline': 5,
            'email': 8,
            'account_number': 15,
            'ssn': 25  # Social Security Number (if present)
        }

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Levenshtein distance (lower is more similar)
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def levenshtein_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate normalized Levenshtein similarity between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0 (1.0 = identical)
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        distance = self.levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)

    def jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate Jaro-Winkler similarity between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0 (1.0 = identical)
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        s1 = s1.lower()
        s2 = s2.lower()

        # Calculate Jaro distance
        len_s1 = len(s1)
        len_s2 = len(s2)
        match_distance = max(len_s1, len_s2) // 2 - 1

        s1_matches = [False] * len_s1
        s2_matches = [False] * len_s2

        matches = 0
        transpositions = 0

        # Find matches
        for i in range(len_s1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len_s2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        # Count transpositions
        k = 0
        for i in range(len_s1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (matches / len_s1 + matches / len_s2 + (matches - transpositions / 2) / matches) / 3

        # Apply Winkler prefix scaling
        prefix_length = 0
        for i in range(min(4, len_s1, len_s2)):
            if s1[i] == s2[i]:
                prefix_length += 1
            else:
                break

        return jaro + (0.1 * prefix_length * (1 - jaro))

    def fuzzy_string_similarity(self, s1: str, s2: str, algorithm: str = 'jaro_winkler') -> float:
        """
        Calculate fuzzy string similarity using specified algorithm.

        Args:
            s1: First string
            s2: Second string
            algorithm: Algorithm to use ('jaro_winkler', 'levenshtein', 'difflib')

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if algorithm == 'jaro_winkler':
            return self.jaro_winkler_similarity(s1, s2)
        elif algorithm == 'levenshtein':
            return self.levenshtein_similarity(s1, s2)
        elif algorithm == 'difflib':
            return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def ocr_aware_string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity with OCR error awareness.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Make strings same length by padding shorter one
        max_len = max(len(s1), len(s2))
        s1 = s1.ljust(max_len)
        s2 = s2.ljust(max_len)

        matches = 0
        total_comparisons = 0

        for i in range(max_len):
            char1 = s1[i].lower()
            char2 = s2[i].lower()

            if char1 == char2:
                matches += 1
            elif char1 in self.char_similarity and char2 in self.char_similarity[char1]:
                # Characters are similar based on OCR confusion patterns
                matches += 0.8  # Partial credit for similar characters

            total_comparisons += 1

        return matches / total_comparisons if total_comparisons > 0 else 0.0

    def name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two patient names.

        Args:
            name1: First patient name
            name2: Second patient name

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not name1 and not name2:
            return 1.0
        if not name1 or not name2:
            return 0.0

        # Normalize names
        norm1 = ' '.join(name1.lower().split())
        norm2 = ' '.join(name2.lower().split())

        # Calculate multiple similarity metrics
        jaro_score = self.jaro_winkler_similarity(norm1, norm2)
        ocr_score = self.ocr_aware_string_similarity(norm1, norm2)
        token_score = self.token_similarity(norm1, norm2)

        # Weighted combination
        return (jaro_score * 0.4 + ocr_score * 0.3 + token_score * 0.3)

    def token_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate token-based similarity between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())

        if not tokens1 and not tokens2:
            return 1.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union) if union else 0.0

    def date_similarity(self, date1: str, date2: str) -> float:
        """
        Calculate similarity between two dates.

        Args:
            date1: First date string
            date2: Second date string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not date1 and not date2:
            return 1.0
        if not date1 or not date2:
            return 0.0

        # Normalize dates to YYYY-MM-DD format
        norm1 = self.normalize_date(date1)
        norm2 = self.normalize_date(date2)

        if norm1 == norm2:
            return 1.0

        return 0.0

    def normalize_date(self, date_str: str) -> str:
        """
        Normalize date string to YYYY-MM-DD format.

        Args:
            date_str: Input date string

        Returns:
            Normalized date string
        """
        if not date_str:
            return ""

        # Extract digits only
        digits = re.sub(r'[^\d]', '', date_str)

        if len(digits) == 8:
            try:
                year = int(digits[4:8])
                month = int(digits[0:2])
                day = int(digits[2:4])

                if year < 100:
                    year += 1900 if year >= 50 else 2000

                if 1 <= month <= 12 and 1 <= day <= 31:
                    return f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass

        return date_str.lower().strip()

    def phone_similarity(self, phone1: str, phone2: str) -> float:
        """
        Calculate similarity between two phone numbers.

        Args:
            phone1: First phone number
            phone2: Second phone number

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not phone1 and not phone2:
            return 1.0
        if not phone1 or not phone2:
            return 0.0

        # Extract digits only
        digits1 = re.sub(r'[^\d]', '', phone1)
        digits2 = re.sub(r'[^\d]', '', phone2)

        # Remove leading 1 (country code)
        if len(digits1) == 11 and digits1.startswith('1'):
            digits1 = digits1[1:]
        if len(digits2) == 11 and digits2.startswith('1'):
            digits2 = digits2[1:]

        return 1.0 if digits1 == digits2 else 0.0

    def address_similarity(self, addr1: str, addr2: str) -> float:
        """
        Calculate similarity between two addresses.

        Args:
            addr1: First address
            addr2: Second address

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not addr1 and not addr2:
            return 1.0
        if not addr1 or not addr2:
            return 0.0

        # Normalize addresses
        norm1 = ' '.join(addr1.lower().split())
        norm2 = ' '.join(addr2.lower().split())

        # Combine multiple similarity metrics
        jaro_score = self.jaro_winkler_similarity(norm1, norm2)
        token_score = self.token_similarity(norm1, norm2)
        ocr_score = self.ocr_aware_string_similarity(norm1, norm2)

        return (jaro_score * 0.4 + token_score * 0.3 + ocr_score * 0.3)

    def calculate_patient_similarity(self, patient1: Dict[str, Any], patient2: Dict[str, Any],
                                   custom_weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate overall similarity score between two patient records.

        Args:
            patient1: First patient data dictionary
            patient2: Second patient data dictionary
            custom_weights: Optional custom field weights

        Returns:
            Overall similarity score between 0.0 and 1.0
        """
        weights = custom_weights or self.field_weights

        total_score = 0.0
        total_weight = 0.0

        for field, weight in weights.items():
            value1 = patient1.get(field, '')
            value2 = patient2.get(field, '')

            if not value1 and not value2:
                continue  # Skip empty fields for both patients

            field_score = 0.0

            # Calculate field-specific similarity
            if field in ['full_name', 'first_name', 'last_name']:
                field_score = self.name_similarity(str(value1), str(value2))
            elif field == 'dob':
                field_score = self.date_similarity(str(value1), str(value2))
            elif field in ['address_full', 'city', 'state', 'postal_code']:
                field_score = self.address_similarity(str(value1), str(value2))
            elif field in ['phone', 'mobile_landline']:
                field_score = self.phone_similarity(str(value1), str(value2))
            else:
                # Default to string similarity for other fields
                field_score = self.ocr_aware_string_similarity(str(value1), str(value2))

            total_score += field_score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def find_best_match(self, target_patient: Dict[str, Any], candidates: List[Dict[str, Any]],
                       threshold: float = 0.7) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the best matching patient from a list of candidates.

        Args:
            target_patient: Patient to match
            candidates: List of candidate patients
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (best_match, similarity_score) or (None, 0.0) if no match above threshold
        """
        if not candidates:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for candidate in candidates:
            score = self.calculate_patient_similarity(target_patient, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score >= threshold:
            return best_match, best_score
        else:
            return None, 0.0

    def calculate_group_confidence(self, patient_group: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for a group of patient pages.

        Args:
            patient_group: List of patient data from multiple pages

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if len(patient_group) <= 1:
            return 1.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(patient_group)):
            for j in range(i + 1, len(patient_group)):
                sim = self.calculate_patient_similarity(patient_group[i], patient_group[j])
                similarities.append(sim)

        # Average similarity as confidence
        return sum(similarities) / len(similarities) if similarities else 1.0