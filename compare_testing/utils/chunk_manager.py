"""
Document Chunking Manager

Handles intelligent document chunking and patient result consolidation
for processing large documents that exceed AI model context limits.
"""

import logging
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import hashlib

from .token_estimator import TokenEstimator


logger = logging.getLogger(__name__)


class ChunkManager:
    """Manages document chunking and result consolidation."""

    def __init__(self, model_name: str):
        """
        Initialize the chunk manager.

        Args:
            model_name: Name of the AI model being used
        """
        self.model_name = model_name
        self.token_estimator = TokenEstimator()
        self.chunk_config = self.token_estimator.get_optimal_chunk_config(model_name)
        logger.info(f"Initialized ChunkManager for {model_name}: {self.chunk_config}")

    def analyze_document(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze document to determine if chunking is needed.

        Args:
            pages: List of page dictionaries

        Returns:
            Analysis results with chunking recommendations
        """
        needs_chunking, total_tokens = self.token_estimator.needs_chunking(
            pages, self.token_estimator.get_model_context_limits()[self.model_name]
        )

        model_limit = self.token_estimator.get_model_context_limits()[self.model_name]
        usable_limit = int(model_limit * 0.8)

        analysis = {
            'total_pages': len(pages),
            'total_tokens': total_tokens,
            'model_context_limit': model_limit,
            'usable_limit': usable_limit,
            'needs_chunking': needs_chunking,
            'chunk_config': self.chunk_config.copy()
        }

        if needs_chunking:
            pages_per_chunk = self.token_estimator.calculate_pages_per_chunk(
                pages, self.chunk_config['max_tokens_per_chunk']
            )
            estimated_chunks = max(1, (len(pages) + pages_per_chunk - 1) // pages_per_chunk)

            analysis.update({
                'pages_per_chunk': pages_per_chunk,
                'estimated_chunks': estimated_chunks,
                'chunk_overlap_ratio': self.chunk_config['overlap_ratio']
            })

        return analysis

    def create_chunks(self, pages: List[Dict[str, Any]]) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Create document chunks for processing.

        Args:
            pages: List of page dictionaries

        Returns:
            Tuple of (chunks, analysis_info)
        """
        import time
        start_time = time.time()
        timeout_seconds = 60  # Maximum time to spend chunking

        # Validate input
        if not pages:
            logger.error("No pages provided for chunking")
            return [], {}

        # Progress tracking
        logger.info(f"Starting chunk creation for {len(pages)} pages")

        try:
            analysis = self.analyze_document(pages)

            if not analysis['needs_chunking']:
                logger.info("Document fits in single chunk, no chunking needed")
                return [pages], analysis

            pages_per_chunk = analysis.get('pages_per_chunk', 50)
            overlap_ratio = analysis.get('chunk_overlap_ratio', 0.1)

            # Validate chunking parameters
            if pages_per_chunk <= 0:
                logger.error(f"Invalid pages_per_chunk: {pages_per_chunk}, using fallback")
                pages_per_chunk = min(50, len(pages))

            # Create chunks manually with better error handling
            chunks = []
            overlap_size = max(1, int(pages_per_chunk * overlap_ratio))
            current_position = 0

            while current_position < len(pages):
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    logger.warning("Chunking operation timed out, using what we have")
                    break

                # Calculate chunk end position
                chunk_end = min(current_position + pages_per_chunk, len(pages))

                # Extract chunk
                chunk = pages[current_position:chunk_end]

                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
                    logger.debug(f"Created chunk {len(chunks)}: pages {current_position+1} to {chunk_end}")

                # Move to next position with overlap
                current_position = chunk_end - overlap_size

                # Prevent infinite loop
                if current_position >= len(pages) - overlap_size:
                    break

            # Validate chunks
            if not chunks:
                logger.warning("No valid chunks created, falling back to single chunk")
                chunks = [pages]

            logger.info(f"Successfully created {len(chunks)} chunks in {time.time() - start_time:.2f}s")
            logger.info(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")

            return chunks, analysis

        except Exception as e:
            logger.error(f"Error during chunk creation: {e}")
            # Fallback: return single chunk
            logger.warning("Falling back to single chunk due to error")
            return [pages], {'needs_chunking': False, 'error': str(e)}

    def merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge patient identification results from multiple chunks.

        Args:
            chunk_results: List of result dictionaries from each chunk

        Returns:
            Consolidated patient identification results
        """
        if not chunk_results:
            return self._create_empty_result()

        if len(chunk_results) == 1:
            return chunk_results[0]

        logger.info(f"Merging results from {len(chunk_results)} chunks")

        # Extract patient groups from all chunks
        all_patient_groups = {}
        patient_signatures = {}
        chunk_mapping = {}

        # Process each chunk's results
        for chunk_idx, chunk_result in enumerate(chunk_results):
            chunk_patient_groups = chunk_result.get('patient_groups', {})

            for patient_id, patient_data in chunk_patient_groups.items():
                # Create patient signature for cross-chunk matching
                signature = self._create_patient_signature(patient_data)
                original_patient_id = patient_id

                # Check if this patient matches an existing one across chunks
                matching_patient_id = self._find_matching_patient(
                    signature, patient_signatures, all_patient_groups
                )

                if matching_patient_id:
                    # Merge with existing patient
                    self._merge_patient_data(
                        all_patient_groups[matching_patient_id],
                        patient_data,
                        chunk_idx
                    )
                    chunk_mapping[original_patient_id] = matching_patient_id
                else:
                    # Add as new patient
                    new_patient_id = self._generate_new_patient_id(all_patient_groups)
                    all_patient_groups[new_patient_id] = patient_data.copy()
                    patient_signatures[new_patient_id] = signature
                    chunk_mapping[original_patient_id] = new_patient_id

        # Handle cross-chunk entry continuity
        self._resolve_cross_chunk_entries(all_patient_groups, chunk_results)

        # Create consolidated result
        consolidated_result = self._create_consolidated_result(
            chunk_results, all_patient_groups
        )

        logger.info(f"Consolidated into {len(all_patient_groups)} unique patients")

        return consolidated_result

    def _create_patient_signature(self, patient_data: Dict[str, Any]) -> str:
        """
        Create a unique signature for patient matching across chunks.

        Args:
            patient_data: Patient data dictionary

        Returns:
            Patient signature string
        """
        patient_info = patient_data.get('patient_info', {})

        # Normalize name
        name = patient_info.get('full_name', '').lower().strip()
        name = ' '.join(name.split())  # Remove extra whitespace

        # Normalize DOB
        dob = patient_info.get('dob', '')
        if dob:
            # Standardize DOB format
            dob = dob.replace('/', '-').replace('.', '-').replace(' ', '-')
            dob_parts = [part.strip() for part in dob.split('-') if part.strip()]
            if len(dob_parts) == 3:
                # Try to standardize to YYYY-MM-DD
                if len(dob_parts[2]) == 2:  # MM-DD-YY format
                    year = '20' + dob_parts[2] if int(dob_parts[2]) < 50 else '19' + dob_parts[2]
                    dob = f"{year}-{dob_parts[0].zfill(2)}-{dob_parts[1].zfill(2)}"
                elif len(dob_parts[0]) == 4:  # YYYY-MM-DD format
                    dob = f"{dob_parts[0]}-{dob_parts[1].zfill(2)}-{dob_parts[2].zfill(2)}"

        # Create signature (name + DOB for primary matching)
        signature = f"{name}|{dob}"

        return signature

    def save_chunking_metadata(self, analysis: Dict[str, Any], chunks: List[List[Dict[str, Any]]],
                            algorithm_name: str, timestamp: str) -> str:
        """
        Save chunking metadata to separate file as expected by user.

        Args:
            analysis: Chunking analysis dictionary
            chunks: List of page chunks
            algorithm_name: Name of the algorithm
            timestamp: Timestamp string

        Returns:
            Path to saved metadata file
        """
        # CRITICAL FIX: Generate chunking metadata file as expected by user
        import json
        import os
        from datetime import datetime

        metadata_filename = f"chunking_metadata_{algorithm_name}_{timestamp}.json"
        metadata_path = self.output_dir / metadata_filename

        metadata = {
            'algorithm_name': algorithm_name,
            'timestamp': timestamp,
            'chunking_analysis': analysis,
            'chunks_info': [
                {
                    'chunk_index': i,
                    'pages_count': len(chunk),
                    'page_numbers': [page['page_number'] for page in chunk],
                    'estimated_tokens': sum(len(page.get('text', '')) for page in chunk)
                }
                for i, chunk in enumerate(chunks)
            ],
            'total_chunks': len(chunks),
            'original_total_pages': analysis.get('total_pages', 0)
        }

        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Chunking metadata saved to: {metadata_path}")
            return metadata_path
        except Exception as e:
            logger.error(f"Error saving chunking metadata: {e}")
            return None

    def _find_matching_patient(self, signature: str, patient_signatures: Dict[str, str],
                             all_patient_groups: Dict[str, Any]) -> str:
        """
        Find matching patient across chunks using signature matching.

        Args:
            signature: Patient signature to match
            patient_signatures: Dictionary of patient_id -> signature
            all_patient_groups: Dictionary of patient_id -> patient_data

        Returns:
            Matching patient ID or None if no match found
        """
        # Exact match first
        for patient_id, existing_signature in patient_signatures.items():
            if signature == existing_signature:
                return patient_id

        # Fuzzy matching for name variations
        name_parts = signature.split('|')[0].split()
        for patient_id, existing_signature in patient_signatures.items():
            existing_name_parts = existing_signature.split('|')[0].split()

            # Check for significant name overlap
            common_words = set(name_parts) & set(existing_name_parts)
            if len(common_words) >= min(2, len(name_parts) - 1):
                # Additional DOB check if available
                dob = signature.split('|')[1] if len(signature.split('|')) > 1 else ''
                existing_dob = existing_signature.split('|')[1] if len(existing_signature.split('|')) > 1 else ''

                if dob and existing_dob and dob == existing_dob:
                    return patient_id
                elif not dob or not existing_dob:
                    # If DOB is missing for either, rely on name matching
                    return patient_id

        return None

    def _merge_patient_data(self, existing_patient: Dict[str, Any], new_patient: Dict[str, Any],
                          chunk_idx: int):
        """
        Merge patient data from different chunks.

        Args:
            existing_patient: Existing patient data
            new_patient: New patient data to merge
            chunk_idx: Chunk index for tracking
        """
        # Merge entries (handling potential overlaps)
        existing_entries = existing_patient.get('entries', [])
        new_entries = new_patient.get('entries', [])

        # Combine and sort entries by page number
        all_entries = existing_entries + new_entries

        # Remove duplicate entries (same page ranges)
        unique_entries = []
        seen_page_sets = set()

        for entry in all_entries:
            page_tuple = tuple(sorted(entry['pages']))
            if page_tuple not in seen_page_sets:
                unique_entries.append(entry)
                seen_page_sets.add(page_tuple)

        # Sort entries by first page number
        unique_entries.sort(key=lambda x: min(x['pages']))

        # Update patient data
        existing_patient['entries'] = unique_entries
        existing_patient['total_entries'] = len(unique_entries)

        # Update all pages list
        all_pages = set()
        for entry in unique_entries:
            all_pages.update(entry['pages'])
        existing_patient['all_pages'] = sorted(list(all_pages))
        existing_patient['total_pages'] = len(all_pages)

        # Merge patient info (prefer more complete data)
        existing_info = existing_patient.get('patient_info', {})
        new_info = new_patient.get('patient_info', {})

        for field, value in new_info.items():
            if value and (not existing_info.get(field) or len(str(value)) > len(str(existing_info.get(field, '')))):
                existing_info[field] = value

        existing_patient['patient_info'] = existing_info

        # Update confidence (use average of confidence scores)
        existing_confidence = existing_patient.get('group_confidence', 0.0)
        new_confidence = new_patient.get('group_confidence', 0.0)
        avg_confidence = (existing_confidence + new_confidence) / 2
        existing_patient['group_confidence'] = avg_confidence

    def _resolve_cross_chunk_entries(self, all_patient_groups: Dict[str, Any],
                                   chunk_results: List[Dict[str, Any]]):
        """
        Resolve entry continuity across chunk boundaries.

        Args:
            all_patient_groups: Consolidated patient groups
            chunk_results: Original chunk results for reference
        """
        for patient_id, patient_data in all_patient_groups.items():
            entries = patient_data.get('entries', [])

            if len(entries) <= 1:
                continue

            # Check if entries are consecutive or should be merged
            merged_entries = []
            current_entry = entries[0]

            for next_entry in entries[1:]:
                current_pages = set(current_entry['pages'])
                next_pages = set(next_entry['pages'])

                # Check if entries are close enough to merge
                min_current = min(current_pages)
                max_current = max(current_pages)
                min_next = min(next_pages)

                # If entries are within 2 pages of each other, consider merging
                if min_next - max_current <= 2:
                    # Merge entries
                    merged_pages = sorted(list(current_pages | next_pages))
                    merged_entry = {
                        'entry_id': current_entry['entry_id'],
                        'pages': merged_pages,
                        'page_range': f"{min(merged_pages)}-{max(merged_pages)}",
                        'confidence': (current_entry.get('confidence', 0.0) + next_entry.get('confidence', 0.0)) / 2,
                        'supporting_data': current_entry.get('supporting_data', {})
                    }
                    current_entry = merged_entry
                else:
                    merged_entries.append(current_entry)
                    current_entry = next_entry

            merged_entries.append(current_entry)

            # Update patient data with merged entries
            patient_data['entries'] = merged_entries
            patient_data['total_entries'] = len(merged_entries)

    def _generate_new_patient_id(self, existing_patients: Dict[str, Any]) -> str:
        """
        Generate a new unique patient ID.

        Args:
            existing_patients: Existing patient groups

        Returns:
            New patient ID
        """
        if not existing_patients:
            return "patient_001"

        max_num = 0
        for patient_id in existing_patients.keys():
            if patient_id.startswith("patient_"):
                try:
                    num = int(patient_id.split("_")[1])
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    continue

        return f"patient_{max_num + 1:03d}"

    def _create_consolidated_result(self, chunk_results: List[Dict[str, Any]],
                                  all_patient_groups: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the final consolidated result.

        Args:
            chunk_results: Results from individual chunks
            all_patient_groups: Merged patient groups

        Returns:
            Consolidated result dictionary
        """
        if not chunk_results:
            return self._create_empty_result()

        # Base the result on the first chunk result
        base_result = chunk_results[0].copy()

        # Update patient groups
        base_result['patient_groups'] = all_patient_groups

        # Calculate new summary
        total_patients = len(all_patient_groups)
        total_entries = sum(
            patient.get('total_entries', 1)
            for patient in all_patient_groups.values()
        )
        total_assigned_pages = sum(
            patient.get('total_pages', 0)
            for patient in all_patient_groups.values()
        )

        # Get total pages from first chunk result
        total_pages = base_result.get('total_pages', 0)

        # Calculate average confidence
        confidences = [
            patient.get('group_confidence', 0.0)
            for patient in all_patient_groups.values()
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Update summary
        base_result['summary'] = {
            'total_patients': total_patients,
            'total_entries': total_entries,
            'assigned_pages': total_assigned_pages,
            'unassigned_pages': total_pages - total_assigned_pages,
            'average_confidence': avg_confidence
        }

        # Add chunking metadata
        base_result['chunking_info'] = {
            'processed_in_chunks': True,
            'num_chunks': len(chunk_results),
            'model_name': self.model_name
        }

        return base_result

    def _create_empty_result(self) -> Dict[str, Any]:
        """
        Create an empty result structure.

        Returns:
            Empty result dictionary
        """
        return {
            'patient_groups': {},
            'unassigned_pages': {
                'pages': [],
                'reasons': ['No processing performed']
            },
            'summary': {
                'total_patients': 0,
                'total_entries': 0,
                'assigned_pages': 0,
                'unassigned_pages': 0,
                'average_confidence': 0.0
            }
        }