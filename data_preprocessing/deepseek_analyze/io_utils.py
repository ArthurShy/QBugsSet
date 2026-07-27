#!/usr/bin/env python3
"""IO utilities: data loading and result saving."""

import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from .data_types import BugSample, AnalysisResult
from .validators import VALID_LIFECYCLE_STAGES

class DataLoader:
    """Data loader."""
    
    @staticmethod
    def load_samples(json_path: Path, limit: Optional[int] = None, silent: bool = False) -> List[BugSample]:
        if not silent:
            logging.info(f"Loading dataset: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples_data = data.get('samples', [])
        total_samples = len(samples_data)
        
        if limit is not None and limit > 0:
            samples_data = samples_data[:limit]
            if not silent:
                logging.info(f"Limited to {len(samples_data)}/{total_samples} samples")
        else:
            if not silent:
                logging.info(f"Loaded {total_samples} samples")
        
        samples = [BugSample.from_dict(s) for s in samples_data]
        return samples


class ResultSaver:
    """Result saver."""
    
    def __init__(self, output_path: Path, load_existing: bool = True):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []
        self._lock = threading.Lock()
        self._lifecycle_distribution: Dict = {}
        self._batch_counter: int = 0
        
        if load_existing and self.output_path.exists():
            self._load_existing_results()
    
    def _load_existing_results(self) -> None:
        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            existing_results = data.get('results', [])
            for idx, result in enumerate(existing_results):
                if '_sort_index' not in result:
                    result['_sort_index'] = idx
                if 'sample_id' in result and 'id' not in result:
                    result['id'] = result.pop('sample_id')
                if 'is_quantum_related' in result and 'quantum_specific' not in result:
                    result['quantum_specific'] = result.pop('is_quantum_related')
                if 'bug_type' in result and 'submodule' not in result:
                    result['submodule'] = result.pop('bug_type')
            
            self.results = existing_results
            metadata = data.get('metadata', {})
            self._lifecycle_distribution = metadata.get('lifecycle_distribution', {})
            
            logging.info(f"Loaded {len(self.results)} existing results")
        except Exception as e:
            logging.warning(f"Failed to load existing results: {e}")
            self.results = []
    
    def get_lifecycle_distribution(self) -> Dict:
        return self._lifecycle_distribution
    
    def get_processed_sample_ids(self) -> set:
        result_set = set()
        for result in self.results:
            sample_id = result.get('id') or result.get('sample_id')
            if sample_id and result.get('success', False):
                result_set.add(str(sample_id))
        return result_set
    
    def get_all_processed_sample_ids(self) -> set:
        result_set = set()
        for result in self.results:
            sample_id = result.get('id') or result.get('sample_id')
            if sample_id:
                result_set.add(str(sample_id))
        return result_set
    
    def _count_unique_submodules(self) -> int:
        unique_submodules = set()
        for r in self.results:
            submodule_value = r.get('submodule') or r.get('bug_type')
            if r.get('quantum_specific') and submodule_value:
                unique_submodules.add(submodule_value)
        return len(unique_submodules)
    
    def add_result(self, result: AnalysisResult) -> None:
        result_dict = result.to_dict()
        if result._sort_index is not None:
            result_dict['_sort_index'] = result._sort_index
        
        sample_id = result_dict.get('id') or result_dict.get('sample_id')
        existing_index = None
        if sample_id:
            for i, r in enumerate(self.results):
                r_id = r.get('id') or r.get('sample_id')
                if r_id == sample_id:
                    existing_index = i
                    break
        
        if existing_index is not None:
            old_sort_index = self.results[existing_index].get('_sort_index')
            if old_sort_index is not None and result_dict.get('_sort_index') is None:
                result_dict['_sort_index'] = old_sort_index
            self.results[existing_index] = result_dict
        else:
            self.results.append(result_dict)
    
    def update_bug_types_after_merge(self, operations: List[Dict]) -> int:
        if not operations:
            return 0
        
        type_mapping = {}
        for op in operations:
            op_type = op.get('type')
            lifecycle_stage = op.get('lifecycle_stage', '')
            if op_type == 'merge':
                from_types = op.get('from', [])
                to_type = op.get('to', '')
                if lifecycle_stage:
                    for from_type in from_types:
                        type_mapping[(from_type, lifecycle_stage)] = to_type
            elif op_type == 'rename':
                from_name = op.get('from', '')
                to_name = op.get('to', '')
                if lifecycle_stage:
                    type_mapping[(from_name, lifecycle_stage)] = to_name
        
        if not type_mapping:
            return 0
        
        updated_count = 0
        for result in self.results:
            old_bug_type = result.get('submodule') or result.get('bug_type')
            lifecycle_stage = result.get('lifecycle_stage')
            if old_bug_type and lifecycle_stage:
                key = (old_bug_type, lifecycle_stage)
                if key in type_mapping:
                    result['submodule'] = type_mapping[key]
                    result.pop('bug_type', None)
                    updated_count += 1
        return updated_count
    
    def save(self) -> None:
        def get_sort_key(r):
            sample_id = r.get('id') or r.get('sample_id', '')
            if isinstance(sample_id, int):
                return sample_id
            if isinstance(sample_id, str):
                if sample_id.isdigit():
                    return int(sample_id)
                if sample_id.startswith('single_'):
                    try: return int(sample_id.replace('single_', ''))
                    except ValueError: return float('inf')
            return float('inf')
        
        self.results.sort(key=get_sort_key)
        
        successful = sum(1 for r in self.results if r['success'])
        failed = sum(1 for r in self.results if not r['success'])
        quantum_bug_fixes = sum(1 for r in self.results 
                              if r.get('change_category') == 'Bug Fix' 
                              and r.get('quantum_specific'))
        
        lifecycle_distribution_temp = {}
        for r in self.results:
            submodule_value = r.get('submodule') or r.get('bug_type')
            if r.get('success') and submodule_value:
                bug_type = submodule_value
                lifecycle_stage = r.get('lifecycle_stage', '')
                if not lifecycle_stage: continue
                
                if lifecycle_stage not in lifecycle_distribution_temp:
                    lifecycle_distribution_temp[lifecycle_stage] = {}
                
                if bug_type not in lifecycle_distribution_temp[lifecycle_stage]:
                    lifecycle_distribution_temp[lifecycle_stage][bug_type] = 0
                lifecycle_distribution_temp[lifecycle_stage][bug_type] += 1
        
        lifecycle_distribution = {}
        for stage in VALID_LIFECYCLE_STAGES:
            if stage in lifecycle_distribution_temp:
                submodules_dict = lifecycle_distribution_temp[stage]
                from .validators import LIFECYCLE_HIERARCHY
                ordered_submodules = {}
                
                if stage in LIFECYCLE_HIERARCHY:
                    for submodule in LIFECYCLE_HIERARCHY[stage]:
                        if submodule in submodules_dict:
                            ordered_submodules[submodule] = submodules_dict[submodule]
                    remaining = {k: v for k, v in submodules_dict.items() if k not in ordered_submodules}
                    for submodule in sorted(remaining.keys()):
                        ordered_submodules[submodule] = remaining[submodule]
                else:
                    ordered_submodules = dict(sorted(submodules_dict.items()))
                
                lifecycle_distribution[stage] = ordered_submodules

        prompt_cache_hit_total = 0
        prompt_cache_miss_total = 0
        for r in self.results:
            usage = r.get('token_usage') or {}
            prompt_cache_hit_total += int(usage.get('prompt_cache_hit_tokens') or 0)
            prompt_cache_miss_total += int(usage.get('prompt_cache_miss_tokens') or 0)
        
        FIELD_ORDER = [
            "id", "framework", "parent_file_path",
            "commit_message", "change_category", "repository", "commit_sha",
            "success", "lifecycle_stage", "submodule", "quantum_specific",
            "quantum_reason", "lifecycle_reason",
            "error", "api_response_time", "token_usage", "reason"
        ]
        FIELD_MAPPING = {
            'sample_id': 'id',
            'is_quantum_related': 'quantum_specific', 
            'bug_type': 'submodule',
            'bug_reason': 'reason'
        }
        FIELDS_TO_REMOVE = {'_sort_index', 'raw_response'}
        
        results_for_output = []
        for r in self.results:
            result_copy = {}
            for k, v in r.items():
                new_key = FIELD_MAPPING.get(k, k)
                if new_key not in FIELDS_TO_REMOVE:
                    result_copy[new_key] = v
            
            ordered = {}
            for key in FIELD_ORDER:
                if key in result_copy:
                    ordered[key] = result_copy[key]
            for key, value in result_copy.items():
                if key not in ordered:
                    ordered[key] = value
            
            results_for_output.append(ordered)
        
        change_category_distribution = {}
        for r in results_for_output:
            cat = r.get('change_category', 'unknown')
            if cat:
                change_category_distribution[cat] = change_category_distribution.get(cat, 0) + 1
        
        output_data = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_samples": len(self.results),
                "successful": successful,
                "failed": failed,
                "quantum_bug_fixes": quantum_bug_fixes,
                "prompt_cache_hit_tokens": prompt_cache_hit_total,
                "prompt_cache_miss_tokens": prompt_cache_miss_total
            },
            "results": results_for_output
        }
        if change_category_distribution:
            output_data["metadata"]["change_category_distribution"] = change_category_distribution
        if lifecycle_distribution:
            output_data["metadata"]["lifecycle_distribution"] = lifecycle_distribution
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def save_incremental(self, result: AnalysisResult) -> None:
        with self._lock:
            self.add_result(result)
            self.save()
    
    def add_result_batch(self, result: AnalysisResult, batch_size: int = 50) -> bool:
        with self._lock:
            self.add_result(result)
            self._batch_counter += 1
            if self._batch_counter >= batch_size:
                self.save()
                self._batch_counter = 0
                return True
            return False
    
    def flush(self) -> None:
        with self._lock:
            if self.results:
                self.save()


class ErrorLogger:
    """Error logger for LLM analysis."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.errors: List[Dict] = []
        self._lock = threading.Lock()
        
        if self.output_path.exists():
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.errors = data.get('errors', [])
                logging.info(f"Loaded {len(self.errors)} existing error logs")
            except Exception as e:
                logging.warning(f"Failed to load existing error logs: {e}")
    
    def log_type_classification_error(self, sample_id: str, error_type: str, error_message: str, llm_response: str, sample_info: Optional[Dict] = None) -> None:
        error_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_category": "type_classification",
            "sample_id": sample_id,
            "error_type": error_type,
            "error_message": error_message,
            "llm_response": llm_response,
            "sample_info": sample_info or {}
        }
        with self._lock:
            self.errors.append(error_entry)
            self._save()
    
    def log_intra_class_merge_error(self, error_message: str, llm_response: str, current_types: List[Dict], recent_samples: Optional[List[str]] = None) -> None:
        error_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_category": "intra_class_merge",
            "error_message": error_message,
            "llm_response": llm_response,
            "current_types": [bt['name'] for bt in current_types],
            "current_types_count": len(current_types),
            "recent_samples": recent_samples or []
        }
        with self._lock:
            self.errors.append(error_entry)
            self._save()
    
    def log_inter_class_merge_error(self, error_message: str, llm_response: str, parsed_result: Optional[Dict], current_types: List[Dict], operations: Optional[List[Dict]] = None, validation_details: Optional[Dict] = None) -> None:
        error_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_category": "inter_class_merge",
            "error_message": error_message,
            "llm_response": llm_response,
            "parsed_result": parsed_result,
            "current_types": [bt['name'] for bt in current_types],
            "current_types_count": len(current_types),
            "operations": operations or [],
            "validation_details": validation_details or {}
        }
        with self._lock:
            self.errors.append(error_entry)
            self._save()
    
    def _save(self) -> None:
        try:
            output_data = {
                "metadata": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_errors": len(self.errors),
                    "error_distribution": self._get_error_distribution()
                },
                "errors": self.errors
            }
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save error logs: {e}")
    
    def _get_error_distribution(self) -> Dict[str, int]:
        distribution = {}
        for error in self.errors:
            cat = error.get('error_category', 'unknown')
            distribution[cat] = distribution.get(cat, 0) + 1
        return distribution


class NullErrorLogger:
    """Null error logger (Null Object Pattern)."""
    
    def log_type_classification_error(
        self, 
        sample_id: str, 
        error_type: str, 
        error_message: str, 
        llm_response: str, 
        sample_info: Optional[Dict] = None
    ) -> None:
        pass
    
    def log_intra_class_merge_error(
        self, 
        error_message: str, 
        llm_response: str, 
        current_types: List[Dict], 
        recent_samples: Optional[List[str]] = None
    ) -> None:
        pass
    
    def log_inter_class_merge_error(
        self, 
        error_message: str, 
        llm_response: str, 
        parsed_result: Optional[Dict], 
        current_types: List[Dict], 
        operations: Optional[List[Dict]] = None, 
        validation_details: Optional[Dict] = None
    ) -> None:
        pass

