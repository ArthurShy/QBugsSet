#!/usr/bin/env python3
"""Stage 3 Analyzer: Batch submodule classification."""

import json
import time
import logging
import threading
import copy
import os
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .data_types import (
    BugSample,
    AnalysisResult,
    REQUEST_DELAY,
)
from .io_utils import ResultSaver, ErrorLogger, NullErrorLogger
from .prompt_templates import PromptTemplate
from deepseek_analyze.validators import (
    parse_json_response,
    LIFECYCLE_HIERARCHY,
    VALID_LIFECYCLE_STAGES,
    validate_submodule,
    validate_batch_output_consistency,
    validate_stage3_output,
    validate_classification_submodules,
    validate_no_none_submodules,
)


DEFAULT_BATCH_SIZE = 10
STAGE_ORDER = {stage: i for i, stage in enumerate(VALID_LIFECYCLE_STAGES)}


class StageThreeAnalyzer:
    """Stage 3 Analyzer: Batch submodule classification."""
    
    def __init__(
        self,
        client: Any,
        prompt_template: PromptTemplate,
        result_saver: ResultSaver,
        error_logger: Optional[ErrorLogger] = None,
        test_mode: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        self.client = client
        self.prompt_template = prompt_template
        self.result_saver = result_saver
        self.error_logger = error_logger or NullErrorLogger()
        self.test_mode = test_mode
        self.batch_size = batch_size
        
        self.lifecycle_hierarchy = copy.deepcopy(LIFECYCLE_HIERARCHY)
        self.new_submodules: List[Dict] = []
    
    def _call_api_with_messages(
        self, 
        system_prompt: str, 
        user_prompt: str,
        use_json_mode: bool = True
    ) -> Dict[str, Any]:
        response_format = {"type": "json_object"} if use_json_mode else None
        llm_response = self.client.complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_format=response_format
        )
        
        return {
            "choices": [{"message": {"content": llm_response.content}}],
            "usage": llm_response.token_usage or {},
            "_response_time": llm_response.response_time or 0
        }
    
    def filter_samples_without_submodule(
        self,
        samples: List[BugSample],
        stage_two_results: List[Dict]
    ) -> Tuple[List[BugSample], List[Dict]]:
        id_to_result = {}
        for r in stage_two_results:
            rid = r.get('id') or r.get('sample_id')
            if rid:
                id_to_result[str(rid)] = r
        
        filtered_samples = []
        filtered_results = []
        
        for sample in samples:
            result = id_to_result.get(str(sample.id))
            if result is None:
                continue
            
            if not result.get('success'):
                continue
            
            lifecycle_stage = result.get('lifecycle_stage')
            submodule = result.get('submodule') or result.get('bug_type')
            
            if lifecycle_stage and lifecycle_stage != "None":
                if not submodule or submodule == "None" or not submodule.strip():
                    filtered_samples.append(sample)
                    filtered_results.append(result)
        
        return filtered_samples, filtered_results
    
    def _register_new_submodule(self, stage: str, name: str, description: str) -> bool:
        if stage not in self.lifecycle_hierarchy:
            logging.warning(f"Invalid lifecycle stage: {stage}")
            return False
        
        existing = self.lifecycle_hierarchy[stage]
        if name in existing:
            logging.info(f"Submodule '{name}' already exists in stage '{stage}'")
            return False
        
        if "None" in existing:
            idx = existing.index("None")
            existing.insert(idx, name)
        else:
            existing.append(name)
        
        self.new_submodules.append({
            "lifecycle_stage": stage,
            "name": name,
            "description": description
        })
        
        logging.info(f"New submodule: [{stage}] {name}")
        return True
    
    def analyze_batch(
        self,
        batch_samples: List[BugSample],
        batch_results: List[Dict]
    ) -> Dict[str, Any]:
        try:
            system_prompt, user_prompt = self.prompt_template.get_stage3_prompts(
                batch_samples,
                batch_results,
                self.lifecycle_hierarchy,
                dynamic_submodules=self.new_submodules
            )
            
            if self.test_mode:
                logging.info("=" * 80)
                logging.info(f"Stage 3: Batch processing {len(batch_samples)} samples")
                logging.info("=" * 80)
                logging.info(f"System prompt: {len(system_prompt)} chars")
                logging.info(f"User prompt: {len(user_prompt)} chars\n")
            response = self._call_api_with_messages(system_prompt, user_prompt)
            content = response['choices'][0]['message']['content']
            response_time = response.get('_response_time', 0)
            token_usage = response.get('usage', {})
            
            parsed_result = parse_json_response(content)
            
            if not parsed_result:
                logging.error(f"Batch analysis failed: cannot parse JSON response")
                return {
                    "success": False,
                    "error": "Cannot parse JSON response",
                    "raw_response": content,
                    "response_time": response_time,
                    "token_usage": token_usage
                }
            
            new_submodules = parsed_result.get('new_submodules', [])
            classifications = parsed_result.get('classifications', [])
            
            expected_ids = {s.id for s in batch_samples}
            classified_ids = {c.get('id') for c in classifications}
            
            missing_ids = expected_ids - classified_ids
            extra_ids = classified_ids - expected_ids
            
            if len(missing_ids) == 1 and len(extra_ids) == 1:
                missing_id = list(missing_ids)[0]
                extra_id = list(extra_ids)[0]
                logging.warning(f"Auto-fix: replacing invalid ID '{extra_id}' with missing ID '{missing_id}'")
                for clf in classifications:
                    if str(clf.get('id')) == extra_id:
                        clf['id'] = missing_id
                        break
                classified_ids = classified_ids - extra_ids | missing_ids
                missing_ids = set()
                extra_ids = set()
            
            if missing_ids:
                logging.warning(f"Samples not classified: {missing_ids}")
            temp_hierarchy = {k: v[:] for k, v in self.lifecycle_hierarchy.items()}
            batch_lifecycle_stage = None
            for r in batch_results:
                if r.get('lifecycle_stage') and r.get('lifecycle_stage') != 'Unknown':
                    batch_lifecycle_stage = r.get('lifecycle_stage')
                    break
            
            declared_new_submodule_names = set()
            for new_sub in new_submodules:
                name = new_sub.get('name')
                stage = new_sub.get('lifecycle_stage')
                if not stage or stage not in temp_hierarchy:
                    stage = batch_lifecycle_stage
                    new_sub['lifecycle_stage'] = stage
                
                if stage and name and stage in temp_hierarchy:
                    declared_new_submodule_names.add(name)
                    if name not in temp_hierarchy[stage]:
                        if "None" in temp_hierarchy[stage]:
                            idx = temp_hierarchy[stage].index("None")
                            temp_hierarchy[stage].insert(idx, name)
                        else:
                            temp_hierarchy[stage].append(name)
            
            
            id_to_stage = {str(r.get('id') or r.get('sample_id')): r.get('lifecycle_stage') for r in batch_results if r.get('id') or r.get('sample_id')}
            is_valid_subs, sub_errors = validate_classification_submodules(
                classifications, id_to_stage, temp_hierarchy
            )
            
            if not is_valid_subs:
                logging.error(f"Batch analysis failed: {len(sub_errors)} invalid submodule references")
                for err in sub_errors[:5]:
                    logging.error(f"   - {err}")
                
                return {
                    "success": False,
                    "error": f"Invalid submodule reference: {sub_errors[0]}",
                    "raw_response": content,
                    "response_time": response_time,
                    "token_usage": token_usage
                }

            for new_sub in new_submodules:
                stage = new_sub.get('lifecycle_stage')
                name = new_sub.get('name')
                desc = new_sub.get('description', '')
                if stage and name:
                    self._register_new_submodule(stage, name, desc)

            is_valid_none, none_errors = validate_no_none_submodules(classifications)
            if not is_valid_none:
                logging.error(f"{none_errors[0]}")
                return {
                    "success": False,
                    "error": f"Stage 3 must assign specific submodule: {none_errors[0]}",
                    "raw_response": content,
                    "response_time": response_time,
                    "token_usage": token_usage
                }
            
            validated_classifications = []
            for clf in classifications:
                sample_id = clf.get('id')
                submodule = clf.get('submodule')
                reason = clf.get('reason', '')
                stage = id_to_stage.get(sample_id)
                
                if not sample_id or not stage:
                    continue
                
                validated_classifications.append({
                    "id": sample_id,
                    "submodule": submodule,
                    "reason": reason,
                    "lifecycle_stage": stage
                })
            
            if self.test_mode:
                logging.info(f"Batch analysis completed:")
                logging.info(f"   New submodules: {len(new_submodules)}")
                logging.info(f"   Classifications: {len(validated_classifications)}")
            
            return {
                "success": True,
                "new_submodules": new_submodules,
                "classifications": validated_classifications,
                "raw_response": content,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_time": response_time,
                "token_usage": token_usage
            }
            
        except Exception as e:
            logging.error(f"Batch analysis failed: {e}")
            import traceback
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run(
        self,
        samples: List[BugSample],
        stage_two_results: List[Dict],
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
        target_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        processed_sample_ids = set()
        existing_results = []
        
        if output_path and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get('batch_results', [])
                    
                    for batch in existing_results:
                        if batch.get('success'):
                            processed_sample_ids.update(batch.get('sample_ids', []))
                            
                            raw_response = batch.get('raw_response', '')
                            batch_stage = batch.get('lifecycle_stage')
                            if raw_response:
                                try:
                                    parsed = parse_json_response(raw_response)
                                    if parsed:
                                        new_subs = parsed.get('new_submodules', [])
                                        for sub in new_subs:
                                            stage = sub.get('lifecycle_stage') or batch_stage
                                            name = sub.get('name')
                                            desc = sub.get('description', '')
                                            if stage and name:
                                                self._register_new_submodule(stage, name, desc)
                                except Exception:
                                    pass
                
                logging.info(f"Loaded {len(existing_results)} batches, {len(processed_sample_ids)} samples")
                logging.info(f"   Restored {len(self.new_submodules)} new submodules context")
                
            except Exception as e:
                logging.error(f"Failed to load existing results: {e}, starting fresh")
        
        filtered_samples, filtered_results = self.filter_samples_without_submodule(
            samples, stage_two_results
        )
        target_samples = []
        target_results = []
        skipped_count = 0
        
        for sample, result in zip(filtered_samples, filtered_results):
            if str(sample.id) in processed_sample_ids:
                skipped_count += 1
            else:
                target_samples.append(sample)
                target_results.append(result)
        
        if skipped_count > 0:
            logging.info(f"Skipped {skipped_count} already processed samples")
        
        if not target_samples:
            logging.info("All target samples already processed")
            return {
                "total_samples": len(filtered_samples),
                "batches_processed": len(existing_results),
                "success_batches": sum(1 for r in existing_results if r.get('success')),
                "batch_results": existing_results
            }
        
        filtered_samples = target_samples
        filtered_results = target_results
        
        stage_groups: Dict[str, List[Tuple[BugSample, Dict]]] = {}
        for sample, result in zip(filtered_samples, filtered_results):
            stage = result.get('lifecycle_stage', 'Unknown')
            if stage not in stage_groups:
                stage_groups[stage] = []
            stage_groups[stage].append((sample, result))
        
        if target_stage:
            if target_stage not in stage_groups:
                logging.warning(f"Stage '{target_stage}' not found in pending samples")
                sorted_stages = []
            else:
                logging.info(f"Processing only stage: {target_stage}")
                sorted_stages = [target_stage]
        else:
            sorted_stages = sorted(
                stage_groups.keys(), 
                key=lambda s: STAGE_ORDER.get(s, 999)
            )
        
        total_pending_samples = len(filtered_samples)
        
        if limit is not None and limit > 0:
            limited_samples = []
            limited_results = []
            remaining = limit
            
            for stage in sorted_stages:
                if remaining <= 0:
                    break
                items = stage_groups[stage]
                take_count = min(len(items), remaining)
                for i in range(take_count):
                    sample, result = items[i]
                    limited_samples.append(sample)
                    limited_results.append(result)
                remaining -= take_count
            
            stage_groups = {}
            for sample, result in zip(limited_samples, limited_results):
                stage = result.get('lifecycle_stage', 'Unknown')
                if stage not in stage_groups:
                    stage_groups[stage] = []
                stage_groups[stage].append((sample, result))
            
            sorted_stages = sorted(stage_groups.keys(), key=lambda s: STAGE_ORDER.get(s, 999))
            filtered_samples = limited_samples
            
            logging.info(f"Pending: {total_pending_samples}, limit: {limit}")
        
        logging.info(f"Stage 3 starting: {len(filtered_samples)} pending samples")
        
        all_batch_results = existing_results
        batches_processed = len(existing_results)
        
        total_remaining_batches = sum(
            (len(items) + self.batch_size - 1) // self.batch_size 
            for items in stage_groups.values()
        )
        
        batch_counter = 0
        
        for stage in sorted_stages:
            items = stage_groups[stage]
            stage_samples = [item[0] for item in items]
            stage_results = [item[1] for item in items]
            
            logging.info(f"\nProcessing stage: {stage} ({len(items)} samples)")
            
            for i in range(0, len(stage_samples), self.batch_size):
                batch_samples = stage_samples[i:i + self.batch_size]
                batch_results_input = stage_results[i:i + self.batch_size]
                
                batch_counter += 1
                stage_batch_idx = i // self.batch_size + 1
                stage_total_batches = (len(stage_samples) + self.batch_size - 1) // self.batch_size
                
                logging.info(f"\nBatch {stage_batch_idx}/{stage_total_batches}")
                logging.info(f"   Sample IDs: {[s.id for s in batch_samples]}")
                
                batch_result = self.analyze_batch(batch_samples, batch_results_input)
                batch_record = {
                    "batch_index": batches_processed,
                    "lifecycle_stage": stage,
                    "sample_ids": [s.id for s in batch_samples],
                    "success": batch_result.get('success', False),
                    "new_submodules": batch_result.get('new_submodules', []),
                    "classifications": batch_result.get('classifications', []),
                    "system_prompt": batch_result.get('system_prompt', ''),
                    "user_prompt": batch_result.get('user_prompt', ''),
                    "raw_response": batch_result.get('raw_response', ''),
                    "response_time": batch_result.get('response_time', 0),
                    "token_usage": batch_result.get('token_usage', {}),
                    "error": batch_result.get('error')
                }
                
                current_ids = set(batch_record['sample_ids'])
                all_batch_results = [
                    r for r in all_batch_results 
                    if not (not r.get('success') and set(r.get('sample_ids', [])) == current_ids)
                ]
                
                all_batch_results.append(batch_record)
                batches_processed += 1
                
                if output_path:
                    self._save_batch_results(output_path, all_batch_results)
                    self._save_new_submodules(output_path)
                
                if batch_result.get('success'):
                    new_subs_count = len(batch_result.get('new_submodules', []))
                    total_subs_count = len(self.new_submodules)
                    logging.info(f"   Batch success (new submodules: {new_subs_count} | total: {total_subs_count})")
                else:
                    logging.error(f"   Batch failed: {batch_result.get('error')}")
                
                time.sleep(REQUEST_DELAY)

        success_count = sum(1 for r in all_batch_results if r.get('success'))
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Stage 3 completed")
        logging.info(f"   Total batches: {len(all_batch_results)} (new: {batch_counter})")
        logging.info(f"   Success: {success_count}/{len(all_batch_results)}")
        logging.info(f"{'='*80}")
        
        return {
            "total_samples": len(samples),
            "batches_processed": len(all_batch_results),
            "success_batches": success_count,
            "batch_results": all_batch_results
        }
    
    def _save_batch_results(self, output_path: str, batch_results: List[Dict]) -> None:
        from pathlib import Path
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "total_batches": len(batch_results),
            "success_batches": sum(1 for r in batch_results if r.get('success')),
            "batch_results": batch_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Batch results saved to: {output_path}")
    
    def _update_stage2_results(self, classifications: List[Dict]) -> None:
        id_to_clf = {c.get('id'): c for c in classifications}
        for result in self.result_saver.results:
            sample_id = str(result.get('id') or result.get('sample_id'))
            if sample_id in id_to_clf:
                clf = id_to_clf[sample_id]
                new_submodule = clf.get('submodule')
                if new_submodule and new_submodule != "None":
                    result['submodule'] = new_submodule
                    result['stage3_reason'] = clf.get('reason', '')
                    logging.debug(f"Updated sample {sample_id}: submodule = {new_submodule}")
        
        self.result_saver.save()
    
    def _save_new_submodules(self, stage3_output_path: Optional[str] = None) -> None:
        if not self.new_submodules:
            return
        if stage3_output_path:
            from pathlib import Path
            submodules_path = Path(stage3_output_path).parent / "new_submodules.json"
        else:
            output_path = self.result_saver.output_path
            submodules_path = output_path.parent / "new_submodules.json"
        
        existing = []
        if submodules_path.exists():
            try:
                with open(submodules_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                pass
        
        existing_keys = {(s.get('lifecycle_stage'), s.get('name')) for s in existing}
        for new_sub in self.new_submodules:
            key = (new_sub.get('lifecycle_stage'), new_sub.get('name'))
            if key not in existing_keys:
                existing.append(new_sub)
                existing_keys.add(key)
        
        with open(submodules_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        
        logging.info(f"New submodules saved to: {submodules_path}")



