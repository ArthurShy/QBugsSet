#!/usr/bin/env python3
"""Stage 2 Analyzer: Quantum relevance analysis and bug classification."""

import json
import time
import logging
import threading
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .data_types import (
    BugSample,
    AnalysisResult,
    REQUEST_DELAY,
)
from .io_utils import ResultSaver, ErrorLogger, NullErrorLogger
from .prompt_templates import PromptTemplate
from .validators import (
    parse_json_response,
    validate_stage2_output,
    validate_submodule,
)

try:
    from api_clients.deepseek_client import MODEL_TIMEOUT, MODEL_MAX_OUTPUT_TOKENS
except ImportError:
    MODEL_TIMEOUT = {"deepseek-reasoner": 120}
    MODEL_MAX_OUTPUT_TOKENS = {"deepseek-reasoner": 64000}


class StageTwoAnalyzer:
    """Stage 2 Analyzer: Quantum relevance and bug classification."""
    
    def __init__(
        self,
        client: Any,
        prompt_template: PromptTemplate,
        result_saver: ResultSaver,
        error_logger: Optional[ErrorLogger] = None,
        test_mode: bool = False
    ):
        self.client = client
        self.prompt_template = prompt_template
        self.result_saver = result_saver
        self.error_logger = error_logger or NullErrorLogger()
        self.test_mode = test_mode
    
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
    
    

    
    def analyze_single_sample(self, sample: BugSample, bug_reason: Optional[str] = None) -> AnalysisResult:
        try:
            stage2_system, stage2_user = self.prompt_template.get_analysis_prompts(sample)
            
            if self.test_mode:
                logging.info("=" * 80)
                logging.info(f"Stage 2: Sample {sample.id}")
                logging.info("=" * 80)
                logging.info(f"System prompt: {len(stage2_system)} chars")
                logging.info(f"User prompt: {len(stage2_user)} chars")
            
            response = self._call_api_with_messages(stage2_system, stage2_user)
            content = response['choices'][0]['message']['content']
            response_time = response.get('_response_time', 0)
            token_usage = response.get('usage', {})
            
            parsed_result = parse_json_response(content)
            
            if not parsed_result:
                logging.error(f"Sample {sample.id}: Failed to parse response")
                return AnalysisResult(
                    sample_id=sample.id,
                    success=False,
                    change_category='Bug Fix',
                    bug_reason=bug_reason,
                    commit_message=sample.commit_message,
                    error="Failed to parse JSON response",
                    raw_response=content,
                    api_response_time=response_time,
                    token_usage=token_usage,
                    parent_file_path=sample.file_path,
                    repository=sample.repository,
                    commit_sha=sample.commit_sha,
                    framework=sample.framework
                )
            
            quantum_specific = parsed_result.get('quantum_specific')
            lifecycle_stage = parsed_result.get('lifecycle_stage')
            submodule = parsed_result.get('submodule')
            
            if not submodule and parsed_result.get('bug_type'):
                submodule = parsed_result.get('bug_type')
            
            _, submodule, _ = validate_submodule(lifecycle_stage, submodule)
            
            quantum_reason = parsed_result.get('quantum_reason')
            lifecycle_reason = parsed_result.get('lifecycle_reason')
            
            bug_type = submodule
            
            validation_record = {
                'sample_id': sample.id,
                'lifecycle_stage': lifecycle_stage,
                'submodule': submodule,
                'quantum_specific': quantum_specific,
                'lifecycle_reason': lifecycle_reason,
                'quantum_reason': quantum_reason
            }
            is_valid, validation_errors = validate_stage2_output(validation_record)
            
            if not is_valid:
                error_msg = "; ".join(validation_errors)
                logging.warning(f"Sample {sample.id} validation failed: {error_msg}")
                return AnalysisResult(
                    sample_id=sample.id,
                    success=False,
                    change_category='Bug Fix',
                    bug_reason=bug_reason,
                    error=f"Validation failed: {error_msg}",
                    raw_response=content,
                    api_response_time=response_time,
                    token_usage=token_usage,
                    parent_file_path=sample.file_path,
                    commit_message=sample.commit_message,
                    repository=sample.repository,
                    commit_sha=sample.commit_sha,
                    framework=sample.framework
                )
            
            if self.test_mode:
                logging.info(f"Stage 2 done:")
                logging.info(f"   quantum_specific = {quantum_specific}")
                logging.info(f"   lifecycle_stage = {lifecycle_stage}")
                logging.info(f"   submodule = {bug_type}")
                logging.info(f"   quantum_reason = {quantum_reason}")
                logging.info(f"   lifecycle_reason = {lifecycle_reason}\n")
            
            return AnalysisResult(
                sample_id=sample.id,
                success=True,
                change_category='Bug Fix',
                bug_reason=bug_reason,
                quantum_specific=quantum_specific,
                lifecycle_stage=lifecycle_stage,
                bug_type=bug_type,
                quantum_reason=quantum_reason,
                lifecycle_reason=lifecycle_reason,
                commit_message=sample.commit_message,
                raw_response=content,
                api_response_time=response_time,
                token_usage=token_usage,
                parent_file_path=sample.file_path,
                repository=sample.repository,
                commit_sha=sample.commit_sha,
                framework=sample.framework
            )
        
        except Exception as e:
            logging.error(f"Stage 2 failed for {sample.id}: {e}")
            return AnalysisResult(
                sample_id=sample.id,
                success=False,
                change_category='Bug Fix',
                bug_reason=bug_reason,
                commit_message=sample.commit_message,
                error=str(e),
                parent_file_path=sample.file_path,
                repository=sample.repository,
                commit_sha=sample.commit_sha,
                framework=sample.framework
            )
            
    def analyze_batch(
        self,
        samples: List[BugSample],
        stage_one_results: List[Dict],
        save_incremental: bool = True
    ) -> List[AnalysisResult]:
        id_to_change_category = {}
        id_to_bug_reason = {}
        for result in stage_one_results:
            if result.get('success'):
                sid = str(result.get('id') or result.get('sample_id'))
                if result.get('change_category'):
                    id_to_change_category[sid] = result['change_category']
                if result.get('bug_reason'):
                    id_to_bug_reason[sid] = result['bug_reason']
                elif result.get('reason'):
                    id_to_bug_reason[sid] = result['reason']
        
        bug_fix_samples = []
        for sample in samples:
            change_category = id_to_change_category.get(str(sample.id))
            if change_category == 'Bug Fix':
                bug_fix_samples.append(sample)
        
        if len(bug_fix_samples) == 0:
            logging.info("No Bug Fix samples, Stage 2 skipped")
            return []
        
        processed_ids = set()
        for r in self.result_saver.results:
            if r.get('success') and r.get('quantum_specific') is not None:
                rid = r.get('id') or r.get('sample_id')
                if rid:
                    processed_ids.add(str(rid))
        
        unprocessed_samples = [s for s in bug_fix_samples if str(s.id) not in processed_ids]
        
        if len(unprocessed_samples) == 0:
            logging.info("All Bug Fix samples already processed")
            return []
        
        results = []
        quantum_related_count = 0
        
        logging.info(f"Stage 2: Processing {len(unprocessed_samples)} samples...")
        
        total_quantum = sum(1 for r in self.result_saver.results if r.get('quantum_specific') and (r.get('submodule') or r.get('bug_type')))
        logging.info(f"Quantum samples so far: {total_quantum}")
        
        batch_size = 50
        for idx, sample in enumerate(tqdm(unprocessed_samples, desc="Stage 2"), 1):
            try:
                bug_reason = id_to_bug_reason.get(sample.id)
                result = self.analyze_single_sample(sample, bug_reason=bug_reason)
                
                if result.success:
                    quantum_specific = result.quantum_specific
                    bug_type = result.bug_type
                    
                    results.append(result)
                    
                    if save_incremental:
                        saved = self.result_saver.add_result_batch(result, batch_size=batch_size)
                        if saved:
                            logging.info(f"Saved {len(self.result_saver.results)} results")
                    
                else:
                    results.append(result)
                    if save_incremental:
                        self.result_saver.add_result_batch(result, batch_size=batch_size)
                    
                    logging.error(f"Sample {sample.id} failed: {result.error}")
                    
                    self.error_logger.log_type_classification_error(
                        sample_id=sample.id,
                        error_type="stage2_analysis_failed",
                        error_message=str(result.error),
                        llm_response=result.raw_response or "",
                        sample_info={
                            "file_path": sample.file_path,
                            "repository": sample.repository,
                            "commit_sha": sample.commit_sha
                        }
                    )
                
                time.sleep(REQUEST_DELAY)
            
            except Exception as e:
                logging.error(f"\n{'='*80}")
                logging.error(f"Stage 2 error: {e}")
                logging.error(f"{'='*80}")
                import traceback
                logging.error(f"Traceback:\n{traceback.format_exc()}")
        
        if save_incremental:
            self.result_saver.flush()
        else:
            for result in results:
                self.result_saver.add_result(result)
            self.result_saver.save()
        
        success_count = sum(1 for r in results if r.success)
        quantum_bug_fix_count = sum(1 for r in results if r.quantum_specific)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Stage 2 done: {success_count}/{len(results)} succeeded")
        logging.info(f"Quantum Bug Fix: {quantum_bug_fix_count}")
        logging.info(f"{'='*80}")
        
        return results
    
    def analyze_batch_parallel(
        self,
        samples: List[BugSample],
        stage_one_results: List[Dict],
        save_incremental: bool = True,
        max_workers: int = 3,
        realtime_save: bool = True
    ) -> List[AnalysisResult]:
        id_to_change_category = {}
        id_to_bug_reason = {}
        for result in stage_one_results:
            if result.get('success'):
                sid = str(result.get('id') or result.get('sample_id'))
                if result.get('change_category'):
                    id_to_change_category[sid] = result['change_category']
                if result.get('bug_reason'):
                    id_to_bug_reason[sid] = result['bug_reason']
                elif result.get('reason'):
                    id_to_bug_reason[sid] = result['reason']
        
        bug_fix_samples_with_index = []
        for idx, sample in enumerate(samples):
            change_category = id_to_change_category.get(str(sample.id))
            if change_category == 'Bug Fix':
                bug_fix_samples_with_index.append((idx, sample))
        
        if len(bug_fix_samples_with_index) == 0:
            logging.info("No Bug Fix samples, Stage 2 skipped")
            return []
        
        processed_ids = set()
        for r in self.result_saver.results:
            if r.get('success') and r.get('quantum_specific') is not None:
                rid = r.get('id') or r.get('sample_id')
                if rid:
                    processed_ids.add(str(rid))
        
        unprocessed_samples_with_index = [
            (idx, sample) for idx, sample in bug_fix_samples_with_index 
            if str(sample.id) not in processed_ids
        ]
        
        if len(unprocessed_samples_with_index) == 0:
            logging.info("All Bug Fix samples already processed")
            return []
        
        total_quantum = sum(1 for r in self.result_saver.results if r.get('quantum_specific') and (r.get('submodule') or r.get('bug_type')))
        
        logging.info(f"Stage 2 parallel: {len(unprocessed_samples_with_index)} samples, {max_workers} workers")
        logging.info(f"Quantum samples so far: {total_quantum}")
        
        results_dict = {}
        results_lock = threading.Lock()
        batch_size = 50
        
        def process_single_sample(idx_and_sample):
            original_idx, sample = idx_and_sample
            
            try:
                bug_reason = id_to_bug_reason.get(sample.id)
                result = self.analyze_single_sample(sample, bug_reason=bug_reason)
                result._sort_index = original_idx
                
                with results_lock:
                    results_dict[original_idx] = result
                    
                    if save_incremental and realtime_save:
                        saved = self.result_saver.add_result_batch(result, batch_size=batch_size)
                        if saved:
                            logging.info(f"Saved {len(self.result_saver.results)} results")
                    
                    if not result.success:
                        logging.error(f"Sample {sample.id} failed: {result.error}")
                        self.error_logger.log_type_classification_error(
                            sample_id=sample.id,
                            error_type="stage2_analysis_failed",
                            error_message=str(result.error),
                            llm_response=result.raw_response or "",
                            sample_info={
                                "file_path": sample.file_path,
                                "repository": sample.repository,
                                "commit_sha": sample.commit_sha
                            }
                        )
                
                return result
                
            except Exception as e:
                logging.error(f"Stage 2 error for {sample.id}: {e}")
                import traceback
                logging.error(f"Traceback:\n{traceback.format_exc()}")
                
                error_result = AnalysisResult(
                    sample_id=sample.id,
                    success=False,
                    change_category='Bug Fix',
                    commit_message=sample.commit_message,
                    bug_reason=id_to_bug_reason.get(sample.id),
                    error=str(e),
                    parent_file_path=sample.file_path,
                    repository=sample.repository,
                    commit_sha=sample.commit_sha,
                    framework=sample.framework,
                    _sort_index=original_idx
                )
                
                with results_lock:
                    results_dict[original_idx] = error_result
                
                return error_result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_sample, idx_sample) 
                for idx_sample in unprocessed_samples_with_index
            ]
            
            for _ in tqdm(
                as_completed(futures), 
                total=len(futures),
                desc="Stage 2 parallel"
            ):
                pass
        
        sorted_indices = sorted(results_dict.keys())
        results = [results_dict[idx] for idx in sorted_indices]
        
        if save_incremental and not realtime_save:
            logging.info(f"Batch saving {len(results)} results...")
            for result in results:
                self.result_saver.add_result(result)
            self.result_saver.save()
        elif save_incremental and realtime_save:
            self.result_saver.flush()
            logging.info(f"Saved {len(results)} results (batch mode)")
        else:
            for result in results:
                self.result_saver.add_result(result)
            self.result_saver.save()
        
        success_count = sum(1 for r in results if r.success)
        quantum_bug_fix_count = sum(1 for r in results if r.quantum_specific)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Stage 2 parallel done: {success_count}/{len(results)} succeeded")
        logging.info(f"Quantum Bug Fix: {quantum_bug_fix_count}")
        logging.info(f"{'='*80}")
        
        return results
