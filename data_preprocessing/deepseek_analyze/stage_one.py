#!/usr/bin/env python3
"""Stage 1 Analyzer: Code change classification."""

import json
import time
import logging
import threading

from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .data_types import BugSample, AnalysisResult, REQUEST_DELAY
from .io_utils import ResultSaver, ErrorLogger
from .prompt_templates import PromptTemplate
from .validators import parse_json_response, validate_change_type


class StageOneAnalyzer:
    """Stage 1 Analyzer: Code change classification."""
    
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
        self.error_logger = error_logger
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
    
    
    def analyze_single_sample(self, sample: BugSample) -> AnalysisResult:
        try:
            stage1_system, stage1_user = self.prompt_template.get_change_category_prompts(sample)
            
            if self.test_mode:
                logging.info("=" * 80)
                logging.info(f"Stage 1: Sample {sample.id}")
                logging.info("=" * 80)
                logging.info(f"System prompt: {len(stage1_system)} chars")
                logging.info(f"User prompt: {len(stage1_user)} chars")
            
            response = self._call_api_with_messages(stage1_system, stage1_user)
            content = response['choices'][0]['message']['content']
            response_time = response.get('_response_time', 0)
            token_usage = response.get('usage', {})
            
            parsed_result = parse_json_response(content)
            
            if not parsed_result or 'change_category' not in parsed_result:
                logging.error(f"Sample {sample.id}: Failed to get change_category")
                logging.error(f"Raw LLM response:\n{content}")
                return AnalysisResult(
                    sample_id=sample.id,
                    success=False,
                    commit_message=sample.commit_message,
                    error="Failed to parse change_category",
                    api_response_time=response_time,
                    token_usage=token_usage,
                    parent_file_path=sample.file_path,
                    raw_response=content,
                    framework=sample.framework
                )
            
            raw_change_category = parsed_result.get('change_category')
            bug_reason = parsed_result.get('reason', '')
            
            is_valid, change_category, _ = validate_change_type(raw_change_category)
            if not is_valid:
                logging.warning(f"Sample {sample.id}: Invalid change_category '{raw_change_category}' normalized to '{change_category}'")
            
            if self.test_mode:
                logging.info(f"Done: change_category = {change_category}")
                if raw_change_category != change_category:
                    logging.info(f"   (raw: {raw_change_category})")
                if bug_reason:
                    logging.info(f"   reason: {bug_reason}")
            
            return AnalysisResult(
                sample_id=sample.id,
                success=True,
                change_category=change_category,
                bug_reason=bug_reason,
                
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
            logging.error(f"Stage 1 failed for {sample.id}: {e}")
            return AnalysisResult(
                sample_id=sample.id,
                success=False,
                commit_message=sample.commit_message,
                error=f"Stage 1 API call failed: {e}",
                parent_file_path=sample.file_path,
                repository=sample.repository,
                commit_sha=sample.commit_sha,
                framework=sample.framework
            )
    
    def analyze_batch(
        self,
        samples: List[BugSample],
        save_incremental: bool = True
    ) -> List[AnalysisResult]:
        results = []
        processed_ids = self.result_saver.get_processed_sample_ids()
        if processed_ids:
            logging.info(f"Found {len(processed_ids)} processed samples, skipping...")
        
        unprocessed_samples = [s for s in samples if str(s.id) not in processed_ids]
        
        if len(unprocessed_samples) == 0:
            logging.info("All samples processed, nothing to do")
            return []
        
        logging.info(f"Stage 1: Classifying {len(unprocessed_samples)} samples...")
        
        batch_size = 50
        for idx, sample in enumerate(tqdm(unprocessed_samples, desc="Stage 1"), 1):
            result = self.analyze_single_sample(sample)
            results.append(result)
            
            if save_incremental:
                saved = self.result_saver.add_result_batch(result, batch_size=batch_size)
                if saved:
                    logging.info(f"Saved {len(self.result_saver.results)} results")
            
            time.sleep(REQUEST_DELAY)
            
            if idx % 10 == 0:
                success_count = sum(1 for r in results if r.success)
                logging.info(f"Stage 1: {idx}/{len(unprocessed_samples)} done, {success_count} succeeded")
        
        if save_incremental:
            self.result_saver.flush()
        else:
            for result in results:
                self.result_saver.add_result(result)
            self.result_saver.save()
        
        success_count = sum(1 for r in results if r.success)
        change_category_dist = {}
        for r in results:
            if r.success and r.change_category:
                change_category_dist[r.change_category] = change_category_dist.get(r.change_category, 0) + 1
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Stage 1 done: {success_count}/{len(results)} succeeded")
        logging.info(f"Distribution: {change_category_dist}")
        logging.info(f"{'='*80}")
        
        return results
    
    def analyze_batch_parallel(
        self,
        samples: List[BugSample],
        save_incremental: bool = True,
        max_workers: int = 3,
        realtime_save: bool = True
    ) -> List[AnalysisResult]:
        processed_ids = self.result_saver.get_processed_sample_ids()
        
        unprocessed_samples_with_index = [
            (idx, sample) for idx, sample in enumerate(samples) 
            if str(sample.id) not in processed_ids
        ]
        
        if len(processed_ids) > 0:
            logging.info(f"Found {len(processed_ids)} processed samples, skipping...")
        
        if len(unprocessed_samples_with_index) == 0:
            logging.info("All samples processed, nothing to do")
            return []
        
        logging.info(f"Stage 1 parallel: {len(unprocessed_samples_with_index)} samples, {max_workers} workers")
        
        results_dict = {}
        results_lock = threading.Lock()
        
        batch_size = 50
        def process_single_sample(idx_and_sample):
            original_idx, sample = idx_and_sample
            result = self.analyze_single_sample(sample)
            result._sort_index = original_idx
            
            with results_lock:
                results_dict[original_idx] = result
                if save_incremental and realtime_save:
                    saved = self.result_saver.add_result_batch(result, batch_size=batch_size)
                    if saved:
                        logging.info(f"Saved {len(self.result_saver.results)} results")
            
            return result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_sample, idx_sample) 
                for idx_sample in unprocessed_samples_with_index
            ]
            
            for _ in tqdm(
                as_completed(futures), 
                total=len(futures),
                desc="Stage 1 parallel"
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
        
        success_count = sum(1 for r in results if r.success)
        change_category_dist = {}
        for r in results:
            if r.success and r.change_category:
                change_category_dist[r.change_category] = change_category_dist.get(r.change_category, 0) + 1
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Stage 1 parallel done: {success_count}/{len(results)} succeeded")
        logging.info(f"Distribution: {change_category_dist}")
        logging.info(f"{'='*80}")
        
        return results
