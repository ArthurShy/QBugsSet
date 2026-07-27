
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# from .base_analyzer import BaseAnalyzer # Removed
from .prompt_templates import PromptTemplate
from .validators import parse_json_response, validate_stage4_coverage

class StageFourAnalyzer:
    """Stage 4 Analyzer: Submodule merging and optimization."""
    
    def __init__(self, llm_client, output_dir: str):
        self.client = llm_client
        self.output_dir = Path(output_dir)
        self.new_submodules_path = self.output_dir / "stage3" / "new_submodules.json"
        self.stage3_results_path = self.output_dir / "stage3" / "all_stage3.json"
        
    def _call_api_with_messages(
        self, 
        system_prompt: str, 
        user_prompt: str,
        use_json_mode: bool = True
    ) -> Dict[str, Any]:
        response_format = {"type": "json_object"} if use_json_mode else None
        
        if hasattr(self.client, 'model') and 'reasoner' in str(self.client.model):
            response_format = None
            
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

    def run(self, target_stage: str) -> Dict[str, Any]:
        logging.info(f"Stage 4: Merging submodules [{target_stage}]")
        static_subs = []
        if target_stage in PromptTemplate.LIFECYCLE_DEFINITIONS:
            static_subs = PromptTemplate.LIFECYCLE_DEFINITIONS[target_stage]["submodules"]
        else:
            logging.warning(f"Unknown lifecycle stage: {target_stage}")
            return {"success": False, "error": f"Unknown stage: {target_stage}"}
            
        dynamic_subs = self._load_dynamic_submodules(target_stage)
        
        logging.info(f"Input for merging:")
        logging.info(f"   - Static submodules: {len(static_subs)}")
        logging.info(f"   - Dynamic submodules: {len(dynamic_subs)}")
        
        if not static_subs and not dynamic_subs:
            logging.warning("No submodules to merge")
            return {"success": False, "message": "No submodules to merge"}
        # Load submodule counts for better merging decisions
        submodule_counts = self._load_submodule_counts(target_stage)
        logging.info(f"   - Submodule counts: {submodule_counts}")
        
        system_prompt, user_prompt = PromptTemplate.get_stage4_prompts(
            target_stage, static_subs, dynamic_subs, submodule_counts
        )
        
        logging.info("Requesting LLM for merge analysis...")
        start_time = time.time()
        
        try:
            response = self._call_api_with_messages(system_prompt, user_prompt)
            content = response['choices'][0]['message']['content']
            
            parsed = parse_json_response(content)
            
            if not parsed or 'merged_submodules' not in parsed:
                logging.error("Cannot parse LLM response or invalid format")
                return {"success": False, "error": "Invalid LLM response", "raw_response": content}
                
            merged_subs = parsed['merged_submodules']
            
            input_names = [s[0] for s in static_subs] + [d.get('name', '') for d in dynamic_subs]
            is_covered, coverage_errors = validate_stage4_coverage(input_names, merged_subs)
            
            if not is_covered:
                logging.warning(f"Coverage check warning: {coverage_errors}")
            else:
                logging.info("All input submodules are covered")
            self._save_merged_submodules(target_stage, merged_subs)
            
            elapsed = time.time() - start_time
            logging.info(f"Merge completed ({elapsed:.2f}s)")
            logging.info(f"   Before: {len(static_subs) + len(dynamic_subs)}")
            logging.info(f"   After: {len(merged_subs)}")
            
            return {
                "success": True,
                "target_stage": target_stage,
                "merged_submodules": merged_subs,
                "warnings": coverage_errors,
                "raw_response": content
            }
            
        except Exception as e:
            logging.error(f"Error during analysis: {e}")
            return {"success": False, "error": str(e)}

    def _load_submodule_counts(self, target_stage: str) -> Dict[str, int]:
        """Load sample counts for each submodule from stage2 (static) and stage3 (dynamic)."""
        counts: Dict[str, int] = {}
        
        # 1. Load static submodule counts from stage2
        stage2_path = self.output_dir / "stage2" / "all_stage2.json"
        if stage2_path.exists():
            try:
                with open(stage2_path, 'r', encoding='utf-8') as f:
                    stage2_data = json.load(f)
                
                for result in stage2_data.get('results', []):
                    if result.get('lifecycle_stage') != target_stage:
                        continue
                    submodule = result.get('submodule')
                    if submodule and submodule != 'None':
                        counts[submodule] = counts.get(submodule, 0) + 1
                        
            except Exception as e:
                logging.error(f"Failed to load stage2 submodule counts: {e}")
        
        # 2. Load dynamic submodule counts from stage3
        if self.stage3_results_path.exists():
            try:
                with open(self.stage3_results_path, 'r', encoding='utf-8') as f:
                    stage3_data = json.load(f)
                
                batches = stage3_data.get('batch_results', [])
                for batch in batches:
                    if batch.get('lifecycle_stage') != target_stage:
                        continue
                    if not batch.get('success'):
                        continue
                    for classification in batch.get('classifications', []):
                        submodule = classification.get('submodule')
                        if submodule and submodule != 'None':
                            counts[submodule] = counts.get(submodule, 0) + 1
                            
            except Exception as e:
                logging.error(f"Failed to load stage3 submodule counts: {e}")
            
        return counts

    def _load_dynamic_submodules(self, target_stage: str) -> List[Dict]:
        if not self.new_submodules_path.exists():
            logging.warning(f"Dynamic submodules file not found: {self.new_submodules_path}")
            return []
            
        try:
            with open(self.new_submodules_path, 'r', encoding='utf-8') as f:
                all_subs = json.load(f)
                
            return [s for s in all_subs if s.get('lifecycle_stage') == target_stage]
            
        except Exception as e:
            logging.error(f"Failed to load dynamic submodules: {e}")
            return []

    def _save_merged_submodules(self, stage: str, merged_subs: List[Dict]) -> None:
        output_dir = self.output_dir / "stage4"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"merged_{stage}.json"
        
        data = {
            "lifecycle_stage": stage,
            "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "submodules": merged_subs
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Results saved to: {output_file}")
