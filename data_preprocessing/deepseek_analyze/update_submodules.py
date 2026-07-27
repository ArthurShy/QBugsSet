#!/usr/bin/env python3
"""Build final positive samples from Stage 2/3/4 analysis outputs."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Keep positive samples consistent with downstream dataset construction.
POSITIVE_MAX_CODE_LENGTH = 250000

def load_merge_maps(stage4_dir: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[Dict]]]:
    """Load Stage 4 merge rules and final lifecycle definitions."""
    mapping = {}
    lifecycle_defs = {}
    
    stage4_files = list(stage4_dir.glob("merged_*.json"))
    
    logging.info(f"📂 Found {len(stage4_files)} merge-rule files")
    
    for file_path in stage4_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            lifecycle_stage = data.get('lifecycle_stage', 'Unknown')
            merged_subs = data.get('submodules', [])
            
            cleaned_subs = []
            for sub in merged_subs:
                cleaned_sub = {k: v for k, v in sub.items() if k != 'source_submodules'}
                cleaned_subs.append(cleaned_sub)
            lifecycle_defs[lifecycle_stage] = cleaned_subs
            
            if lifecycle_stage not in mapping:
                mapping[lifecycle_stage] = {}
            
            count = 0
            for item in merged_subs:
                new_name = item.get('name')
                source_subs = item.get('source_submodules', [])
                
                if not new_name:
                    continue
                    
                for old_name in source_subs:
                    mapping[lifecycle_stage][old_name] = new_name
                    count += 1
                        
            logging.info(f"   - [{lifecycle_stage}]: loaded {count} merge rules, {len(merged_subs)} final submodules")
            
        except Exception as e:
            logging.error(f"❌ Failed to read {file_path.name}: {e}")
            
    return mapping, lifecycle_defs

def load_framework_mapping(merged_path: Path) -> Dict[str, str]:
    """Load ``sample_id -> framework`` from the merged sample file."""
    if not merged_path.exists():
        logging.warning(f"⚠️ Merged input file does not exist: {merged_path}")
        return {}
    
    framework_map = {}
    try:
        with open(merged_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for sample in data.get('samples', []):
            sample_id = str(sample.get('id') or sample.get('sample_id', ''))
            framework = sample.get('framework', '')
            if sample_id and framework:
                framework_map[sample_id] = framework
        
        logging.info(f"📦 Loaded {len(framework_map)} framework mappings")
    except Exception as e:
        logging.error(f"❌ Failed to read merged input file: {e}")
    
    return framework_map


def load_code_mapping(merged_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load code fields keyed by sample id from the merged sample file."""
    if not merged_path.exists():
        logging.warning(f"⚠️ Merged input file does not exist: {merged_path}")
        return {}
    
    code_map = {}
    try:
        with open(merged_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for sample in data.get('samples', []):
            sample_id = str(sample.get('id') or sample.get('sample_id', ''))
            if sample_id:
                code_map[sample_id] = {
                    'buggy_file': sample.get('buggy_file', ''),
                    'buggy_code': sample.get('buggy_code', ''),
                    'parent_file_path': sample.get('parent_file_path', ''),
                    'function_name': sample.get('function_name', ''),
                }
        
        logging.info(f"📝 Loaded {len(code_map)} code mappings")
    except Exception as e:
        logging.error(f"❌ Failed to read code mappings: {e}")
    
    return code_map


def load_commit_parents(raw_data_dir: Path) -> Dict[Tuple[str, str], str]:
    """Load ``(repository, commit_sha) -> parent_sha`` from raw commit data."""
    frameworks = ['qiskit', 'cirq', 'pennylane']
    parent_map = {}
    
    for framework in frameworks:
        commits_file = raw_data_dir / framework / 'commits.json'
        if not commits_file.exists():
            logging.warning(f"⚠️ Missing commits.json: {commits_file}")
            continue
        
        try:
            with open(commits_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            commits = data.get('commits', {})
            for repo, commit_list in commits.items():
                for info in commit_list:
                    c_sha = info.get('commit_sha')
                    p_sha = info.get('parent_sha')
                    if c_sha and p_sha:
                        parent_map[(repo, c_sha)] = p_sha
            
            logging.info(f"🔗 [{framework}] loaded {sum(len(v) for v in commits.values())} parent_sha mappings")
        except Exception as e:
            logging.error(f"❌ Failed to read {framework}/commits.json: {e}")
    
    logging.info(f"🔗 Loaded {len(parent_map)} parent_sha mappings in total")
    return parent_map


def load_stage3_data(stage3_path: Path) -> Dict[str, str]:
    """Load updated Stage 3 submodule labels."""
    if not stage3_path.exists():
        logging.warning(f"⚠️ Stage 3 output file does not exist: {stage3_path}")
        return {}
        
    stage3_map = {}
    total_loaded = 0
    success_batches = 0
    
    try:
        with open(stage3_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        batches = data.get('batch_results', [])
        for batch in batches:
            if not batch.get('success', False):
                continue
            success_batches += 1
            
            classifications = batch.get('classifications', [])
            for item in classifications:
                sample_id = str(item.get('id') or item.get('sample_id', ''))
                submodule = item.get('submodule')
                
                if sample_id and submodule and submodule != 'None':
                    stage3_map[sample_id] = submodule
                    total_loaded += 1
                    
        logging.info(f"📊 Loaded {total_loaded} Stage 3 submodule labels from {success_batches} successful batches")
        
    except Exception as e:
        logging.error(f"❌ Failed to read Stage 3 file: {e}")
        
    return stage3_map

def update_dataset(
    input_path: Path, 
    output_path: Path, 
    mapping: Dict[str, Dict[str, str]], 
    stage3_map: Dict[str, str],
    framework_map: Dict[str, str],
    code_map: Optional[Dict[str, Dict[str, Any]]] = None,
    parent_map: Optional[Dict[Tuple[str, str], str]] = None
):
    """Apply Stage 3 and Stage 4 normalization to the Stage 2 dataset."""
    if not input_path.exists():
        logging.error(f"❌ Input file does not exist: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"📖 Reading source data: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            results = data
            is_dict_wrapper = False
        elif isinstance(data, dict):
            results = data.get('results', [])
            is_dict_wrapper = True
        else:
            logging.error("❌ Unknown input data format")
            return
            
        logging.info(f"📊 Updating {len(results)} samples")
        
        stage3_update_count = 0
        final_mapping_count = 0
        missing_submodules = set()
        
        for record in results:
            sample_id = str(record.get('id') or record.get('sample_id', ''))
            
            if sample_id and sample_id in stage3_map:
                new_sub = stage3_map[sample_id]
                if new_sub and new_sub != "None":
                    record['submodule'] = new_sub
                    stage3_update_count += 1
            
            current_sub = record.get('submodule')
            current_stage = record.get('lifecycle_stage', '')
            
            if current_sub and current_sub != "None":
                stage_mapping = mapping.get(current_stage, {})
                if current_sub in stage_mapping:
                    record['submodule'] = stage_mapping[current_sub]
                    final_mapping_count += 1
                elif current_stage in mapping:
                    missing_submodules.add(f"{current_stage}/{current_sub}")
            
            if sample_id in framework_map:
                record['framework'] = framework_map[sample_id]
            
            if code_map and sample_id in code_map:
                code_info = code_map[sample_id]
                record['buggy_file'] = code_info.get('buggy_file', '')
                record['buggy_code'] = code_info.get('buggy_code', '')
                record['parent_file_path'] = code_info.get('parent_file_path', '')
                record['function_name'] = code_info.get('function_name', '')
            
            if parent_map:
                repo = record.get('repository', '')
                commit_sha = record.get('commit_sha', '')
                if repo and commit_sha:
                    parent_sha = parent_map.get((repo, commit_sha), '')
                    if parent_sha:
                        record['parent_sha'] = parent_sha

        if missing_submodules:
            error_msg = f"❌ {len(missing_submodules)} submodules were not found in the merge map: {list(missing_submodules)[:10]}..."
            logging.error(error_msg)
            raise ValueError(error_msg)

        before_filter = len(results)
        filtered_results = []
        filtered_count = 0
        for record in results:
            buggy_file = record.get("buggy_file", "")
            if buggy_file and len(buggy_file) > POSITIVE_MAX_CODE_LENGTH:
                filtered_count += 1
                continue
            filtered_results.append(record)
        results = filtered_results
                    
        new_metadata = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_type": "bugfix_positive_samples",
            "total_samples": len(results),
        }
            
        logging.info(f"✅ Stage 3 updates applied: {stage3_update_count}")
        logging.info(f"✅ Stage 4 mappings applied: {final_mapping_count}")
        logging.info(f"✅ Code-length filter: {before_filter} -> {len(results)} (removed {filtered_count})")
        
        FIELD_ORDER = [
            "id", "framework", "repository", "commit_sha", "parent_sha",
            "parent_file_path", "function_name", "buggy_file", "buggy_code",
            "success", "change_category", "lifecycle_stage", "submodule", "quantum_specific",
            "quantum_reason", "lifecycle_reason", "bug_reason",
            "commit_message",
        ]
        
        FIELDS_TO_REMOVE = ["file_path", "bug_type", "is_quantum_related", "reason"]
        
        def reorder_record(record: Dict) -> Dict:
            """Reorder fields and drop deprecated aliases."""
            ordered = {}
            for key in FIELD_ORDER:
                if key in record:
                    ordered[key] = record[key]
            for key, value in record.items():
                if key not in ordered and key not in FIELDS_TO_REMOVE:
                    ordered[key] = value
            return ordered
        
        ordered_results = [reorder_record(r) for r in results]
        
        output_data = {
            "metadata": new_metadata,
            "results": ordered_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        logging.info(f"💾 Saved results to: {output_path}")
        
    except Exception as e:
        logging.error(f"❌ Dataset update failed: {e}")

def save_lifecycle_definitions(output_path: Path, lifecycle_defs: Dict[str, List[Dict]]):
    """Save final lifecycle definitions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sorted_defs = {}
    ordered_stages = [
        "Problem Mapping and Model Construction",
        "Data Encoding and State Preparation",
        "Algorithm Logic Evolution",
        "Transpilation",
        "Execution and Simulation",
        "Post-processing and Result Analysis",
        "General Engineering Infrastructure",
    ]
    
    for stage in ordered_stages:
        if stage in lifecycle_defs:
            sorted_defs[stage] = lifecycle_defs[stage]
            
    for stage, content in lifecycle_defs.items():
        if stage not in sorted_defs:
            sorted_defs[stage] = content
            
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_defs, f, indent=2, ensure_ascii=False)
        logging.info(f"💾 Saved final lifecycle definitions to: {output_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save lifecycle definitions: {e}")

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data/04_analyzed"
    extracted_dir = project_root / "data/03_extracted"
    raw_data_dir = project_root / "data/01_raw"
    
    stage4_dir = data_dir / "stage4"
    stage2_input = data_dir / "stage2/all_stage2.json"
    stage3_input = data_dir / "stage3/all_stage3.json"
    merged_input = extracted_dir / "method_level_single_merged.json"
    
    final_dir = data_dir / "final"
    output_path = final_dir / "bugfix_positive_samples.json"
    lifecycle_output = final_dir / "final_lifecycle.json"
    
    mapping, lifecycle_defs = load_merge_maps(stage4_dir)
    if not mapping and not lifecycle_defs:
        logging.warning("⚠️ No merge rules were loaded; exiting")
        return
        
    total_rules = sum(len(m) for m in mapping.values())
    logging.info(f"🔗 Merge map contains {len(mapping)} stage(s) and {total_rules} rule(s)")
    
    stage3_map = load_stage3_data(stage3_input)
    
    framework_map = load_framework_mapping(merged_input)
    
    code_map = load_code_mapping(merged_input)
    
    parent_map = load_commit_parents(raw_data_dir)
    
    update_dataset(stage2_input, output_path, mapping, stage3_map, framework_map, code_map, parent_map)
    
    save_lifecycle_definitions(lifecycle_output, lifecycle_defs)

if __name__ == "__main__":
    main()
