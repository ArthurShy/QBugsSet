#!/usr/bin/env python3
"""DeepSeek Analyze validators module."""

import json
import re
import logging
from typing import List, Dict, Optional, Tuple, Set, Any

VALID_LIFECYCLE_STAGES = [
    "Problem Modeling",
    "Data Encoding",
    "Circuit Design",
    "Transpilation",
    "Execution",
    "Post-processing",
    "Infrastructure",
    "None"  # For non-quantum samples
]

LIFECYCLE_HIERARCHY = {
    "Problem Modeling": [
        "Problem Formulation",
        "Hamiltonian Construction",
        "Model Validation",
        "None"
    ],
    "Data Encoding": [
        "Input Validation",
        "Feature Mapping",
        "State Preparation",
        "None"
    ],
    "Circuit Design": [
        "Register Allocation",
        "Parameter Definition",
        "Circuit Construction",
        "Measurement Configuration",
        "None"
    ],
    "Transpilation": [
        "Backend Constraints",
        "Gate Decomposition",
        "Qubit Routing",
        "Circuit Optimization",
        "None"
    ],
    "Execution": [
        "Backend Setup",
        "Job Management",
        "Noise Modeling",
        "Result Retrieval",
        "None"
    ],
    "Post-processing": [
        "Result Parsing",
        "Metrics Computation",
        "Optimizer Control",
        "None"
    ],
    "Infrastructure": [
        "Visualization",
        "File Processing",
        "Logging",
        "Version Migration",
        "None"
    ],
    "None": ["None", ""]
}

VALID_CHANGE_CATEGORIES = [
    "Bug Fix",
    "Feature",
    "Refactor",
    "Style",
    "Test",
    "Build/Chore",
    "Documentation",
    "Other"
]

def parse_json_response(content: str) -> Optional[Dict]:
    """Parse JSON response from LLM with multi-level fallback."""
    if not content or not content.strip():
        logging.warning("Received empty response")
        return None
    
    content = content.strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.debug(f"Direct parse failed: {e}")
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = content[first_brace:last_brace+1]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        fixed_str = _fix_nested_quotes(json_str)
        if fixed_str:
            try:
                return json.loads(fixed_str)
            except json.JSONDecodeError:
                pass
        
        result = _extract_fields_manually(json_str)
        if result:
            return result
    
    if first_brace != -1:
        truncated_json = content[first_brace:]
        open_braces = truncated_json.count('{') - truncated_json.count('}')
        open_brackets = truncated_json.count('[') - truncated_json.count(']')
        
        if open_braces > 0 or open_brackets > 0:
            logging.warning(f"Detected truncated JSON, attempting to fix ({open_braces} '}}', {open_brackets} ']')")
            fixed_json = truncated_json + ']' * open_brackets + '}' * open_braces
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError as e:
                logging.debug(f"Parse still failed after fix: {e}")
    
    result = _extract_fields_manually(content)
    if result:
        return result
    
    logging.error(f"All JSON parse methods failed, preview: {content[:200]}...")
    logging.error(f"End of content: ...{content[-100:]}")
    return None


def _extract_fields_manually(content: str) -> Optional[Dict]:
    """Manually extract key fields when JSON parse fails."""
    result = {}
    
    cat_match = re.search(r'"change_category"\s*:\s*"([^"]+)"', content)
    if cat_match:
        result['change_category'] = cat_match.group(1)
    
    reason_match = re.search(r'"(?:bug_)?reason"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
    if reason_match:
        result['bug_reason'] = reason_match.group(1)
    
    id_match = re.search(r'"id"\s*:\s*"([^"]+)"', content)
    if id_match:
        result['id'] = id_match.group(1)
    
    qs_match = re.search(r'"quantum_specific"\s*:\s*(true|false)', content, re.IGNORECASE)
    if qs_match:
        result['quantum_specific'] = qs_match.group(1).lower() == 'true'
    
    ls_match = re.search(r'"lifecycle_stage"\s*:\s*"([^"]+)"', content)
    if ls_match:
        result['lifecycle_stage'] = ls_match.group(1)
    
    sub_match = re.search(r'"submodule"\s*:\s*"([^"]+)"', content)
    if sub_match:
        result['submodule'] = sub_match.group(1)
    
    if result.get('change_category') or result.get('quantum_specific') is not None:
        logging.debug(f"Manually extracted fields: {list(result.keys())}")
        return result
    
    return None


def _fix_nested_quotes(json_str: str) -> Optional[str]:
    """Fix nested quotes in JSON string."""
    try:
        result = []
        in_string = False
        escape_next = False
        i = 0
        
        while i < len(json_str):
            char = json_str[i]
            
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                result.append(char)
                i += 1
                continue
            
            if char == '"':
                if not in_string:
                    in_string = True
                    result.append(char)
                else:
                    next_non_space = i + 1
                    while next_non_space < len(json_str) and json_str[next_non_space] in ' \t\n\r':
                        next_non_space += 1
                    
                    if next_non_space < len(json_str):
                        next_char = json_str[next_non_space]
                        if next_char in ',}]:':
                            in_string = False
                            result.append(char)
                        else:
                            result.append("'")
                    else:
                        result.append(char)
                        in_string = False
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    except Exception:
        return None

def validate_batch_output_consistency(
    output_ids: Set[str], 
    expected_ids: Set[str]
) -> Tuple[bool, List[str]]:
    """Validate batch output consistency."""
    errors = []
    
    missing_ids = expected_ids - output_ids
    if missing_ids:
        errors.append(f"Missing outputs for: {missing_ids}")
        
    extra_ids = output_ids - expected_ids
    if extra_ids:
        errors.append(f"Unexpected outputs (hallucination): {extra_ids}")
        
    if len(output_ids) != len(expected_ids):
        errors.append(
            f"Count mismatch: expected {len(expected_ids)}, got {len(output_ids)}"
        )
        
    return len(errors) == 0, errors

def validate_classification_submodules(
    classifications: List[Dict],
    id_to_stage: Dict[str, str],
    hierarchy: Dict[str, List[str]]
) -> Tuple[bool, List[str]]:
    """Validate submodule classifications against hierarchy."""
    errors = []
    
    for idx, clf in enumerate(classifications):
        sample_id = clf.get('id')
        submodule = clf.get('submodule')
        
        if not sample_id or not submodule or submodule == "None":
            continue
            
        stage = id_to_stage.get(sample_id)
        if not stage:
            errors.append(f"Sample {sample_id}: lifecycle stage not found")
            continue
            
        valid_submodules = hierarchy.get(stage, [])
        if submodule not in valid_submodules:
            errors.append(
                f"Sample {sample_id}: submodule '{submodule}' not in valid list for stage '{stage}'"
            )
            
    return len(errors) == 0, errors


def validate_no_none_submodules(classifications: List[Dict]) -> Tuple[bool, List[str]]:
    """Validate Stage 3 submodules are not None or empty."""
    none_submodule_ids = []
    
    for clf in classifications:
        sample_id = clf.get('id')
        submodule = clf.get('submodule')
        
        if not sample_id:
            continue
            
        if not submodule or submodule.lower() == 'none' or submodule.strip() == '':
            none_submodule_ids.append(sample_id)
    
    if none_submodule_ids:
        return False, [f"Samples with None submodule: {none_submodule_ids}"]
    
    return True, []

def validate_change_type(category: str) -> Tuple[bool, str, str]:
    """Validate and normalize change type (Stage 1)."""
    if not category or not isinstance(category, str):
        return False, "Other", "Category is None or not a string"
    
    category_clean = category.strip()
    
    if category_clean in VALID_CHANGE_CATEGORIES:
        return True, category_clean, ""
    
    category_lower = category_clean.lower()
    for valid_cat in VALID_CHANGE_CATEGORIES:
        if category_lower == valid_cat.lower():
            return True, valid_cat, ""
    
    return False, "Other", f"Invalid change_category '{category}', mapped to 'Other'"

def validate_lifecycle_stage(stage: str) -> Tuple[bool, str, str]:
    """Validate and normalize lifecycle stage (Stage 2/3)."""
    if not stage:
        return False, "", "Stage is empty"
        
    stage_clean = stage.strip()
    if stage_clean in VALID_LIFECYCLE_STAGES:
        return True, stage_clean, ""
        
    return False, "", f"Invalid lifecycle stage '{stage}'"

def validate_submodule(stage: str, submodule: str) -> Tuple[bool, str, str]:
    """Validate and normalize submodule (Stage 2/3)."""
    if not submodule or submodule == "None" or not submodule.strip():
        submodule_clean = "None"
    else:
        submodule_clean = submodule.strip()

    is_stage_valid, stage_clean, _ = validate_lifecycle_stage(stage)
    if not is_stage_valid and stage != "None":
        return False, submodule_clean, f"Precondition failed: invalid stage '{stage}'"
        
    if not stage or stage not in LIFECYCLE_HIERARCHY:
       if submodule_clean == "None":
            return True, "None", ""
       return False, submodule_clean, f"Stage '{stage}' not defined in hierarchy"

    valid_submodules = LIFECYCLE_HIERARCHY[stage]
    
    if submodule_clean in valid_submodules:
        return True, submodule_clean, ""
        
    cleaned_no_num = re.sub(r'^[\d\.]+\s*', '', submodule_clean).strip()
    if cleaned_no_num in valid_submodules:
        return True, cleaned_no_num, ""
        
    return False, submodule_clean, f"Submodule '{submodule_clean}' not in stage '{stage}'"


def validate_stage1_output(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate Stage 1 output record."""
    errors = []
    
    if 'change_category' not in record:
        errors.append("Missing required field: change_category")
    if 'reason' not in record:
        errors.append("Missing required field: reason")
        
    if errors:
        return False, errors
        
    is_valid, _, error_msg = validate_change_type(record['change_category'])
    if not is_valid:
        errors.append(error_msg)
        
    change_category = record.get('change_category')
    reason = record.get('reason')
    
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason is empty or invalid format")
        
    return len(errors) == 0, errors

def validate_stage2_output(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate Stage 2 output record."""
    errors = []
    sample_id = record.get("sample_id", record.get("id", "unknown"))
    
    required_fields = ["lifecycle_stage", "quantum_specific", "submodule", "lifecycle_reason", "quantum_reason"]
    for field in required_fields:
        if field not in record:
            errors.append(f"[{sample_id}] Missing required field: {field}")
            
    if errors:
        return False, errors
        
    lifecycle_stage = record.get("lifecycle_stage")
    submodule = record.get("submodule")
    quantum_specific = record.get("quantum_specific")
    lifecycle_reason = record.get("lifecycle_reason")
    quantum_reason = record.get("quantum_reason")
    
    is_stage_valid, _, stage_error = validate_lifecycle_stage(lifecycle_stage)
    if not is_stage_valid and lifecycle_stage != "None": 
        errors.append(f"[{sample_id}] {stage_error}")
        
    if is_stage_valid:
        is_sub_valid, _, sub_error = validate_submodule(lifecycle_stage, submodule)
        if not is_sub_valid:
            errors.append(f"[{sample_id}] {sub_error}")

    if not isinstance(lifecycle_reason, str) or not lifecycle_reason.strip():
        errors.append(f"[{sample_id}] lifecycle_reason is empty or invalid")
    if not isinstance(quantum_reason, str) or not quantum_reason.strip():
        errors.append(f"[{sample_id}] quantum_reason is empty or invalid")

    pass
            
    return len(errors) == 0, errors

def validate_stage3_output(
    parsed_result: Dict, 
    expected_ids: Set[str],
    hierarchy: Optional[Dict[str, List[str]]] = None
) -> Tuple[bool, List[str]]:
    """Validate Stage 3 output."""
    errors = []
    target_hierarchy = hierarchy if hierarchy is not None else LIFECYCLE_HIERARCHY
    
    if 'classifications' not in parsed_result:
        errors.append("Missing 'classifications' field")
        return False, errors
    
    classifications = parsed_result.get('classifications', [])
    if not isinstance(classifications, list):
        errors.append("'classifications' must be an array")
        return False, errors
        
    classified_ids = set()
    for idx, clf in enumerate(classifications):
        if not isinstance(clf, dict):
            errors.append(f"classifications[{idx}] is not an object")
            continue
        
        sample_id = clf.get('id')
        if not sample_id:
            errors.append(f"classifications[{idx}] missing 'id' field")
        else:
            classified_ids.add(sample_id)
            
        if 'submodule' not in clf:
            errors.append(f"classifications[{idx}] (id={sample_id}) missing 'submodule' field")

    is_consistent, consistency_errors = validate_batch_output_consistency(classified_ids, expected_ids)
    if not is_consistent:
        errors.extend(consistency_errors)
        
    new_submodules = parsed_result.get('new_submodules', [])
    if new_submodules and isinstance(new_submodules, list):
        for idx, sub in enumerate(new_submodules):
            if not isinstance(sub, dict):
                errors.append(f"new_submodules[{idx}] is not an object")
                continue
            if 'lifecycle_stage' not in sub:
                errors.append(f"new_submodules[{idx}] missing 'lifecycle_stage'")
            if 'name' not in sub:
                errors.append(f"new_submodules[{idx}] missing 'name'")
    
    return len(errors) == 0, errors

def validate_stage4_coverage(
    input_submodules: List[str],
    merged_output: List[Dict]
) -> Tuple[bool, List[str]]:
    """Validate Stage 4 merge result covers all input submodules."""
    errors = []
    
    covered_sources = set()
    for merged_item in merged_output:
        sources = merged_item.get('source_submodules', [])
        if isinstance(sources, list):
            covered_sources.update(sources)
            
    input_set = set(input_submodules)
    missing = input_set - covered_sources
    
    if missing:
        errors.append(f"Submodules not covered in merge result: {list(missing)}")
    
    return len(errors) == 0, errors
