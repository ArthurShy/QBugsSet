#!/usr/bin/env python3
"""
Utility functions for dataset construction.

Provides common helpers used when building the quantum bug-detection dataset, including:
- Git file checks and path handling
- Loading Stage2 results
- Loading code mappings
- Building positive samples
- Computing statistical distributions
- Saving datasets
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime


# ============================================================================
# Git file checks and path handling.
# ============================================================================

def file_unchanged_since_commit(
    repo_root_dir: Path,
    repository: str,
    commit_sha: str,
    file_path: str,
    default_if_no_repo: bool = False,
    head_sha: Optional[str] = None,
) -> bool:
    """
    Check whether the file has remained unchanged from commit_sha to HEAD or a specified head_sha.
    
    Args:
        repo_root_dir: Repository root directory, for example data/02_source.
        repository: Repository name in owner/repo format.
        commit_sha: commit SHA
        file_path: File path relative to the repository root.
        default_if_no_repo: Default return value if the repo is missing. Defaults to False as a conservative fallback.
        head_sha: Optional HEAD SHA. If omitted, the current HEAD is used.
    
    Returns:
        True if the file was unchanged after the commit, otherwise False if it was modified or the check failed.
    """
    # Use the shared get_repo_dir helper to locate the repository.
    repo_dir = get_repo_dir(repo_root_dir, repository)
    
    if not repo_dir:
        logging.debug(
            "Repo not found for %s, skipping file modification check for %s",
            repository, file_path
        )
        return default_if_no_repo
    
    try:
        # First verify that commit_sha exists.
        subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--verify", commit_sha],
            capture_output=True,
            check=True,
        )
        
        # Determine the HEAD to inspect, preferring head_sha when provided.
        target_head = head_sha if head_sha else "HEAD"
        
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_dir),
                "log",
                "--format=%H",
                f"{commit_sha}..{target_head}",
                "--",
                file_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # No output means there were no later modifications.
        return len(result.stdout.strip()) == 0
    except subprocess.CalledProcessError:
        # If the command fails, for example due to a missing commit or file, conservatively return False.
        logging.debug(
            "Failed to check file modification for %s@%s:%s",
            repository, commit_sha, file_path
        )
        return False


# Supported framework directories.
FRAMEWORKS = ["qiskit", "cirq", "pennylane"]

def get_repo_dir(repo_root_dir: Path, repository: str) -> Optional[Path]:
    """
    Get the repository directory path.
    Directory structure: data/02_source/{framework}/{owner}/{repo}/project/
    
    Args:
        repo_root_dir: Repository root directory, for example data/02_source.
        repository: Repository name in owner/repo format.
    
    Returns:
        The repository directory path, or None if it does not exist.
    """
    try:
        owner, name = repository.split("/", 1)
    except ValueError:
        return None
    
    # Directory structure: data/02_source/{framework}/{owner}/{repo}/project/
    for framework in FRAMEWORKS:
        repo_dir = repo_root_dir / framework / owner / name / "project"
        if repo_dir.is_dir() and (repo_dir / ".git").is_dir():
            return repo_dir
    
    return None


def get_latest_commit_sha(repo_dir: Path) -> Optional[str]:
    """
    Get the latest commit for the repository, preferring the remote default-branch tip.
    Priority: origin/HEAD -> origin/main -> origin/master -> current HEAD.
    
    Args:
        repo_dir: Repository directory path.
    
    Returns:
        Latest commit SHA, or None if resolution fails.
    """
    candidate_refs = ["origin/HEAD", "origin/main", "origin/master", "HEAD"]
    for ref in candidate_refs:
        try:
            res = subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", ref],
                capture_output=True, text=True, check=True
            )
            return res.stdout.strip()
        except subprocess.CalledProcessError:
            continue
    return None


def get_all_modified_files_in_range(
    repo_dir: Path, 
    base_sha: str, 
    head_sha: str
) -> Optional[Set[str]]:
    """
    Get the set of files modified by all commits in base_sha..head_sha, excluding base and including head.
    Uses git log to capture every touched file in history, even if the final file state is unchanged.
    
    Args:
        repo_dir: Repository directory path.
        base_sha: Starting commit SHA, exclusive.
        head_sha: Ending commit SHA, inclusive.
    
    Returns:
        A set of file paths, or None if the command fails to avoid false negatives.
    """
    try:
        res = subprocess.run(
            [
                "git",
                "-C",
                str(repo_dir),
                "log",
                "--name-only",
                "--no-renames",
                "--format=",
                f"{base_sha}..{head_sha}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return {line.strip() for line in res.stdout.splitlines() if line.strip()}
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to get modified files in range {base_sha}..{head_sha}: {e}")
        return None


def file_unchanged_in_range(
    repo_root_dir: Path,
    repository: str,
    base_sha: str,
    file_path: str,
    head_sha: Optional[str] = None,
) -> Optional[bool]:
    """
    Check whether a file was unchanged in the base_sha..head_sha range.
    
    Args:
        repo_root_dir: Repository root directory.
        repository: Repository name.
        base_sha: Starting commit SHA.
        file_path: Relative file path.
        head_sha: Ending commit SHA, or the latest commit if omitted.
    
    Returns:
        True if the file was unchanged, False if it was modified, or None if the check failed.
    """
    repo_dir = get_repo_dir(repo_root_dir, repository)
    if not repo_dir:
        return None
    
    # Get the latest commit.
    target_head = head_sha if head_sha else get_latest_commit_sha(repo_dir)
    if not target_head:
        return None
    
    # Get the list of modified files.
    modified_set = get_all_modified_files_in_range(repo_dir, base_sha, target_head)
    if modified_set is None:
        return None
    
    return file_path not in modified_set


def normalize_file_path(
    file_path: str,
    repository: Optional[str] = None,
    commit_sha: Optional[str] = None,
) -> str:
    """
    Normalize a stored path into a path relative to the repository root.
    
    In the dataset, file_path may be:
    - data/02_source/{repo}/{commit_sha}/{parent|commit}/<rel>
    - a relative path that is already normalized
    
    This extracts the relative path after parent/, commit/, or project/.
    
    Args:
        file_path: Original file path.
        repository: Optional repository name used to help parse the path.
        commit_sha: Optional commit SHA used to help parse the path.
    
    Returns:
        Normalized relative path.
    """
    if not file_path:
        return file_path

    path_str = file_path.replace("\\", "/")
    
    # Prefer explicit markers first.
    markers = ["/parent/", "/commit/", "/project/"]
    for m in markers:
        if m in path_str:
            return path_str.split(m, 1)[1]

    # If commit_sha is present, try extracting the path segment after it.
    if commit_sha and commit_sha in path_str:
        after_commit = path_str.split(commit_sha, 1)[-1].lstrip("/").split("/", 1)
        if len(after_commit) == 2:
            return after_commit[1]

    # If repository is present, extract the path segment after it.
    if repository and repository in path_str:
        after_repo = path_str.split(repository, 1)[-1].lstrip("/").split("/", 1)
        if len(after_repo) == 2:
            return after_repo[1]

    return path_str


# ============================================================================
# Stage1 data loading.
# ============================================================================

def load_stage1_bugfix_ids(stage1_file: Path) -> Set[str]:
    """
    Load the set of sample IDs with change_category="Bug Fix" from all_stage1.json.
    
    Args:
        stage1_file: Path to all_stage1.json.
    
    Returns:
        Set of sample IDs.
    """
    with stage1_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    ids: Set[str] = set()
    for r in results:
        if r.get("change_category") != "Bug Fix":
            continue
        sid = r.get("sample_id")
        if sid:
            ids.add(sid)
    logging.info("Loaded %d Bug Fix sample IDs from stage1", len(ids))
    return ids


def load_stage1_results_dict(stage1_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load results from all_stage1.json and build a sample_id -> {change_category, reason} mapping.
    
    Args:
        stage1_file: Path to all_stage1.json.
    
    Returns:
        Mapping from sample_id to Stage1 result data.
    """
    with stage1_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    mapping: Dict[str, Dict[str, Any]] = {}
    for r in data.get("results", []):
        sid = r.get("sample_id")
        if sid:
            mapping[sid] = {
                "change_category": r.get("change_category", ""),
                "reason": r.get("reason", ""),
            }
    
    logging.info("Loaded stage1 results for %d samples", len(mapping))
    return mapping


# Stage2 data loading.
# ============================================================================

def load_stage2_ids(stage2_file: Path) -> Set[str]:
    """
    Load the set of sample IDs with quantum_specific=true from all_stage2.json.
    
    Args:
        stage2_file: Path to all_stage2.json.
    
    Returns:
        Set of sample IDs.
    """
    with stage2_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    ids: Set[str] = set()
    for r in results:
        if not r.get("quantum_specific", r.get("is_quantum_related")):
            continue
        sid = r.get("id") or r.get("sample_id")
        if sid:
            ids.add(sid)
    return ids


def load_stage2_results_list(stage2_file: Path) -> List[Dict]:
    """
    Load all_stage2.json, filter quantum-related samples, and return the full result list.
    
    Args:
        stage2_file: Path to all_stage2.json.
    
    Returns:
        List of results for quantum-related samples.
    """
    logging.info("\n" + "="*80)
    logging.info("Loading Stage2 analysis results...")
    logging.info("="*80)
    
    with open(stage2_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    total_samples = metadata.get('total_samples', 0)
    quantum_bug_fixes = metadata.get('quantum_bug_fixes', 0)
    
    logging.info(f"   Total samples: {total_samples}")
    logging.info(f"   Quantum-related bug fixes: {quantum_bug_fixes}")
    
    # Filter samples with quantum_specific: true.
    results = data.get('results', [])
    quantum_related_samples = [
        r for r in results 
        if r.get('quantum_specific', r.get('is_quantum_related')) is True
    ]
    
    logging.info(f"\nSelected {len(quantum_related_samples)} quantum-related samples")
    
    return quantum_related_samples


def load_stage2_results_dict(stage2_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load results from a Stage2-style result file or a positive-sample result file
    and build a sample_id -> analysis result mapping.
    Supports both id and sample_id field names.
    
    Args:
        stage2_file: Path to all_stage2.json or bugfix_positive_samples.json.
    
    Returns:
        Mapping from sample_id to analysis results.
    """
    with stage2_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    mapping: Dict[str, Dict[str, Any]] = {}
    for r in data.get("results", []):
        # Support both id and sample_id field names.
        sid = str(r.get("sample_id") or r.get("id", ""))
        if sid:
            mapping[sid] = {
                "quantum_specific": r.get("quantum_specific", r.get("is_quantum_related", False)),
                "submodule": r.get("submodule", r.get("bug_type", "")),
                "lifecycle_stage": r.get("lifecycle_stage", ""),
                "quantum_reason": r.get("quantum_reason", ""),
                "lifecycle_reason": r.get("lifecycle_reason", ""),
                "bug_reason": r.get("bug_reason", r.get("reason", "")),
                "change_category": r.get("change_category", ""),
            }
    
    logging.info("Loaded stage2 results for %d samples", len(mapping))
    return mapping


def load_code_mapping(
    metadata_file: Path,
    allowed_ids: Optional[Set[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load the code mapping from method_level_single.json, including buggy_file and fixed_file.
    
    Args:
        metadata_file: Path to method_level_single.json.
        allowed_ids: Optional set of sample IDs. If provided, only mappings for those IDs are loaded.
    
    Returns:
        Mapping from sample_id to {buggy_file, fixed_file, ...}.
    """
    logging.info("\n" + "="*80)
    logging.info("Building code mapping (buggy/fixed)...")
    logging.info("="*80)
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    total_metadata_samples = metadata.get('total_samples', 0)
    logging.info(f"   Total metadata samples: {total_metadata_samples}")
    
    mapping: Dict[str, Dict[str, Any]] = {}
    samples = data.get('samples', [])
    
    for sample in samples:
        sample_id = sample.get('id')
        
        # If allowed_ids is provided, only process permitted IDs.
        if allowed_ids is not None and sample_id not in allowed_ids:
            continue
        
        buggy_file = sample.get('buggy_file')
        fixed_file = sample.get('fixed_file')
        file_path = sample.get('parent_file_path') or sample.get('file_path') or sample.get('buggy_file_path', '')
        repository = sample.get('repository')
        commit_sha = sample.get('commit_sha')
        commit_url = sample.get('commit_url')
        function_name = sample.get('function_name', '')
        
        if sample_id and buggy_file:
            mapping[sample_id] = {
                'buggy_file': buggy_file,
                'fixed_file': fixed_file or '',
                'parent_file_path': file_path or '',
                'repository': repository or '',
                'commit_sha': commit_sha or '',
                'function_name': function_name
            }
    
    logging.info(f"Successfully built mappings for {len(mapping)} samples")
    
    return mapping


def build_positive_sample(
    sample_id: str,
    code_data: Dict[str, Any],
    analysis_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a positive sample from the buggy version with label=1.
    
    Args:
        sample_id: Sample ID.
        code_data: Code data from code_mapping.
        analysis_data: Combined analysis data including change_category, bug_reason, and Stage2 fields.
    
    Returns:
        Positive-sample dictionary.
    """
    repository = code_data.get('repository', '')
    commit_sha = code_data.get('commit_sha', '')
    file_path_raw = code_data.get('parent_file_path', '') or code_data.get('file_path', '')
    file_path = normalize_file_path(file_path_raw, repository, commit_sha)
    
    buggy_file_content = code_data.get('buggy_file', '')
    if not buggy_file_content:
        raise ValueError(f"Sample {sample_id} missing buggy_file content (full file required)")
    
    code_blob = buggy_file_content

    return {
        "id": sample_id,
        "label": 1,
        "quantum_specific": analysis_data.get('quantum_specific', analysis_data.get('is_quantum_related', False)),
        "submodule": analysis_data.get('submodule', analysis_data.get('bug_type', '')),
        "lifecycle_stage": analysis_data.get('lifecycle_stage', ''),
        "parent_file_path": file_path,
        "repository": repository,
        "commit_sha": commit_sha,
        "commit_date": code_data.get("commit_date", ""),
        "change_category": analysis_data.get('change_category', ''),
        "bug_reason": analysis_data.get('bug_reason', ''),
        "quantum_reason": analysis_data.get('quantum_reason', ''),
        "lifecycle_reason": analysis_data.get('lifecycle_reason', ''),
        "framework": analysis_data.get('framework', ''),
        "bug_codesnipt_content": code_data.get("bug_codesnipt_content", ""),
        "diff": code_data.get("diff", ""),
        "bug_file_content": code_blob,
    }


def compute_distributions(
    positive_samples: List[Dict[str, Any]]
) -> tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Compute lifecycle-stage and submodule distributions.
    
    Args:
        positive_samples: List of positive samples.
    
    Returns:
        Tuple of (lifecycle_distribution, submodule_distribution).
    """
    lifecycle_distribution: Dict[str, int] = {}
    submodule_distribution: Dict[str, Dict[str, int]] = {}
    
    for item in positive_samples:
        # Prefer submodule and fall back to bug_type.
        submodule = item.get('submodule') or item.get('bug_type', '')
        lifecycle_stage = item.get('lifecycle_stage', '')
        
        # Count the lifecycle-stage distribution.
        lifecycle_distribution[lifecycle_stage] = lifecycle_distribution.get(lifecycle_stage, 0) + 1
        
        # Count submodule distribution per lifecycle stage.
        if lifecycle_stage not in submodule_distribution:
            submodule_distribution[lifecycle_stage] = {}
        
        submodule_distribution[lifecycle_stage][submodule] = \
            submodule_distribution[lifecycle_stage].get(submodule, 0) + 1
    
    return lifecycle_distribution, submodule_distribution


def save_dataset(
    dataset: List[Dict[str, Any]],
    output_file: Path,
    description: str,
    source_files: Dict[str, str],
    label_description: Dict[str, str],
    fields: List[str],
    additional_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save the dataset to disk.
    
    Args:
        dataset: Dataset sample list.
        output_file: Output file path.
        description: Dataset description.
        source_files: Dictionary describing source files.
        label_description: Label-description dictionary.
        fields: List of field names.
        additional_metadata: Optional extra metadata.
    """
    logging.info("\n" + "="*80)
    logging.info("Saving dataset...")
    logging.info("="*80)
    
    # Statistics.
    positive_samples = [s for s in dataset if s.get('label') == 1]
    negative_samples = [s for s in dataset if s.get('label') == 0]
    
    lifecycle_distribution, submodule_distribution = compute_distributions(positive_samples)
    
    # Build the output payload.
    metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        "source_files": source_files,
        "total_samples": len(dataset),
        "positive_samples": len(positive_samples),
        "negative_samples": len(negative_samples),
        "original_bug_count": len(positive_samples),
        "lifecycle_distribution": lifecycle_distribution,
        "submodule_distribution": submodule_distribution,
        "label_description": label_description,
        "fields": fields
    }
    
    # Add extra metadata.
    if additional_metadata:
        metadata.update(additional_metadata)
    
    output_data = {
        "metadata": metadata,
        "samples": dataset
    }
    
    # Save to file.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Dataset saved: {output_file}")
    logging.info(f"\nStatistics:")
    logging.info(f"   Total samples: {len(dataset)}")
    logging.info(f"   Positive samples: {len(positive_samples)}")
    logging.info(f"   Negative samples: {len(negative_samples)}")
    logging.info(f"   Lifecycle-stage distribution:")
    for stage, count in sorted(lifecycle_distribution.items(), key=lambda x: -x[1]):
        logging.info(f"      - {stage}: {count}")
    logging.info(f"   Total submodules: {sum(len(types) for types in submodule_distribution.values())}")
