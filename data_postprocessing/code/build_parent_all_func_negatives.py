#!/usr/bin/env python3
"""
Build negative samples from parent revisions by extracting all eligible files for each target commit.
(Parent All File Negatives)

Features:
1. **Commit-based construction**:
   For each commit containing a Bug Fix, extract **all eligible files** from its parent commit as negatives
   at file level rather than function level. The output contains both positive samples for the commit and
   negative files from the parent revision.

2. **Repository-level deduplication**:
   While processing a single repository, negative samples are deduplicated **by file content**.
   If the same file content appears in multiple parent commits and remains unchanged, only the first occurrence is kept.

3. **Parallel strategy**:
   - **Across repos**: Different repositories are processed in different threads or processes.
   - **Within a repo**: Different commits in the same repository are processed serially to preserve dedup state
     and avoid Git lock conflicts.

Usage:
    python data_postprocessing/code/build_parent_all_func_negatives.py
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


def repository_relative_path(path: Path) -> str:
    """Return a portable path relative to the repository root."""
    return Path(os.path.relpath(path.resolve(), PROJECT_ROOT.resolve())).as_posix()

# Reused utility helpers.
from data_postprocessing.code.util.dataset_utils import (
    normalize_file_path,
    build_positive_sample,
    save_dataset,
    get_latest_commit_sha,
    get_all_modified_files_in_range,
)  # type: ignore

# Import the framework-import checker.
from config import contains_framework_import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Keep both positive and negative samples bounded for storage/runtime cost.
POSITIVE_MAX_CODE_LENGTH = 250000
NEGATIVE_MAX_CODE_LENGTH = 250000


@dataclass
class BugSample:
    """Sample metadata containing all required information from the positive result file."""
    sample_id: str
    repository: str
    commit_sha: str
    parent_sha: str
    parent_file_path: str  # Unified field name.
    function_name: str
    buggy_file: str  # Full buggy file content.
    # Analysis-result fields.
    quantum_specific: bool = False  # Unified field name.
    submodule: str = ""
    lifecycle_stage: str = ""
    quantum_reason: str = ""
    lifecycle_reason: str = ""
    bug_reason: str = ""
    framework: str = ""  # Framework (qiskit/cirq/pennylane).


def load_all_from_final(final_file: Path) -> Tuple[List[BugSample], Dict[str, Dict], Dict[Tuple[str, str], str]]:
    """
    Load all data from the positive result file, including sample metadata, code, and parent_sha.
    
    Returns:
        samples: List of BugSample objects.
        code_mapping: {sample_id: {buggy_file, fixed_file, ...}}
        commit_parents: {(repo, commit_sha): parent_sha}
    """
    with final_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    code_mapping = {}
    commit_parents = {}
    
    for r in data.get("results", []):
        # Only process Bug Fix samples.
        if r.get("change_category") != "Bug Fix":
            continue
        
        sid = str(r.get("id") or r.get("sample_id", ""))
        repository = r.get("repository", "")
        commit_sha = r.get("commit_sha", "")
        parent_sha = r.get("parent_sha", "")
        buggy_file = r.get("buggy_file", "")
        bug_codesnipt_content = r.get("buggy_code", "")
        file_path = r.get("parent_file_path", "") or r.get("file_path", "")
        function_name = r.get("function_name", "")
        
        # buggy_file and parent_sha are required.
        if not buggy_file or not parent_sha:
            logging.debug(f"Skipping sample {sid}: missing buggy_file or parent_sha")
            continue
        
        sample = BugSample(
            sample_id=sid,
            repository=repository,
            commit_sha=commit_sha,
            parent_sha=parent_sha,
            parent_file_path=file_path,
            function_name=function_name,
            buggy_file=buggy_file,
            quantum_specific=r.get("quantum_specific", False),
            submodule=r.get("submodule", ""),
            lifecycle_stage=r.get("lifecycle_stage", ""),
            quantum_reason=r.get("quantum_reason", ""),
            lifecycle_reason=r.get("lifecycle_reason", ""),
            bug_reason=r.get("bug_reason", ""),
            framework=r.get("framework", ""),
        )
        samples.append(sample)
        
        # Build code_mapping.
        code_mapping[sid] = {
            "buggy_file": buggy_file,
            "bug_codesnipt_content": bug_codesnipt_content,
            "parent_file_path": file_path,
            "repository": repository,
            "commit_sha": commit_sha,
            "function_name": function_name,
        }
        
        # Build commit_parents.
        if repository and commit_sha and parent_sha:
            commit_parents[(repository, commit_sha)] = parent_sha
    
    logging.info(f"Loaded {len(samples)} positive samples from {final_file.name}")
    
    return samples, code_mapping, commit_parents


def load_diff_mapping_from_merged(merged_file: Path) -> Dict[str, str]:
    """Load sample_id -> diff mapping from method_level_single_merged.json."""
    if not merged_file.exists():
        logging.warning(f"Merged file not found: {merged_file}")
        return {}

    with merged_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, str] = {}
    for sample in data.get("samples", []):
        sid = str(sample.get("id") or sample.get("sample_id", ""))
        if not sid:
            continue
        mapping[sid] = sample.get("diff", "") or ""

    logging.info(f"Loaded {len(mapping)} diff entries from merged dataset")
    return mapping


def load_commit_dates(raw_data_dir: Path) -> Dict[Tuple[str, str], str]:
    """Load (repository, commit_sha) -> commit date ISO string."""
    commit_dates: Dict[Tuple[str, str], str] = {}
    for framework in ("qiskit", "cirq", "pennylane"):
        commits_file = raw_data_dir / framework / "commits.json"
        if not commits_file.exists():
            continue
        try:
            with commits_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load commit dates from {commits_file}: {e}")
            continue

        for repository, commit_list in data.get("commits", {}).items():
            for commit in commit_list:
                commit_sha = commit.get("commit_sha")
                date_str = commit.get("date")
                if repository and commit_sha and date_str:
                    commit_dates[(repository, commit_sha)] = date_str

    logging.info(f"Loaded {len(commit_dates)} commit-date mappings")
    return commit_dates





def checkout_commit(repo_dir: Path, commit_sha: str, force: bool = False):
    """Checkout the specified commit."""
    cmd = ["git", "-C", str(repo_dir), "checkout", "--quiet"]
    if force:
        cmd.append("--force")
    cmd.append(commit_sha)
    subprocess.run(cmd, check=True)


def fetch_origin(repo_dir: Path):
    """Ensure remote references are up to date."""
    subprocess.run(
        ["git", "-C", str(repo_dir), "fetch", "origin", "--prune"],
        capture_output=True,
        text=True,
        check=True,
    )





def list_python_files_at_commit(repo_dir: Path, commit_sha: str, exclude_test: bool = False) -> List[str]:
    """List all Python file paths at a specific commit without checkout, using git ls-tree."""
    res = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", commit_sha],
        capture_output=True,
        text=True,
        check=True,
    )
    files: List[str] = []
    for line in res.stdout.splitlines():
        path = line.strip()
        if not path.endswith(".py"):
            continue
        parts = path.split("/")
        if any(part in {".venv", "venv", "__pycache__", ".git", ".ipynb_checkpoints"} for part in parts):
            continue
        if exclude_test and 'test' in path.lower():
            continue
        files.append(path)
    return files


def get_file_content_at_commit(repo_dir: Path, commit_sha: str, file_path: str) -> Optional[str]:
    """Read a file's contents at a specific commit without checkout, using git show."""
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_dir), "show", f"{commit_sha}:{file_path}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        return None




def calculate_content_hash(code: str) -> str:
    """Compute a hash of the code content after trimming leading and trailing whitespace."""
    return hashlib.md5(code.strip().encode("utf-8")).hexdigest()


def list_modified_files_between(repo_dir: Path, base_sha: str, head_sha: str) -> List[str]:
    """Get the deduplicated, sorted list of modified files between base_sha and head_sha."""
    res = subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "diff",
            "--name-only",
            base_sha,
            head_sha,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted({line.strip() for line in res.stdout.splitlines() if line.strip()})


class RepositoryProcessor:
    """Process all commits in one repository serially so deduplication can be maintained."""
    
    def __init__(
        self,
        repository: str,
        commit_groups: Dict[str, List[BugSample]],
        repo_root_dir: Path,
        commit_parents: Dict[Tuple[str, str], str],
        code_mapping: Dict[str, Dict],
        analysis_results: Dict[str, Dict],
        collect_modified: bool = False,
        fetch_latest: bool = False,
        exclude_test: bool = False,
        fail_fast: bool = False,
    ):
        self.repository = repository
        self.commit_groups = commit_groups
        self.repo_root_dir = repo_root_dir
        self.commit_parents = commit_parents
        self.code_mapping = code_mapping
        self.analysis_results = analysis_results
        self.collect_modified = collect_modified
        self.fetch_latest = fetch_latest
        self.exclude_test = exclude_test
        self.fail_fast = fail_fast
        # Cache key: (parent_sha, target_head) -> modified_set (or None if failed)
        self.modified_set_cache: Dict[Tuple[str, str], Optional[Set[str]]] = {}
        
        # Simple owner/repo split.
        try:
            self.owner, self.name = repository.split("/", 1)
        except:
            self.owner, self.name = None, None

        # Use the shared helper to locate the repository directory.
        from data_postprocessing.code.util.dataset_utils import get_repo_dir
        self.repo_dir = get_repo_dir(repo_root_dir, repository) if self.owner and self.name else None

    def get_current_ref(self) -> Optional[str]:
        """Get the current branch name or HEAD commit so it can be restored later."""
        if not self.repo_dir:
            return None
        try:
            res = subprocess.run(
                ["git", "-C", str(self.repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True
            )
            ref = res.stdout.strip()
            if ref != "HEAD":
                return ref
            # In detached HEAD mode, restore using the concrete commit hash.
            res = subprocess.run(
                ["git", "-C", str(self.repo_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True
            )
            return res.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def process(self) -> Dict[str, Any]:
        """Run the processing logic for this repository."""
        results = []
        modified_records: List[Dict[str, Any]] = []
        failed_commits: List[Dict[str, str]] = []
        
        if not self.repo_dir or not self.repo_dir.is_dir():
            logging.warning(f"Repo dir not found: {self.repository}")
            return {"samples": [], "modified": []}

        # Narrow the type so later calls are no longer Optional-based.
        assert self.repo_dir is not None
        repo_dir: Path = self.repo_dir

        # Track seen code hashes for repository-level deduplication.
        seen_content_hashes: Set[str] = set()

        # Record the initial branch/commit so it can be restored later.
        original_ref = self.get_current_ref()
        head_upper_sha = get_latest_commit_sha(repo_dir)

        if self.collect_modified and self.fetch_latest:
            try:
                fetch_origin(self.repo_dir)
                # Resolve the latest commit again after fetch.
                head_upper_sha = get_latest_commit_sha(repo_dir)
            except Exception as e:
                logging.warning(f"Fetch origin failed for {self.repository}: {e}")
        
        try:
            # Process commits in sorted SHA order to keep deduplication deterministic.
            for commit_sha in sorted(self.commit_groups.keys()):
                samples = self.commit_groups[commit_sha]
                commit_results, commit_mods, commit_error = self.process_commit(
                    commit_sha, samples, seen_content_hashes, head_upper_sha
                )
                results.extend(commit_results)
                modified_records.extend(commit_mods)
                if commit_error:
                    failed_commits.append(
                        {
                            "repository": self.repository,
                            "commit_sha": commit_sha,
                            "error": commit_error,
                        }
                    )
                    if self.fail_fast:
                        break
        finally:
            if original_ref:
                try:
                    checkout_commit(self.repo_dir, original_ref)
                except Exception as e:
                    logging.warning(f"Failed to restore repo {self.repository} to {original_ref}: {e}")

        # Return data plus modified-file records.
        return {"samples": results, "modified": modified_records, "failed_commits": failed_commits}

    def process_commit(
        self, 
        commit_sha: str, 
        samples: List[BugSample], 
        seen_content_hashes: Set[str],
        head_upper_sha: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
        assert self.repo_dir is not None
        repo_dir: Path = self.repo_dir

        output_items = []
        modified_records: List[Dict[str, Any]] = []
        commit_error: Optional[str] = None
        
        # 1. Add all positive samples for this commit.
        skipped_positive_too_long = 0
        for sample in samples:
            code_data = self.code_mapping.get(sample.sample_id)
            analysis_data = self.analysis_results.get(sample.sample_id, {})
            if code_data:
                # Only inspect buggy_file at file level.
                buggy_file_content = code_data.get("buggy_file", "")
                
                if not buggy_file_content:
                    logging.debug(f"Skipping positive sample {sample.sample_id}: missing buggy_file")
                    continue

                # Filter out positive samples that exceed the size threshold.
                if len(buggy_file_content) > POSITIVE_MAX_CODE_LENGTH:
                    skipped_positive_too_long += 1
                    continue
                
                try:
                    pos_item = build_positive_sample(sample.sample_id, code_data, analysis_data)
                    output_items.append(pos_item)
                except Exception as e:
                    logging.warning(f"Failed to build positive sample {sample.sample_id}: {e}")
        if skipped_positive_too_long:
            logging.debug(
                "Commit %s@%s: skipped_positive_too_long=%d",
                self.repository[:30],
                commit_sha[:8],
                skipped_positive_too_long,
            )

        # 2. Mine negative samples at file level.
        # Get the parent commit.
        parent_sha = self.commit_parents.get((self.repository, commit_sha))
        if not parent_sha:
            logging.warning(f"No parent sha for {self.repository} {commit_sha}")
            return output_items, modified_records, "missing_parent_sha"  # Return positive samples only.

        # Optional: record the modified-file list from parent to latest.
        if self.collect_modified and head_upper_sha:
            try:
                modified_files = list_modified_files_between(repo_dir, parent_sha, head_upper_sha)
                modified_records.append(
                    {
                        "sample_id": [s.sample_id for s in samples],
                        "repository": self.repository,
                        "commit_sha": commit_sha,
                        "parent_sha": parent_sha,
                        "latest_sha": head_upper_sha,
                        "modified_files": modified_files,
                    }
                )
            except Exception as e:
                logging.warning(f"List modified files failed for {self.repository} {commit_sha}: {e}")
        
        try:
            # Get all files modified between Parent and Latest (or HEAD).
            # This covers:
            # 1. Files changed by the current Bug Fix commit (included in parent..latest)
            # 2. Files changed by any future commit
            target_head = head_upper_sha if head_upper_sha else "HEAD"
            cache_key = (parent_sha, target_head)
            if cache_key in self.modified_set_cache:
                modified_set = self.modified_set_cache[cache_key]
            else:
                modified_set = get_all_modified_files_in_range(repo_dir, parent_sha, target_head)
                self.modified_set_cache[cache_key] = modified_set
            
            # If modified files cannot be retrieved, skip negative mining for this commit as a conservative fallback.
            if modified_set is None:
                logging.warning(f"Skipping negative mining for {self.repository}@{commit_sha}: failed to get modified files")
                return output_items, modified_records, "failed_to_get_modified_files"

            # Get all Python files from the parent commit.
            candidate_files = list_python_files_at_commit(repo_dir, parent_sha, self.exclude_test)
            
            # Iterate over Python files from the parent commit without checkout.
            skipped_modified = 0
            skipped_content = 0
            skipped_no_framework = 0
            framework = samples[0].framework if samples else ""
            for rel_path in candidate_files:
                # Filter 1: exclude files modified at any point from Parent to Latest.
                if rel_path in modified_set:
                    skipped_modified += 1
                    continue
                
                # Read the file contents from the parent commit.
                file_content = get_file_content_at_commit(repo_dir, parent_sha, rel_path)
                if file_content is None or not file_content.strip():
                    skipped_content += 1
                    continue
                
                # Filter out oversized samples above 250k characters.
                if len(file_content) > NEGATIVE_MAX_CODE_LENGTH:
                    skipped_content += 1
                    continue
                
                # Filter: the file must contain framework imports such as import qiskit / from qiskit.
                if framework and not contains_framework_import(file_content, framework):
                    skipped_no_framework += 1
                    continue
                
                # Filter 4: deduplicate by file content.
                c_hash = calculate_content_hash(file_content)
                if c_hash in seen_content_hashes:
                    continue
                
                seen_content_hashes.add(c_hash)
                
                # Generate a unique negative-sample ID.
                neg_id = f"neg_{self.repository.replace('/', '_')}_{commit_sha[:8]}_{len(output_items)}"
                
                # Keep negative-sample fields aligned with positive samples and fill missing values with empty strings.
                neg_item = {
                    "id": neg_id,
                    "label": 0,
                    "quantum_specific": False,
                    "submodule": "",
                    "lifecycle_stage": "",
                    "parent_file_path": rel_path,
                    "repository": self.repository,
                    "commit_sha": commit_sha,
                    "commit_date": "",
                    "change_category": "",
                    "bug_reason": "",
                    "quantum_reason": "",
                    "lifecycle_reason": "",
                    "framework": samples[0].framework if samples else "",
                    "diff": "",
                    "bug_codesnipt_content": "",
                    "bug_file_content": file_content,
                }
                output_items.append(neg_item)

            neg_count = len(output_items) - len(samples)  # Excluding the number of positive samples.
            logging.debug(
                "Commit %s@%s: candidate=%d, skipped_modified=%d, skipped_content=%d, skipped_no_framework=%d, dedup_kept=%d",
                self.repository[:30],
                commit_sha[:8],
                len(candidate_files),
                skipped_modified,
                skipped_content,
                skipped_no_framework,
                neg_count,
            )
                    
        except Exception as e:
            logging.error(f"Error processing commit {commit_sha} in {self.repository}: {e}")
            commit_error = str(e)
        
        return output_items, modified_records, commit_error


def main():
    parser = argparse.ArgumentParser(description="Build the full Parent negative-sample dataset")
    parser.add_argument("--final-file", type=Path, default=PROJECT_ROOT / "data/04_analyzed/final/bugfix_positive_samples.json",
                        help="Path to the positive result file containing all sample metadata, code, and parent_sha")
    parser.add_argument("--repo-root-dir", type=Path, default=PROJECT_ROOT / "data/02_source")
    parser.add_argument("--output-file", type=Path, default=PROJECT_ROOT / "data/05_datasets/dataset_parent_all.json")
    parser.add_argument("--merged-file", type=Path, default=PROJECT_ROOT / "data/03_extracted/method_level_single_merged.json")
    parser.add_argument("--raw-data-dir", type=Path, default=PROJECT_ROOT / "data/01_raw")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--output-modified-files", type=Path, default=None, help="Optional JSON output path for saving the modified-file list from parent to latest")
    parser.add_argument("--fetch-latest", action="store_true", help="Run git fetch origin before generating modified-file lists to ensure freshness")
    parser.add_argument("--include-test", action="store_true", help="Include test files as negative samples (excluded by default)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop early within a repository when a commit-level error occurs")
    parser.add_argument("--allow-commit-errors", action="store_true", help="Allow commit-level errors and still exit successfully")
    
    args = parser.parse_args()
    
    # 1. Load all data from the positive result file: samples, code, and parent_sha.
    all_samples, code_mapping, commit_parents = load_all_from_final(args.final_file)
    diff_mapping = load_diff_mapping_from_merged(args.merged_file)
    commit_date_mapping = load_commit_dates(args.raw_data_dir)

    for sid, code_data in code_mapping.items():
        if sid in diff_mapping:
            code_data["diff"] = diff_mapping[sid]
        repo = code_data.get("repository", "")
        commit_sha = code_data.get("commit_sha", "")
        if repo and commit_sha:
            code_data["commit_date"] = commit_date_mapping.get((repo, commit_sha), "")
    
    # Build analysis_results for downstream processing.
    analysis_results = {
        s.sample_id: {
            'quantum_specific': s.quantum_specific,
            'submodule': s.submodule,
            'lifecycle_stage': s.lifecycle_stage,
            'quantum_reason': s.quantum_reason,
            'lifecycle_reason': s.lifecycle_reason,
            'bug_reason': s.bug_reason,
            'change_category': 'Bug Fix',
            'framework': s.framework,
        }
        for s in all_samples
    }
    
    # 2. Group by repository and commit.
    repo_groups: Dict[str, Dict[str, List[BugSample]]] = defaultdict(lambda: defaultdict(list))
    
    for s in all_samples:
        repo_groups[s.repository][s.commit_sha].append(s)
        
    logging.info(f"Processing {len(repo_groups)} repositories...")
    
    final_dataset = []
    final_modified: List[Dict[str, Any]] = []
    failed_commits_all: List[Dict[str, str]] = []
    
    # 3. Process repositories in parallel.
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for repo, commit_map in repo_groups.items():
            processor = RepositoryProcessor(
                repository=repo,
                commit_groups=commit_map,
                repo_root_dir=args.repo_root_dir,
                commit_parents=commit_parents,
                code_mapping=code_mapping,
                analysis_results=analysis_results,
                collect_modified=bool(args.output_modified_files),
                fetch_latest=args.fetch_latest,
                exclude_test=not args.include_test,
                fail_fast=args.fail_fast,
            )
            futures.append(executor.submit(processor.process))
            
        # Progress bar.
        if tqdm:
            pbar = tqdm(total=len(futures), desc="Processing Repos")
            
        for future in as_completed(futures):
            try:
                res = future.result()
                final_dataset.extend(res.get("samples", []))
                final_modified.extend(res.get("modified", []))
                failed_commits_all.extend(res.get("failed_commits", []))
            except Exception as e:
                logging.error(f"Task failed: {e}")
            if tqdm:
                pbar.update(1)
                
        if tqdm:
            pbar.close()
            
    # 4. Sort and save the results to ensure deterministic ordering.
    final_dataset.sort(key=lambda x: (x.get("repository", ""), x.get("id", "")))
    logging.info(f"Collected {len(final_dataset)} total samples.")
    if failed_commits_all:
        logging.warning("Commit-level errors: %d", len(failed_commits_all))
        for item in failed_commits_all[:20]:
            logging.warning("  - %s@%s: %s", item["repository"], item["commit_sha"][:8], item["error"])
        if len(failed_commits_all) > 20:
            logging.warning("  ... (%d more)", len(failed_commits_all) - 20)

    save_dataset(
        dataset=final_dataset,
        output_file=args.output_file,
        description="Quantum bug-detection dataset - positive samples plus full Parent negatives",
        source_files={
            "final": repository_relative_path(args.final_file),
        },
        label_description={
            "0": "Parent Stable Function (Negative)"
        },
        fields=["id", "label", "bug_file_content", "repository", "commit_sha", "file_path", 
                "submodule", "lifecycle_stage", "bug_reason", "quantum_reason", "lifecycle_reason",
                "diff", "bug_codesnipt_content", "commit_date"]
    )

    # Optional: output modified-file lists from parent to latest.
    if args.output_modified_files:
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": {
                "final_file": repository_relative_path(args.final_file),
            },
            "results": final_modified,
        }
        args.output_modified_files.parent.mkdir(parents=True, exist_ok=True)
        with args.output_modified_files.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info("Saved parent->latest modified file lists to %s (items: %d)", args.output_modified_files, len(final_modified))

    if failed_commits_all and not args.allow_commit_errors:
        logging.error("Build finished with commit-level errors. Re-run with --allow-commit-errors to ignore.")
        sys.exit(1)

if __name__ == "__main__":
    main()
