"""Extract method-level bug-fix samples and keep single-change files only."""

import pandas as pd
import json
import gc
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings
import logging

try:
    import config
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    import config

from util.ast_method_extractor import (
    process_single_commit,
    fast_read_file,
    get_commit_source_path,
    get_cache_stats,
    set_current_framework
)

CURRENT_FRAMEWORK = config.ACTIVE_FRAMEWORK

warnings.filterwarnings("ignore", message="Pandas requires version '1.3.6' or newer of 'bottleneck'", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_memory_limit():
    """Apply lightweight GC tuning for large extraction runs."""
    try:
        gc.set_threshold(700, 10, 10)
    except Exception:
        pass

def parse_commit_url(commit_url: str):
    """Parse a GitHub commit URL into ``(owner, repo, commit_sha)``."""
    try:
        parts = commit_url.strip().split('/')
        idx = parts.index('commit')
        owner = parts[idx - 2]
        repo = parts[idx - 1]
        sha = parts[idx + 1]
        return owner, repo, sha
    except Exception:
        return "", "", ""


def load_commits_from_bugs_json(framework: str | None = None) -> list[dict]:
    """Load downloaded commits from ``bugs.json``."""
    if framework:
        framework_data_dir = config.get_framework_data_dir(framework)
        bugs_file = framework_data_dir / "bugs.json"
    else:
        bugs_file = config.BUGS_JSON_PATH
    
    if not bugs_file.exists():
        logging.warning(f"bugs.json does not exist: {bugs_file}; falling back to the status file")
        return load_commits_from_status_fallback(framework)
    
    try:
        with open(bugs_file, 'r', encoding='utf-8') as f:
            bugs_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load bugs.json: {e}")
        return load_commits_from_status_fallback()
    
    bugs_list = bugs_data.get('bugs', bugs_data.get('commits', []))
    if not bugs_list:
        logging.warning("bugs.json does not contain any records")
        return []
    
    is_file_level = bugs_list and ('buggy_file_path' in bugs_list[0] or 'file_path' in bugs_list[0])
    
    if is_file_level:
        seen_commits = set()
        results = []
        for bug in bugs_list:
            if not isinstance(bug, dict):
                continue
            
            owner = bug.get('owner', '')
            repo_name = bug.get('repo_name', '')
            commit_sha = bug.get('commit_sha', '')
            
            if not owner or not repo_name or not commit_sha:
                continue
            
            commit_key = (owner, repo_name, commit_sha)
            if commit_key in seen_commits:
                continue
            seen_commits.add(commit_key)
            
            results.append({
                'owner': owner,
                'repo_name': repo_name,
                'repository': bug.get('repository', f"{owner}/{repo_name}"),
                'commit_sha': commit_sha,
                'commit_url': bug.get('commit_url', ''),
                'commit_message': bug.get('commit_message', '')
            })
        
        logging.info(f"Loaded {len(results)} commits from bugs.json (file-level layout, {len(bugs_list)} bugs)")
    else:
        results = []
        for bug in bugs_list:
            if not isinstance(bug, dict):
                continue
            
            owner = bug.get('owner', '')
            repo_name = bug.get('repo_name', '')
            commit_sha = bug.get('commit_sha', '')
            
            if not owner or not repo_name or not commit_sha:
                continue
            
            results.append({
                'owner': owner,
                'repo_name': repo_name,
                'repository': bug.get('repository', f"{owner}/{repo_name}"),
                'commit_sha': commit_sha,
                'commit_url': bug.get('commit_url', ''),
                'commit_message': bug.get('commit_message', '')
            })
        
        logging.info(f"Loaded {len(results)} commits from bugs.json")
    
    return results


def load_commits_from_status_fallback(framework: str | None = None) -> list[dict]:
    """Fallback loader that rebuilds commit entries from the status file."""
    if framework:
        framework_data_dir = config.get_framework_data_dir(framework)
        status_file = framework_data_dir / "status" / "bug_file_download_status_main.json"
    else:
        status_file = config.RAW_STATUS_DIR / "bug_file_download_status_main.json"
    if not status_file.exists():
        return []
    
    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
    except Exception:
        return []
    
    commit_details = status_data.get('commit_details', {})
    
    results = []
    for commit_key, info in commit_details.items():
        try:
            if info.get('buggy_files_downloaded', 0) == 0:
                continue
            if info.get('status') != 'completed':
                continue
            
            repository = info.get('repository', '')
            if not repository or repository == "/":
                continue
            
            sha = info.get('commit_sha')
            commit_url = info.get('commit_url', '')
            commit_message = info.get('commit_message', '')
            
            owner, repo = None, None
            if repository and '/' in repository:
                parts = repository.split('/')
                if len(parts) == 2:
                    owner, repo = parts[0], parts[1]
                elif len(parts) >= 3:
                    owner, repo = parts[-2], parts[-1]
            
            if not owner or not repo or not sha:
                try:
                    url_owner, url_repo, url_sha = parse_commit_url(commit_url)
                    owner = owner or url_owner
                    repo = repo or url_repo
                    sha = sha or url_sha
                except Exception:
                    pass
            
            if not owner or not repo or not sha:
                continue
            
            results.append({
                'owner': owner,
                'repo_name': repo,
                'repository': repository,
                'commit_sha': sha,
                'commit_url': commit_url,
                'commit_message': commit_message
            })
        except Exception as e:
            logging.debug(f"Failed to process commit_key {commit_key}: {e}")
            continue
    
    logging.info(f"Loaded {len(results)} commits from the status file fallback")
    return results

def reconfigure_logging():
    """Ensure logging is configured in each worker process."""
    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(exist_ok=True)
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
        
    # Re-add handlers
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "extract_method_level_bug.log"),
            logging.StreamHandler()
        ]
    )


def process_pr_chunk(pr_chunk_with_info):
    """Process one chunk of commits and collect changed samples."""
    pr_chunk, pr_info_dict, framework = pr_chunk_with_info
    
    set_current_framework(framework)
    
    # Reconfigure logging for this worker process
    reconfigure_logging()

    all_modified_samples = []
    total_modified_count = 0
    
    for i, pr in enumerate(pr_chunk):
        try:
            mod_all, mod_count = process_single_commit(pr, pr_info_dict)
            all_modified_samples.extend(mod_all)
            total_modified_count += mod_count
        except Exception:
            continue
        
        if (i + 1) % 5 == 0:
            gc.collect()
    
    return all_modified_samples, total_modified_count


def analyze_prs_for_samples(pr_list, pr_info_dict, use_processes=True, num_workers=None, framework=None):
    """Collect changed samples in parallel using processes or threads."""
    if num_workers is None:
        cpu_count = mp.cpu_count()
        if use_processes:
            num_workers = cpu_count
        else:
            num_workers = min(cpu_count * 2, 16)
    
    fw = framework or CURRENT_FRAMEWORK
    
    all_modified_samples = []
    total_modified_count = 0
    
    if use_processes:
        chunk_size = max(10, len(pr_list) // (num_workers * 3))
        pr_chunks = [pr_list[i:i + chunk_size] for i in range(0, len(pr_list), chunk_size)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_pr_chunk, (chunk, pr_info_dict, fw)) for chunk in pr_chunks]
            for future in tqdm(futures, total=len(pr_chunks), desc="Processing PR chunks", disable=False):
                try:
                    chunk_all_modified, chunk_modified = future.result()
                    all_modified_samples.extend(chunk_all_modified)
                    total_modified_count += chunk_modified
                except Exception as e:
                    continue
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_pr_chunk, ([pr], pr_info_dict, fw)) for pr in pr_list]
            for future in tqdm(futures, total=len(futures), desc="Processing PRs", disable=False):
                try:
                    mod_all, mod_count = future.result()
                    all_modified_samples.extend(mod_all)
                    total_modified_count += mod_count
                except Exception:
                    continue
    
    gc.collect()
    
    return all_modified_samples, total_modified_count


# ============================================================================
# File-level code loading
# ============================================================================

def _get_file_level_code(owner: str, repo_name: str, commit_sha: str, file_path: str, version: str) -> str:
    """Load the full file content for one commit version."""
    try:
        base_path = get_commit_source_path(owner, repo_name, commit_sha, version)
        full_path = base_path / file_path
        if full_path.exists():
            return fast_read_file(full_path)
        return ""
    except Exception:
        return ""


def _get_diff_from_file(owner: str, repo_name: str, commit_sha: str, file_path: str) -> str:
    """Load a previously saved diff file."""
    try:
        diff_base_path = get_commit_source_path(owner, repo_name, commit_sha, 'diffs')
        diff_file_path = diff_base_path / f"{file_path}.diff"
        if diff_file_path.exists():
            return fast_read_file(diff_file_path)
        return ""
    except Exception:
        return ""


# ============================================================================
# Single-function filtering and dataset generation
# ============================================================================

def filter_and_save_single_func_dataset(all_modified_samples, output_json_path):
    """Filter single-function samples and save the dataset."""
    if not all_modified_samples:
        logging.info("No modified samples found. Skipping dataset creation.")
        return None
    
    # Step 1: Convert to a DataFrame.
    all_modified_df = pd.DataFrame(all_modified_samples)
    
    # Step 2: Drop invalid samples where buggy_code == fixed_code.
    before_filter_count = len(all_modified_df)
    all_modified_df = all_modified_df[
        all_modified_df['buggy_code'] != all_modified_df['fixed_code']
    ].copy()
    after_filter_count = len(all_modified_df)
    filtered_invalid = before_filter_count - after_filter_count
    
    if filtered_invalid > 0:
        logging.info("=== Invalid sample filtering ===")
        logging.info(f"Before filtering: {before_filter_count}")
        logging.info(f"After filtering: {after_filter_count}")
        logging.info(f"Removed invalid samples: {filtered_invalid}")
        if before_filter_count > 0:
            logging.info(f"Invalid ratio: {filtered_invalid / before_filter_count * 100:.2f}%")
    
    # Step 3: Keep single-function samples only.
    single_func_count = 0
    try:
        # Keep only files with a single change.
        single_func_df = all_modified_df.copy()
        if not single_func_df.empty and 'file_total_changes' in single_func_df.columns:
            # Retain rows where file_total_changes == 1.
            merged = single_func_df[single_func_df['file_total_changes'] == 1].copy()
            
            # Use a stable ordering for deterministic processing.
            merged = merged.sort_values(by=['repository', 'commit_sha', 'file_path', 'function_name']).reset_index(drop=True)
            
            # Load full-file content for deduplication and final output.
            logging.info(f"Reading full file content for {len(merged)} samples...")
            buggy_files = []
            fixed_files = []
            for idx, row in merged.iterrows():
                commit_sha = str(row.get('commit_sha', ''))
                file_path = str(row['file_path'])
                
                # Resolve repository and relative path metadata.
                owner = str(row.get('owner', ''))
                repo_name = str(row.get('repo_name', ''))
                actual_file_path = str(row.get('relative_file_path', ''))
                
                # Fall back to parsing file_path if structured fields are absent.
                if not owner or not repo_name or not actual_file_path:
                    try:
                        path_parts = file_path.split('/')
                        if len(path_parts) >= 4:
                            owner = owner or path_parts[0]
                            repo_name = repo_name or path_parts[1]
                            actual_file_path = actual_file_path or '/'.join(path_parts[3:])
                        else:
                            if not owner or not repo_name:
                                repository = str(row.get('repository', ''))
                                if '/' in repository:
                                    parts = repository.split('/')
                                    owner = owner or parts[0]
                                    repo_name = repo_name or parts[-1]
                            actual_file_path = actual_file_path or file_path
                    except Exception:
                        pass
                
                # Load buggy and fixed file snapshots.
                buggy_file = _get_file_level_code(owner, repo_name, commit_sha, actual_file_path, 'parent')
                fixed_file = _get_file_level_code(owner, repo_name, commit_sha, actual_file_path, 'commit')
                buggy_files.append(buggy_file)
                fixed_files.append(fixed_file)
            
            # Attach full-file content to the DataFrame.
            merged['buggy_file'] = buggy_files
            merged['fixed_file'] = fixed_files
            
            # Deduplicate with full-file content included.
            # file_path must stay in the key to avoid cross-file false positives.
            before_content_dedup = len(merged)
            
            # Mark duplicates so the removed records can be reported.
            merged['is_duplicate'] = merged.duplicated(subset=['commit_sha', 'file_path', 'buggy_file', 'fixed_file', 'function_name'], keep='first')
            
            # Capture the duplicated rows before dropping them.
            duplicated_samples = merged[merged['is_duplicate']].copy()
            
            # Drop duplicates.
            merged = merged[~merged['is_duplicate']].copy()
            merged = merged.drop(columns=['is_duplicate'])
            
            after_content_dedup = len(merged)
            content_removed = before_content_dedup - after_content_dedup
            
            # Persist duplicate details for inspection.
            if content_removed > 0:
                logging.info("=== Single-function filtering - content deduplication ===")
                logging.info(f"Removed duplicate samples: {content_removed}")
                
                # Build the duplicate summary payload.
                duplicated_info = {
                    "metadata": {
                        "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "description": "Samples removed as duplicates because commit_sha, file_path, buggy_file, fixed_file, and function_name were identical.",
                        "total_duplicates": content_removed,
                        "dedup_criteria": ["commit_sha", "file_path", "buggy_file", "fixed_file", "function_name"]
                    },
                    "duplicated_samples": []
                }
                
                for idx, row in duplicated_samples.iterrows():
                    dup_info = {
                        "repository": str(row.get('repository', '')),
                        "commit_sha": str(row.get('commit_sha', '')),
                        "commit_url": str(row.get('commit_url', '')),
                        "commit_message": str(row.get('commit_head', '')),
                        "file_path": str(row.get('file_path', '')),
                        "function_name": str(row.get('function_name', '')),
                        "buggy_code_length": len(str(row.get('buggy_code', ''))),
                        "fixed_code_length": len(str(row.get('fixed_code', ''))),
                        "buggy_file_length": len(str(row.get('buggy_file', ''))),
                        "fixed_file_length": len(str(row.get('fixed_file', '')))
                    }
                    duplicated_info["duplicated_samples"].append(dup_info)
                
                # Save the duplicate report.
                dedup_output_path = output_json_path.parent / "method_level_single_duplicates.json"
                with open(dedup_output_path, 'w', encoding='utf-8') as f:
                    json.dump(duplicated_info, f, indent=2, ensure_ascii=False)
                
                logging.info(f"Duplicate sample details saved to: {dedup_output_path}")
            else:
                logging.info("=== Single-function filtering - content deduplication ===")
                logging.info("No duplicate samples found using commit_sha, file_path, buggy_file, fixed_file, and function_name.")
            
            # The saved count may be smaller if diff files are missing.
            single_func_count = len(merged)
        else:
            logging.warning("The DataFrame does not contain 'file_total_changes'; skipping single-function filtering.")
            merged = pd.DataFrame()
            single_func_count = 0
        
        if not merged.empty:
            single_dataset = {
                "metadata": {
                    "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "description": "Bug-fix samples where the file has exactly one change.",
                    "filter_criteria": "file_total_changes == 1",
                    "total_samples": 0,
                    "fields": [
                        "repository",
                        "commit_sha",
                        "commit_url",
                        "commit_message",
                        "buggy_file_path",
                        "function_name",
                        "buggy_code",
                        "fixed_code",
                        "buggy_file",
                        "fixed_file",
                        "diff"
                    ]
                },
                "samples": []
            }

            for idx, row in merged.iterrows():
                # Collect the metadata needed to build the final sample.
                commit_sha = str(row.get('commit_sha', ''))
                file_path = str(row['file_path'])
                repository = str(row.get('repository', ''))
                
                # Resolve owner, repo name, and the actual relative file path.
                owner = str(row.get('owner', ''))
                repo_name = str(row.get('repo_name', ''))
                actual_file_path = str(row.get('relative_file_path', ''))
                
                # Fall back to parsing file_path if structured fields are absent.
                if not owner or not repo_name or not actual_file_path:
                    try:
                        path_parts = file_path.split('/')
                        if len(path_parts) >= 4:
                            owner = owner or path_parts[0]
                            repo_name = repo_name or path_parts[1]
                            actual_file_path = actual_file_path or '/'.join(path_parts[3:])
                        else:
                            # Fall back to the repository field.
                            if not owner or not repo_name:
                                repository = str(row.get('repository', ''))
                                if '/' in repository:
                                    parts = repository.split('/')
                                    owner = owner or parts[0]
                                    repo_name = repo_name or parts[-1]
                            actual_file_path = actual_file_path or file_path
                    except Exception:
                        pass
                
                # Reuse the buggy and fixed file snapshots loaded before deduplication.
                buggy_file = str(row.get('buggy_file', ''))
                fixed_file = str(row.get('fixed_file', ''))
                
                # Pull the function-level buggy and fixed code.
                buggy_code = str(row.get('buggy_code', ''))
                fixed_code = str(row.get('fixed_code', ''))
                
                # The diff file must exist for the sample to be valid.
                diff_text = _get_diff_from_file(owner, repo_name, commit_sha, actual_file_path)
                if not diff_text:
                    logging.error(
                        f"Missing diff file, skipping sample: {actual_file_path} "
                        f"(repo={repository}, commit={commit_sha[:8]})"
                    )
                    continue
                
                # Build the final buggy_file_path with the framework segment.
                buggy_file_path = f"data/02_source/{CURRENT_FRAMEWORK}/{owner}/{repo_name}/{commit_sha}/parent/{actual_file_path}"
                
                sample = {
                    'repository': str(row.get('repository', row.get('project_name', ''))),
                    'commit_sha': commit_sha,
                    'commit_url': str(row.get('commit_url', '')),
                    'commit_message': str(row.get('commit_head', '')),
                    'buggy_file_path': buggy_file_path,
                    'function_name': str(row['function_name']),
                    'buggy_code': buggy_code,
                    'fixed_code': fixed_code,
                    'buggy_file': buggy_file,
                    'fixed_file': fixed_file,
                    'diff': diff_text
                }
                single_dataset['samples'].append(sample)
            
            # Update the actual saved count.
            actual_saved_count = len(single_dataset['samples'])
            single_dataset['metadata']['total_samples'] = actual_saved_count
            
            # Keep single_func_count aligned with the true saved count.
            if actual_saved_count != single_func_count:
                logging.warning(f"{single_func_count - actual_saved_count} samples were skipped because their diff files were missing.")
                single_func_count = actual_saved_count
            
            logging.info(f"Single-function filtering complete: saved {actual_saved_count} samples.")

            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(single_dataset, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed during single-function filtering: {e}")
        pass
    
    return {
        'all_modified_samples': len(all_modified_df),
        'single_func_samples': single_func_count
    }


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Run method-level bug extraction with single-function filtering."""
    parser = argparse.ArgumentParser(description="Extract method-level bug samples with single-function filtering.")
    parser.add_argument('--framework', type=str, default=config.ACTIVE_FRAMEWORK,
                        choices=list(config.QUANTUM_FRAMEWORKS.keys()),
                        help='Quantum framework to process.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path. Defaults to the framework-specific location.')
    args = parser.parse_args()
    
    global CURRENT_FRAMEWORK
    CURRENT_FRAMEWORK = args.framework
    framework = args.framework
    framework_config = config.get_framework_config(framework)
    
    # Keep the AST extractor aligned with the active framework.
    set_current_framework(framework)
    
    set_memory_limit()
    
    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"extract_method_level_bug_{framework}.log"),
            logging.StreamHandler()
        ]
    )
    
    start_time = time.time()
    
    # Resolve the output path, using the framework-specific default when needed.
    if args.output:
        output_json_path = Path(args.output)
    else:
        framework_extracted_dir = config.EXTRACTED_DIR / framework
        framework_extracted_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = framework_extracted_dir / "method_level_single.json"
    
    logging.info("=" * 60)
    logging.info(f"Starting method-level bug extraction for {framework_config['name']}.")
    logging.info(f"Output file: {output_json_path}")
    logging.info("=" * 60)
    
    try:
        if hasattr(config, 'ensure_method_extraction_structure'):
            config.ensure_method_extraction_structure(framework)
            
        # Load commits from bugs.json first, then fall back to the status file.
        commits_list = load_commits_from_bugs_json(framework)
        if not commits_list:
            logging.error("No completed commits were found.")
            return
        logging.info(f"Loaded {len(commits_list)} commits.")
        
        if hasattr(config, 'EXCLUDED_REPOS'):
            original_count = len(commits_list)
            commits_list = [c for c in commits_list if c['repository'] not in config.EXCLUDED_REPOS]
            excluded_count = original_count - len(commits_list)
            if excluded_count > 0:
                logging.info(f"Excluded {excluded_count} commits; {len(commits_list)} remain.")
        
        # Use deterministic ordering across runs.
        commits_list = sorted(commits_list, key=lambda x: (x.get('repository', ''), x.get('commit_sha', '')))
        
        use_processes = len(commits_list) > 500
        pr_info_dict = {}
        
        logging.info("Starting method-level sample extraction...")
        analysis_start = time.time()
        
        all_modified, total_mod = analyze_prs_for_samples(
            commits_list, pr_info_dict, use_processes=use_processes, framework=framework
        )
        
        analysis_time = time.time() - analysis_start
        
        logging.info(f"\n{'='*60}")
        logging.info("Sample collection summary")
        logging.info(f"{'='*60}")
        logging.info(f"Processed commits: {len(commits_list)}")
        logging.info(f"Collected samples: {len(all_modified)}")
        logging.info(f"Modified functions: {total_mod}")
        logging.info(f"Processing time: {analysis_time:.2f}s")
        logging.info(f"{'='*60}\n")

        logging.info("Starting single-function filtering...")
        summary_stats = filter_and_save_single_func_dataset(all_modified, output_json_path)
        
        if summary_stats:
            logging.info(f"\n{'='*60}")
            logging.info("Final summary")
            logging.info(f"{'='*60}")
            logging.info(f"All modified samples: {summary_stats['all_modified_samples']}")
            logging.info(f"Single-function samples: {summary_stats['single_func_samples']}")
            if summary_stats['all_modified_samples'] > 0:
                logging.info(f"Retention ratio: {summary_stats['single_func_samples'] / summary_stats['all_modified_samples'] * 100:.2f}%")
            else:
                logging.info("Retention ratio: N/A (no samples)")
            logging.info(f"{'='*60}")
        else:
            logging.warning("Could not generate summary statistics.")

        total_time = time.time() - start_time
        logging.info(f"\nTotal processing time: {total_time:.2f}s")
        logging.info(f"Output file: {output_json_path}")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
