import argparse
import json
import os
import sys
from pathlib import Path
from github import Github, RateLimitExceededException, GithubException
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import time
import tempfile
import shutil
import re
import threading
from datetime import datetime, timezone
from collections import defaultdict

# Global per-status-file locks to avoid concurrent overwrite races.
_STATUS_FILE_LOCKS = {}
_STATUS_FILE_LOCKS_GUARD = threading.Lock()

def _get_status_file_lock(file_path_str):
    """Return a per-file thread lock reused within the same process."""
    with _STATUS_FILE_LOCKS_GUARD:
        lock = _STATUS_FILE_LOCKS.get(file_path_str)
        if lock is None:
            lock = threading.Lock()
            _STATUS_FILE_LOCKS[file_path_str] = lock
        return lock

def _read_existing_status_map(status_file: str):
    """Load the repo -> status map from a status file."""
    status_path = Path(status_file)
    if not status_path.exists():
        return {}
    try:
        with open(status_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and "repo_details" in data:
            return data["repo_details"] if isinstance(data["repo_details"], dict) else {}
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def initialize_status_placeholders(repo_names, status_file, total_repos_count):
    """Atomically initialize placeholders for a batch of repositories."""
    status_path = Path(status_file)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    file_lock = _get_status_file_lock(str(status_path.resolve()))

    with file_lock:
        existing_status = {}
        data = {}
        if status_path.exists():
            try:
                with open(status_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and "repo_details" in data:
                    existing_status = data["repo_details"]
                elif isinstance(data, dict):
                    existing_status = data
                else:
                    existing_status = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_status = {}

        # Add placeholders in bulk without overwriting existing entries.
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for repo in repo_names:
            prev = existing_status.get(repo)
            if not isinstance(prev, dict):
                existing_status[repo] = {"started": False, "last_updated": now}

        valid_statuses = {k: v for k, v in existing_status.items() if isinstance(v, dict)}
        successful_repos = sum(1 for s in valid_statuses.values() if s.get("success", False))
        failed_repos = sum(1 for s in valid_statuses.values() if s.get("started", False) and not s.get("success", False))

        summary = {
            "total_repos": total_repos_count,
            "completed_repos": successful_repos + failed_repos,
            "successful_repos": successful_repos,
            "failed_repos": failed_repos,
            "total_api_commits": sum(s.get("api_count", 0) for s in valid_statuses.values()),
            "total_matched_commits": sum(s.get("matched_count", 0) for s in valid_statuses.values()),
            "last_updated": now,
            "repo_details": valid_statuses
        }

        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=status_path.parent, delete=False, suffix='.tmp'
        ) as tmp_file:
            json.dump(summary, tmp_file, indent=4, ensure_ascii=False)
            tmp_file.flush()
            shutil.move(tmp_file.name, status_path)

# Import shared configuration from the project root.
try:
    import config
    from util.retry import RetryableRequest, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
    from util.token_manager import TokenRotator, detect_github_tokens, calculate_sleep_time
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    import config
    from util.retry import RetryableRequest, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
    from util.token_manager import TokenRotator, detect_github_tokens, calculate_sleep_time

# ==============================================================================
# SCRIPT-SPECIFIC CONFIGURATION
# ==============================================================================
# Default keywords for searching commit messages.
DEFAULT_KEYWORDS = [
    "bug", "fix", "error", "issue", "mistake", "defect",
    "incorrect", "fault", "flaw", "type"
]

# Token helpers now live in util/token_manager.py.

def ensure_get_commit_paths(framework: str | None, output_file: str | Path) -> None:
    """Create only the directories required by get_commit."""
    if hasattr(config, 'ensure_commit_collection_structure'):
        config.ensure_commit_collection_structure(framework)
    else:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        config.get_framework_data_dir(framework).mkdir(parents=True, exist_ok=True)
        (config.get_framework_data_dir(framework) / "status").mkdir(parents=True, exist_ok=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

def load_processed_repos_from_status(status_file):
    """Load processed and pending repositories from the status file."""
    processed_repos = set()
    to_process_repos = set()
    
    if not Path(status_file).exists():
        logging.info("Status file does not exist; all repositories will be processed from scratch.")
        return processed_repos, to_process_repos
    
    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract repo_details from either the new or legacy format.
        if isinstance(data, dict) and "repo_details" in data:
            repo_details = data["repo_details"]
        elif isinstance(data, dict):
            repo_details = data
        else:
            logging.warning("Status file format is invalid; all repositories will be processed from scratch.")
            return processed_repos, to_process_repos
        
        for repo_name, status in repo_details.items():
            if isinstance(status, dict):
                started = status.get("started", False)
                is_success = status.get("success", None)
                if is_success is True:
                    processed_repos.add(repo_name)
                elif (started and is_success is False) or (not started):
                    to_process_repos.add(repo_name)
                else:
                    # Default incomplete or unknown states to pending.
                    to_process_repos.add(repo_name)
        
        logging.info("Status file summary:")
        logging.info(f"  Successful repositories: {len(processed_repos)}")
        logging.info(f"  Repositories still to process: {len(to_process_repos)}")
        
        return processed_repos, to_process_repos
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.warning(f"Could not read status file {status_file}: {e}. Processing will restart from scratch.")
        return processed_repos, to_process_repos

def filter_repos_by_status(repo_list, status_file):
    """Keep only repositories that are unstarted or previously failed."""
    processed_repos, to_process_repos = load_processed_repos_from_status(status_file)
    
    # Keep repositories that still need work.
    repos_to_process = []
    skipped_count = 0
    
    for repo in repo_list:
        if repo in processed_repos:
            skipped_count += 1
            continue
        else:
            repos_to_process.append(repo)
    
    logging.info("Repository filtering summary:")
    logging.info(f"  Skipped already-processed repositories: {skipped_count}")
    logging.info(f"  Repositories to process: {len(repos_to_process)}")
    
    if to_process_repos:
        in_current = [repo for repo in repos_to_process if repo in to_process_repos]
        if in_current:
            logging.info(f"  Repositories marked for retry: {len(in_current)}")
    
    return repos_to_process

def setup_logging(task_id=""):
    """Configure logging."""
    project_root = Path(__file__).resolve().parent.parent
    log_filename = f"get_commit{('_' + task_id) if task_id else ''}.log"
    log_filepath = project_root / "data_preprocessing" / "log" / log_filename
    log_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Log file: {log_filepath}")

def load_repos_from_json(json_file):
    """Load the repository list from JSON."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Only the new format is supported here.
        if isinstance(data, dict) and "repos" in data and isinstance(data["repos"], dict):
            return list(data["repos"].keys())
        else:
            logging.error(f"Unrecognized repository file format or missing 'repos' key: {json_file}")
            return []
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load or parse repository file {json_file}: {e}")
        return []

def append_repo_results(repo_name, commit_list, output_file, keywords=None):
    """Append one repository's commit results to the output file."""
    if not commit_list:
        return
        
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use a file lock to make appends thread-safe.
    file_lock = _get_status_file_lock(str(output_path.resolve()))
    
    max_retries = 10
    for attempt in range(max_retries):
        try:
            with file_lock:
                # Load existing data.
                existing_data = {}
                if output_path.exists():
                    try:
                        with open(output_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        # Support both the new and legacy layouts.
                        if isinstance(data, dict) and "commits" in data:
                            existing_data = data.get("commits", {})
                        elif isinstance(data, dict):
                            existing_data = data
                        else:
                            existing_data = {}
                    except (json.JSONDecodeError, FileNotFoundError):
                        existing_data = {}
                
                # Add the new repository data.
                existing_data[repo_name] = commit_list
                
                # Recompute summary counts.
                total_repos_in_file = len(existing_data)
                total_commits_in_file = sum(len(commits) for commits in existing_data.values())
                
                # Write the normalized output layout.
                final_data = {
                    "summary": {
                        "total_repos": total_repos_in_file,
                        "total_commits": total_commits_in_file,
                        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "data_collection_cutoff": config.DATA_COLLECTION_CUTOFF,
                        "data_collection_cutoff_date": config.DATA_COLLECTION_CUTOFF_DATE
                    },
                    "commits": dict(sorted(existing_data.items()))
                }
                
                # Preserve the keywords in the summary when provided.
                if keywords:
                    final_data["summary"]["keywords"] = keywords
                
                # Write to a temp file and replace atomically.
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    encoding='utf-8', 
                    dir=output_path.parent, 
                    delete=False,
                    suffix='.tmp'
                ) as tmp_file:
                    json.dump(final_data, tmp_file, indent=2, ensure_ascii=False)
                    tmp_file.flush()
                    
                    shutil.move(tmp_file.name, output_path)
         
            logging.info(f"Saved {len(commit_list)} commits for repository {repo_name} ({total_repos_in_file} repos / {total_commits_in_file} commits in file).")
            return
            
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Write failed after the maximum number of retries: {e}")
                raise
            else:
                logging.warning(f"Write attempt {attempt + 1} failed; retrying: {e}")
                time.sleep(0.1)

def update_status_file(repo_name, status, status_file, total_repos_count=None):
    """Update the status file incrementally."""
    status_path = Path(status_file)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    file_lock = _get_status_file_lock(str(status_path.resolve()))
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with file_lock:
                # Load the existing state while holding the file lock.
                existing_status = {}
                data = {}
                if status_path.exists():
                    try:
                        with open(status_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        # Pull repo_details from the summarized layout when present.
                        if isinstance(data, dict) and "repo_details" in data:
                            existing_status = data["repo_details"]
                        elif isinstance(data, dict):
                            existing_status = data
                        else:
                            existing_status = {}
                    except (json.JSONDecodeError, FileNotFoundError):
                        existing_status = {}
                
                # Merge updates instead of overwriting the full repository status.
                previous = existing_status.get(repo_name)
                
                # A fresh "started" marker resets the previous state.
                is_fresh_start = 'started' in status and len(status) == 1

                if is_fresh_start or not isinstance(previous, dict):
                    merged = status
                else:
                    merged = {**previous, **status}

                merged["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                existing_status[repo_name] = merged
                
                # Recompute summary statistics.
                valid_statuses = {k: v for k, v in existing_status.items() if isinstance(v, dict)}
                
                successful_repos = sum(1 for s in valid_statuses.values() if s.get("success", False))
                failed_repos = sum(1 for s in valid_statuses.values() if s.get("started", False) and not s.get("success", False))
                
                # Resolve the total repository count.
                if total_repos_count is not None:
                    final_total_repos = total_repos_count
                elif isinstance(data, dict) and "total_repos" in data:
                    final_total_repos = data["total_repos"]
                else:
                    final_total_repos = len(valid_statuses)
                
                summary = {
                    "total_repos": final_total_repos,
                    "completed_repos": successful_repos + failed_repos,
                    "successful_repos": successful_repos,
                    "failed_repos": failed_repos,
                    "total_api_commits": sum(s.get("api_count", 0) for s in valid_statuses.values()),
                    "total_matched_commits": sum(s.get("matched_count", 0) for s in valid_statuses.values()),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "repo_details": valid_statuses
                }
                
                # Write atomically while still holding the file lock.
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    encoding='utf-8', 
                    dir=status_path.parent, 
                    delete=False,
                    suffix='.tmp'
                ) as tmp_file:
                    json.dump(summary, tmp_file, indent=4, ensure_ascii=False)
                    tmp_file.flush()
                    shutil.move(tmp_file.name, status_path)
                
                logging.debug(f"Updated status for {repo_name} ({summary['successful_repos']}/{summary['total_repos']} successful).")
            return
            
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to write the status file: {e}")
                raise
            else:
                logging.warning(f"Status write attempt {attempt + 1} failed; retrying: {e}")
                time.sleep(0.1)

def matches_keywords(text, keywords):
    """Return True if the text contains any keyword, case-insensitively."""
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)

def fetch_commit_details(commit, repo_name, keywords, max_retries=3):
    """Fetch commit details and keep only keyword-matching single-parent commits."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Keep only normal commits with exactly one parent.
            parent_count = len(commit.parents) if commit.parents is not None else 0
            if parent_count != 1:
                return None, None

            commit_message = commit.commit.message or ""
            
            if not matches_keywords(commit_message, keywords):
                return None, None
            
            parent_sha = commit.parents[0].sha
            
            return {
                "commit_sha": commit.sha,
                "commit_url": commit.html_url,
                "commit_message": commit_message.strip(),
                "parent_sha": parent_sha,
                "author": commit.commit.author.name if commit.commit.author else None,
                "date": commit.commit.committer.date.isoformat() if commit.commit.committer and commit.commit.committer.date else None,
            }, None
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                logging.debug(f"Commit {commit.sha[:8]} processing failed (attempt {attempt + 1}/{max_retries}); retrying...")
            else:
                return None, f"Failed to process commit {commit.sha[:8]} after {max_retries} retries: {e}"
    
    return None, f"Failed to process commit {commit.sha[:8]}: {last_error}"

def search_fix_bug_commits(repo_name, token_rotator, sleep_time, output_file, status_file, keywords, retry_count=0, max_retries=10, total_repos_count=None, keywords_for_save=None):
    """Search one repository for commits whose messages match the keywords."""
    
    logging.info(f"Searching repository: {repo_name}")
    
    token_index = 0
    
    try:
        time.sleep(sleep_time)
        logging.debug(f"API rate-limit guard: waited {sleep_time:.2f}s before searching {repo_name}.")
        
        if isinstance(token_rotator, TokenRotator):
            github_client, token_index = token_rotator.get_client()
        else:
            github_client = token_rotator
            token_index = 0
        
        update_status_file(repo_name, {"started": True}, status_file, total_repos_count)

        repo = github_client.get_repo(repo_name)
        
        # Prefer main, then fall back to master.
        target_branch = None
        try:
            repo.get_branch("main")
            target_branch = "main"
            logging.info(f"Found 'main' branch in {repo_name}.")
        except GithubException:
            try:
                repo.get_branch("master")
                target_branch = "master"
                logging.info(f"Found 'master' branch in {repo_name}.")
            except GithubException:
                logging.warning(f"No 'main' or 'master' branch found in {repo_name}; skipping repository.")
                status = {"success": True, "complete": True, "message": "No main or master branch"}
                update_status_file(repo_name, status, status_file, total_repos_count)
                return repo_name, [], True, status

        logging.info(f"Fetching commits from branch {target_branch} in {repo_name}...")
        
        until_dt = datetime.fromisoformat(config.DATA_COLLECTION_CUTOFF.replace('Z', '+00:00'))
        commits = repo.get_commits(sha=target_branch, until=until_dt)
        
        try:
            total_count = commits.totalCount
        except Exception:
            logging.warning(f"Could not read the commit count directly for {repo_name}; materializing commits first.")
            commits = list(commits)
            total_count = len(commits)

        logging.info(f"Repository {repo_name} has {total_count} commits; starting keyword matching.")
        
        if total_count == 0:
            logging.info(f"No commits found in repository {repo_name}.")
            status = {"api_count": 0, "processed_count": 0, "success": True, "complete": True}
            update_status_file(repo_name, status, status_file, total_repos_count)
            return repo_name, [], True, status
            
        max_workers = min(10, max(2, total_count // 10))
        logging.debug(f"Using {max_workers} worker threads to process commits.")
        
        commit_list = []
        processed_count = 0
        matched_count = 0
        error_count = 0
        failed_commits = []
        
        # Materialize the list to avoid PaginatedList race conditions.
        materialized_commits = list(commits)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_commit = {
                executor.submit(fetch_commit_details, commit, repo_name, keywords): commit 
                for commit in materialized_commits
            }
            
            for future in as_completed(future_to_commit):
                commit_data, error = future.result()
                
                if commit_data:
                    date_str = commit_data.get("date")
                    if not (date_str and date_str[:10] > config.DATA_COLLECTION_CUTOFF_DATE):
                        commit_list.append(commit_data)
                        matched_count += 1
                elif error:
                    error_count += 1
                    failed_commits.append(future_to_commit[future])
                    logging.warning(error)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logging.info(f"  Processed {processed_count}/{total_count} commits (matched={matched_count}, errors={error_count})...")
        
        if failed_commits:
            for attempt in range(3):
                if not failed_commits:
                    break
                logging.info(f"Retrying failed commits with backoff: attempt {attempt + 1}/3, count={len(failed_commits)}")
                time.sleep(2 * (attempt + 1))
                with ThreadPoolExecutor(max_workers=min(5, len(failed_commits))) as executor:
                    ftc2 = {
                        executor.submit(fetch_commit_details, c, repo_name, keywords): c
                        for c in failed_commits
                    }
                    new_failed = []
                    for future in as_completed(ftc2):
                        commit_data, error = future.result()
                        if commit_data:
                            date_str = commit_data.get("date")
                            if not (date_str and date_str[:10] > config.DATA_COLLECTION_CUTOFF_DATE):
                                commit_list.append(commit_data)
                                matched_count += 1
                        elif error:
                            new_failed.append(ftc2[future])
                            logging.warning(f"Retry failed: {error}")
                    failed_commits = new_failed
            error_count = len(failed_commits)
        
        if error_count > 0:
            logging.warning(f"{error_count} commits failed to process.")
        
        logging.info(f"Repository {repo_name} complete: {total_count} commits scanned, {matched_count} matched.")
        
        status = {
            "api_count": total_count,
            "processed_count": processed_count,
            "matched_count": matched_count,
            "success": error_count == 0 and processed_count == total_count,
            "complete": processed_count == total_count,
            "error_count": error_count
        }

        if not status["success"]:
            status["success"] = False
            status["complete"] = False
            status["error"] = "Incomplete commit fetch after targeted retries"
            update_status_file(repo_name, status, status_file, total_repos_count)
            return repo_name, [], False, status

        if commit_list:
            append_repo_results(repo_name, commit_list, output_file, keywords=keywords_for_save)
        update_status_file(repo_name, status, status_file, total_repos_count)
        
        return repo_name, commit_list, status["success"], status
        
    except RateLimitExceededException as e:
        logging.warning(f"Token hit the rate limit: {e}")
        if isinstance(token_rotator, TokenRotator):
            token_rotator.report_error(token_index)
            if len(token_rotator.tokens) > 1:
                logging.warning("Switching to the next token and retrying...")
                token_rotator.rotate(reason="RateLimit")
                return search_fix_bug_commits(repo_name, token_rotator, sleep_time, output_file, status_file, keywords, retry_count, max_retries, total_repos_count, keywords_for_save)
        
        logging.warning("Retrying after a 60-second wait...")
        time.sleep(60)
        return search_fix_bug_commits(repo_name, token_rotator, sleep_time, output_file, status_file, keywords, retry_count, max_retries, total_repos_count, keywords_for_save)
        
    except GithubException as e:
        if e.status == 404:
            logging.debug(f"Repository {repo_name} does not exist or is not accessible; skipping.")
            status = {"api_count": 0, "processed_count": 0, "success": True, "complete": True}
            update_status_file(repo_name, status, status_file, total_repos_count)
            return repo_name, [], True, status
        elif e.status == 403:
            logging.warning(f"Token hit a 403 error for repository {repo_name}: {e.data}")
            
            if isinstance(token_rotator, TokenRotator):
                token_rotator.report_error(token_index)
                
                if retry_count < max_retries and len(token_rotator.tokens) > 1:
                    logging.warning(f"Switching to the next token and retrying (attempt {retry_count + 1}/{max_retries})...")
                    token_rotator.rotate(reason="403 Forbidden")
                    time.sleep(2)
                    return search_fix_bug_commits(repo_name, token_rotator, sleep_time, output_file, status_file, keywords, retry_count + 1, max_retries, total_repos_count, keywords_for_save)
            
            logging.error(f"Repository {repo_name} failed with 403 Forbidden after {retry_count} retries.")
            status = {"api_count": 0, "processed_count": 0, "success": False, "complete": False, "error": "403 Forbidden"}
            update_status_file(repo_name, status, status_file, total_repos_count)
            return repo_name, [], False, status
        else:
            logging.error(f"GitHub API error while processing repository {repo_name}: {e.status} {e.data}")
            status = {"api_count": 0, "processed_count": 0, "success": False, "complete": False, "error": f"{e.status}"}
            update_status_file(repo_name, status, status_file, total_repos_count)
            return repo_name, [], False, status
    except Exception as e:
        logging.error(f"Unexpected error while processing repository {repo_name}: {e}")
        status = {
            "api_count": 0,
            "processed_count": 0,
            "success": False,
            "complete": False,
            "error": str(e)
        }
        update_status_file(repo_name, status, status_file, total_repos_count)
        return repo_name, [], False, status


# ============================================================================
# Commit deduplication
# ============================================================================

def get_repo_stars(repo_full_name, repos_data):
    """Return the star count for one repository from repos.json data."""
    # Support both the new and legacy repos.json layouts.
    if isinstance(repos_data, dict):
        if 'repos' in repos_data and isinstance(repos_data['repos'], dict):
            repo_info = repos_data['repos'].get(repo_full_name, {})
        else:
            repo_info = repos_data.get(repo_full_name, {})
        
        return repo_info.get('stars', 0)
    return 0


def deduplicate_commits(commits_file, repos_file, output_file, backup=True):
    """Deduplicate commits and keep the repository with the highest star count."""
    logging.info(f"Reading {commits_file}")
    with open(commits_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Support both the new and legacy output layouts.
    if isinstance(data, dict) and "commits" in data:
        commits_data = data.get("commits", {})
        original_summary = data.get("summary", {})
    elif isinstance(data, dict):
        commits_data = data
        original_summary = {}
    else:
        commits_data = {}
        original_summary = {}
    
    original_repo_count = len(commits_data)
    original_commit_count = sum(len(commits) for commits in commits_data.values())
    logging.info(f"Original data: {original_repo_count} repositories, {original_commit_count} commits")
    
    # Read repos.json so the deduplicator can compare star counts.
    repos_data = {}
    if repos_file.exists():
        logging.info(f"Reading {repos_file}")
        with open(repos_file, 'r', encoding='utf-8') as f:
            repos_data = json.load(f)
    else:
        logging.warning(f"{repos_file} does not exist; star data will not be available.")
    
    logging.info("Grouping commits by commit_sha...")
    commit_sha_to_repos = defaultdict(list)
    
    for repo, commits in commits_data.items():
        repo_stars = get_repo_stars(repo, repos_data)
        
        for commit in commits:
            sha = commit.get('commit_sha')
            parent_sha = commit.get('parent_sha', '')
            
            if sha:
                commit_sha_to_repos[sha].append({
                    'repository': repo,
                    'stars': repo_stars,
                    'commit': commit,
                    'parent_sha': parent_sha
                })
    
    logging.info(f"Found {len(commit_sha_to_repos)} unique commit_sha values.")
    
    logging.info("Starting deduplication...")
    deduplicated_data = {}
    removed_repos = set()
    removed_commits_count = 0
    
    for sha, repo_commits in commit_sha_to_repos.items():
        if len(repo_commits) == 1:
            repo_commit = repo_commits[0]
            repo = repo_commit['repository']
            commit = repo_commit['commit']
            
            if repo not in deduplicated_data:
                deduplicated_data[repo] = []
            deduplicated_data[repo].append(commit)
        else:
            # For duplicates, keep the repository with the highest star count.
            sorted_repos = sorted(
                repo_commits, 
                key=lambda x: (-x['stars'], x['repository'])
            )
            
            selected = sorted_repos[0]
            repo = selected['repository']
            commit = selected['commit']
            
            if repo not in deduplicated_data:
                deduplicated_data[repo] = []
            deduplicated_data[repo].append(commit)
            
            for rc in sorted_repos[1:]:
                removed_repos.add(rc['repository'])
                removed_commits_count += 1
                
            stars_info = [f"{rc['repository']} ({rc['stars']}★)" for rc in sorted_repos]
            logging.debug(f"Commit {sha[:8]}: kept {selected['repository']} ({selected['stars']}★)")
    
    # Track repositories whose entire commit set was removed.
    all_original_repos = set(commits_data.keys())
    repos_with_commits = set(deduplicated_data.keys())
    completely_removed_repos = all_original_repos - repos_with_commits
    
    # Drop repositories that no longer contain any commits.
    deduplicated_data = {k: v for k, v in deduplicated_data.items() if v}
    
    new_repo_count = len(deduplicated_data)
    new_commit_count = sum(len(commits) for commits in deduplicated_data.values())
    
    logging.info(f"\n{'='*80}")
    logging.info("Deduplication summary")
    logging.info(f"{'='*80}")
    logging.info(f"Original: {original_repo_count} repositories, {original_commit_count} commits")
    logging.info(f"After deduplication: {new_repo_count} repositories, {new_commit_count} commits")
    logging.info(f"Removed duplicate commits: {removed_commits_count}")
    logging.info(f"Deduplication rate: {removed_commits_count / original_commit_count * 100:.2f}%")
    
    if completely_removed_repos:
        logging.info(f"\nRepositories removed entirely after deduplication ({len(completely_removed_repos)} total):")
        sorted_removed = sorted(completely_removed_repos)
        for repo in sorted_removed[:10]:
            logging.info(f"  - {repo}")
        if len(completely_removed_repos) > 10:
            logging.info(f"  ... and {len(completely_removed_repos) - 10} more")
        logging.info("\nNote: repos.json still keeps metadata for these repositories; only commits.json is pruned.")
    
    # Back up the original file before overwriting it.
    if backup and output_file.exists():
        backup_file = output_file.with_suffix('.json.backup')
        logging.info(f"\nBacking up the original file to {backup_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            backup_data = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(backup_data)
    
    logging.info(f"\nWriting deduplicated data to {output_file}")
    
    final_data = {
        "summary": {
            "total_repos": new_repo_count,
            "total_commits": new_commit_count,
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "data_collection_cutoff": config.DATA_COLLECTION_CUTOFF,
            "data_collection_cutoff_date": config.DATA_COLLECTION_CUTOFF_DATE,
            "deduplicated": True,
            "deduplication_stats": {
                "original_repos": original_repo_count,
                "original_commits": original_commit_count,
                "removed_commits": removed_commits_count,
                "deduplication_rate": f"{removed_commits_count / original_commit_count * 100:.2f}%"
            }
        },
        "commits": deduplicated_data
    }
    
    # Preserve the original keyword list when available.
    if original_summary.get("keywords"):
        final_data["summary"]["keywords"] = original_summary["keywords"]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    logging.info("Done.")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Search commits across repositories using message keywords.")
    parser.add_argument(
        '--framework',
        type=str,
        default=config.ACTIVE_FRAMEWORK,
        choices=list(config.QUANTUM_FRAMEWORKS.keys()),
        help=f"Quantum framework to collect. Options: {list(config.QUANTUM_FRAMEWORKS.keys())}. Default: {config.ACTIVE_FRAMEWORK}"
    )
    parser.add_argument(
        '--input_json',
        type=str,
        default=None,
        help="Path to the repository-list JSON file. Defaults to the framework-specific repos.json."
    )
    # Backward-compatible alias.
    parser.add_argument(
        '--repos_json',
        type=str,
        help="Alias for --input_json."
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help="Path to the output JSON file for commits. Defaults to the framework-specific commits.json."
    )
    parser.add_argument(
        '--keywords',
        type=str,
        default=",".join(DEFAULT_KEYWORDS),
        help='Comma-separated keywords to search for.'
    )
    # Optional repository slicing for partial runs.
    parser.add_argument(
        '--repo_range',
        type=str,
        help='Repository range to process, in the form "start,end" using 1-based indices.'
    )
    parser.add_argument(
        '--range',
        type=str,
        help='Alias for --repo_range.'
    )
    parser.add_argument(
        '--task_id',
        type=str,
        default="",
        help='Task identifier used to create separate log and status files.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Enable resume mode and skip repositories already marked as complete in the status file.'
    )
    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help='Disable resume mode and reprocess every repository.'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=40,
        help='Number of repositories to process concurrently. Default: 40.'
    )
    parser.add_argument(
        '--requests_per_token',
        type=int,
        default=100,
        help='Rotate to the next token after this many requests. Default: 100.'
    )
    args = parser.parse_args()

    # Normalize compatible aliases.
    if args.repos_json:
        args.input_json = args.repos_json
    if args.repo_range:
        args.range = args.repo_range

    # Resolve framework-specific defaults.
    framework = args.framework
    framework_config = config.get_framework_config(framework)
    
    if args.input_json is None:
        args.input_json = str(config.get_framework_repos_path(framework))
    
    if args.output_json is None:
        args.output_json = str(config.get_framework_commits_path(framework))

    ensure_get_commit_paths(framework, args.output_json)

    # Parse keywords.
    keywords = [k.strip() for k in args.keywords.split(',') if k.strip()]
    
    setup_logging(args.task_id)

    logging.info(f"=" * 60)
    logging.info(f"Starting commit collection for framework {framework_config['name']}.")
    logging.info(f"Input file: {args.input_json}")
    logging.info(f"Output file: {args.output_json}")
    logging.info(f"=" * 60)
    logging.info(f"Search keywords: {', '.join(keywords)}")

    # Detect GitHub tokens automatically.
    tokens, token_names = detect_github_tokens()
    
    if not tokens:
        logging.error("No GitHub token was found. Set environment variables such as GITHUB_TOKEN_1 and GITHUB_TOKEN_2.")
        return
    
    logging.info(f"Detected {len(tokens)} GitHub tokens: {', '.join(token_names)}")
    
    # Compute the inter-request sleep time from the token count.
    sleep_time = calculate_sleep_time(len(tokens))
    logging.info(f"Configured API request interval to {sleep_time:.2f}s based on {len(tokens)} tokens.")
    
    # Use the token rotator with PyGithub clients enabled.
    token_rotator = TokenRotator(tokens, token_names, requests_per_token=args.requests_per_token, use_pygithub=True)
    
    repo_list = load_repos_from_json(args.input_json)
    if not repo_list:
        logging.error("The input list contains no repositories. Exiting.")
        return
    
    logging.info(f"Loaded {len(repo_list)} repositories.")

    # Apply repository slicing when requested.
    if args.range:
        try:
            start, end = map(int, args.range.split(','))
            repo_list = repo_list[start-1:end]
            logging.info(f"Processing repository range {start}-{end}; {len(repo_list)} repositories selected.")
        except ValueError:
            logging.error("Invalid repository range format. Expected 'start,end'.")
            return
        
    logging.info(f"Will search commits in {len(repo_list)} repositories.")
    
    # Capture the fixed total before resume-mode filtering.
    total_repos_count_fixed = len(repo_list)
    
    # Report a rough API-only time estimate.
    estimated_time_minutes = (len(repo_list) * sleep_time) / 60
    logging.info(f"Estimated search time: {estimated_time_minutes:.1f} minutes (API time only, excluding commit processing).")
    
    # Resolve the framework-specific status file path.
    task_suffix = f"_{args.task_id}" if args.task_id else "_main"
    framework_data_dir = config.get_framework_data_dir(framework)
    status_dir = framework_data_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = str(status_dir / f"commits_status{task_suffix}.json")
    logging.info(f"Live status file: {status_file}")

    # Filter repositories based on existing status.
    if args.resume:
        logging.info("Resume mode is enabled; completed repositories will be skipped.")
        all_target_repos = list(repo_list)
        existing_map = _read_existing_status_map(status_file)
        if all_target_repos:
            logging.info("Initializing repository placeholders in the status file...")
            initialize_status_placeholders(all_target_repos, status_file, total_repos_count_fixed)
        repo_list = filter_repos_by_status(repo_list, status_file)
    else:
        logging.info("Resume mode is disabled; all repositories will be processed.")
        logging.info(f"Repositories to process: {len(repo_list)}")
        all_target_repos = list(repo_list)
        existing_map = _read_existing_status_map(status_file)
        if all_target_repos:
            logging.info("Initializing repository placeholders in the status file...")
            initialize_status_placeholders(all_target_repos, status_file, total_repos_count_fixed)

    if not repo_list:
        logging.info("All repositories are already complete; skipping the search step.")

    all_commit_data = {}
    status_data = {}
    
    # Run the search only when repositories remain.
    if repo_list:
        if args.concurrency > 1:
            logging.info(f"Processing repositories with {args.concurrency} threads.")
            progress_desc = f"Searching commits (task: {args.task_id})" if args.task_id else "Searching commits"
            
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                total_repos_count = total_repos_count_fixed
                future_to_repo = {
                    executor.submit(
                        search_fix_bug_commits,
                        repo_name, token_rotator, sleep_time, args.output_json, status_file, keywords, 0, 10, total_repos_count, keywords
                    ): repo_name
                    for repo_name in repo_list
                }
                
                with tqdm(total=len(repo_list), desc=progress_desc) as pbar:
                    for future in as_completed(future_to_repo):
                        repo_name, commit_list, success, status = future.result()
                        status_data[repo_name] = status
                        
                        if success and commit_list:
                            all_commit_data[repo_name] = commit_list
                        elif not success:
                            logging.error(f"Repository {repo_name} failed: {status.get('error', 'unknown error')}")
                        
                        pbar.update(1)
        else:
            logging.info("Processing repositories sequentially with a single thread.")
            progress_desc = f"Searching commits (task: {args.task_id})" if args.task_id else "Searching commits"
            with tqdm(total=len(repo_list), desc=progress_desc) as pbar:
                total_repos_count = total_repos_count_fixed
                for repo_name in repo_list:
                    repo_name, commit_list, success, status = search_fix_bug_commits(
                        repo_name, token_rotator, sleep_time, args.output_json, status_file, keywords, 0, 10, total_repos_count, keywords
                    )
                    status_data[repo_name] = status
                    
                    if success and commit_list:
                        all_commit_data[repo_name] = commit_list
                    elif not success:
                        logging.error(f"Repository {repo_name} failed: {status.get('error', 'unknown error')}")
                        
                    pbar.update(1)
                
    # Final summary.
    if all_commit_data:
        total_repos = len(all_commit_data)
        total_commits = sum(len(commits) for commits in all_commit_data.values())
        logging.info(f"Processing complete: {total_repos} repositories produced matching commits, for a total of {total_commits} commits.")
        logging.info(f"All results were saved incrementally to: {args.output_json}")
    else:
        logging.info("No matching commits were found.")
    
    logging.info(f"Final status saved to: {status_file}")
    
    # Report token usage statistics.
    if isinstance(token_rotator, TokenRotator):
        logging.info("=" * 60)
        logging.info("Token usage statistics:")
        stats = token_rotator.get_stats()
        for stat in stats:
            logging.info(f"  {stat['name']}: {stat['requests']} requests, {stat['errors']} errors")
        logging.info(f"  Total rotations: {token_rotator.total_rotations}")
        logging.info("=" * 60)
    
    # Deduplicate commits automatically after collection.
    logging.info("\n" + "=" * 60)
    logging.info("Starting commit deduplication...")
    logging.info("=" * 60)
    try:
        commits_file = Path(args.output_json)
        repos_file = config.RAW_DATA_DIR / "repos.json"
        
        deduplicate_commits(commits_file, repos_file, commits_file, backup=True)
        
        logging.info("Commit deduplication complete.")
    except Exception as e:
        logging.warning(f"Commit deduplication failed: {e}")
        import traceback
        logging.warning(traceback.format_exc())

if __name__ == "__main__":
    main()
