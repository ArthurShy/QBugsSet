import os
import json
import requests
import time
from datetime import datetime, timezone
from urllib.parse import urlparse, quote
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
import logging
import argparse
import sys
import shutil
import subprocess
import tempfile
from collections import defaultdict, Counter
import difflib
import threading
import gc
import psutil
import signal

# Import shared configuration from the project root.
try:
    import config
    from util.retry import RetryableRequest, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
    from util.token_manager import TokenRotator, detect_github_token_vars
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    import config
    from util.retry import RetryableRequest, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
    from util.token_manager import TokenRotator, detect_github_token_vars

def ensure_get_bug_paths(framework: str | None = None) -> None:
    """Create only the directories required by get_bug."""
    if hasattr(config, 'ensure_bug_download_structure'):
        config.ensure_bug_download_structure(framework)

GITHUB_API_URL = "https://api.github.com"

# Module logger.
logger = logging.getLogger(__name__)

class FrameworkDetectionError(Exception):
    """Raised when framework import detection fails after retries."""

# Backward-compatible alias.
QiskitDetectionError = FrameworkDetectionError

# Active framework, set in main().
CURRENT_FRAMEWORK = "qiskit"

# Global token rotator reference used by the rate-limit handler.
_global_token_rotator = None

class FileDownloadError(Exception):
    """Raised when a required file download fails."""
    pass

# Status-file write throttling, overridable via environment variables.
STATUS_SAVE_MIN_INTERVAL_SEC = float(os.getenv('STATUS_SAVE_MIN_INTERVAL_SEC', '3.0'))
STATUS_SAVE_COALESCE = int(os.getenv('STATUS_SAVE_COALESCE', '20'))
_status_save_lock = threading.Lock()
_last_status_save_ts = {}  # task_id -> last monotonic ts
_pending_save_counts = {}  # task_id -> pending change count

def get_memory_usage_mb():
    """Return the current process memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024
    except Exception:
        return 0

def recompute_summary(status_data):
    """Recompute summary fields from commit_details."""
    commit_details = status_data.get("commit_details", {})
    summary = status_data.get("summary", {})

    successful_commits = 0
    failed_commits = 0
    total_buggy_files = 0
    total_fixed_files = 0
    full_projects_downloaded = 0
    downloaded_commits = 0
    total_diff_files = 0
    total_diff_files_discarded = 0

    for _, c in commit_details.items():
        status = c.get("status", "processing")
        if status == "completed":
            successful_commits += 1
        elif status == "failed":
            failed_commits += 1

        buggy_count = int(c.get("buggy_files_downloaded", 0) or 0)
        fixed_count = int(c.get("fixed_files_downloaded", 0) or 0)
        diff_count = int(c.get("diff_files_saved", 0) or 0)
        discarded_count = int(c.get("diff_files_discarded", 0) or 0)
        total_buggy_files += buggy_count
        total_fixed_files += fixed_count
        total_diff_files += diff_count
        total_diff_files_discarded += discarded_count

        has_valid_bug = buggy_count > 0
        if has_valid_bug and c.get("full_project_downloaded", False):
            full_projects_downloaded += 1
        if has_valid_bug:
            downloaded_commits += 1

    # Commit-centric summary fields.
    summary.clear()
    planned_commits = status_data.get("task_info", {}).get("total_commits_planned")
    summary["total_commits"] = int(planned_commits) if isinstance(planned_commits, int) else len(commit_details)
    summary["successful_commits"] = successful_commits
    summary["failed_commits"] = failed_commits
    summary["total_buggy_files"] = total_buggy_files
    summary["total_fixed_files"] = total_fixed_files
    summary["total_diff_files"] = total_diff_files
    summary["total_diff_files_discarded"] = total_diff_files_discarded
    summary["full_projects_downloaded"] = full_projects_downloaded
    summary["downloaded_commits"] = downloaded_commits

    status_data["summary"] = summary

# normalize_status_data is retired; migration happens in load_status_file().

def get_unified_paths():
    """Return the shared path configuration."""
    return {
        'data_root': config.DATA_ROOT,
        'source_code_dir': config.SOURCE_CODE_DIR,
        'raw_data_dir': config.RAW_DATA_DIR,
        'extracted_dir': config.EXTRACTED_DIR,
        'analyzed_dir': config.ANALYZED_DIR,
        'datasets_dir': config.DATASETS_DIR,
        'log_dir': config.LOG_DIR,
        # Backward-compatible aliases.
        'metadata_dir': config.RAW_DATA_DIR,
        'analysis_results_dir': config.ANALYZED_DIR,
    }

def get_commit_source_path(owner, repo, commit_sha, file_type="buggy"):
    """Return the per-commit source path for parent/commit/diffs data."""
    global CURRENT_FRAMEWORK
    source_dir = config.get_framework_source_dir(CURRENT_FRAMEWORK)
    return source_dir / owner / repo / commit_sha / file_type

def get_full_project_path(owner, repo, commit_sha, version_hash):
    """Return the full-project checkout path for one commit version."""
    return config.SOURCE_CODE_DIR / owner / repo / commit_sha / version_hash

def configure_logging(task_id=""):
    """Configure logging for this task."""
    paths = get_unified_paths()
    log_dir = paths['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if task_id:
        log_file_path = log_dir / f"get_bug_{task_id}.log"
    else:
        log_file_path = log_dir / "get_bug.log"
    
    # File handler keeps the full debug log.
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console output stays quiet except for warnings and errors.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler.setLevel(logging.WARNING)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Log file: {log_file_path}")

def get_status_file_path(task_id="main"):
    """Return the status file path for the active framework."""
    global CURRENT_FRAMEWORK
    framework_data_dir = config.get_framework_data_dir(CURRENT_FRAMEWORK)
    status_dir = framework_data_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_filename = f"bug_file_download_status_{task_id}.json"
    return status_dir / status_filename

def load_status_file(task_id="main"):
    """Load an existing status file."""
    status_file = get_status_file_path(task_id)
    if status_file.exists():
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Normalize legacy task_info fields into the commit-centric layout.
                ti = data.get("task_info", {})
                if "total_prs" in ti and "total_commits_planned" not in ti:
                    ti["total_commits_planned"] = ti.get("total_prs")
                if "pr_range" in ti and "commit_range" not in ti:
                    ti["commit_range"] = ti.get("pr_range")
                ti.pop("total_prs", None)
                ti.pop("pr_range", None)
                data["task_info"] = ti
                return data
        except Exception as e:
            logger.warning(f"Failed to load status file: {e}. A new status file will be created.")
    return None

def initialize_status_file(task_id, total_prs, pr_range=None):
    """Initialize a fresh status file."""
    status_data = {
        "task_info": {
            "task_id": task_id,
            # Commit-centric task information.
            "total_commits_planned": total_prs,
            "commit_range": pr_range,
            "start_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "status": "running"
        },
        "summary": {
            # Commit-centric summary fields.
            "total_commits": total_prs,
            "successful_commits": 0,
            "failed_commits": 0,
            "total_buggy_files": 0,
            "total_fixed_files": 0,
            "total_diff_files": 0,
            "full_projects_downloaded": 0,
            "downloaded_commits": 0,
            # Legacy PR fields retained as zeroed compatibility placeholders.
            "total_prs": 0,
            "successful_prs": 0,
            "failed_prs": 0,
            "downloaded_prs": 0
        },
        # Commit-only detail records.
        "commit_details": {}
    }
    
    save_status_file(task_id, status_data)
    return status_data

def _write_status_file_immediately(task_id, status_data):
    """Write the status file immediately using an atomic replace."""
    status_file = get_status_file_path(task_id)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_data["task_info"]["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=status_file.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        json.dump(status_data, tmp_file, ensure_ascii=False, indent=4)
        tmp_file.flush()
        shutil.move(tmp_file.name, status_file)
    logger.debug(f"Status file saved atomically: {status_file}")

def save_status_file(task_id, status_data):
    """Throttle status writes to reduce write frequency."""
    try:
        with _status_save_lock:
            now = time.monotonic()
            last_ts = _last_status_save_ts.get(task_id, 0.0)
            pend = _pending_save_counts.get(task_id, 0)
            pend += 1
            should_save = (now - last_ts) >= STATUS_SAVE_MIN_INTERVAL_SEC or pend >= STATUS_SAVE_COALESCE
            if not should_save:
                _pending_save_counts[task_id] = pend
                return
            # Reset the coalescing counters once a save is triggered.
            _pending_save_counts[task_id] = 0
            _last_status_save_ts[task_id] = now

        _write_status_file_immediately(task_id, status_data)
    except Exception as e:
        logger.error(f"Failed to save status file: {e}")

def flush_status_file(task_id, status_data):
    """Force an immediate status-file flush."""
    try:
        with _status_save_lock:
            _pending_save_counts[task_id] = 0
            _last_status_save_ts[task_id] = time.monotonic()
        _write_status_file_immediately(task_id, status_data)
    except Exception as e:
        logger.error(f"Failed to flush status file: {e}")

def ensure_pending_entries_for_tasks(status_data, tasks):
    """Pre-create pending status entries for the current commit task list."""
    commit_details = status_data.setdefault("commit_details", {})

    created_commit = 0
    reset_failed_commit = 0

    for task in tasks:
        owner = task.get("owner", "")
        repo = task.get("repo", "")
        commit_sha = task.get("commit_sha")
        commit_url = task.get("commit_url")
        if not commit_sha and task.get("commit_url"):
            try:
                _o, _r, _sha = parse_commit_url(task["commit_url"])  # type: ignore[name-defined]
                commit_sha = _sha
                # Backfill missing owner/repo metadata when possible.
                owner = owner or _o
                repo = repo or _r
            except Exception:
                commit_sha = None

        if not owner or not repo or not commit_sha:
            continue

        if not commit_url:
            commit_url = f"https://github.com/{owner}/{repo}/commit/{commit_sha}"

        commit_key = f"{owner}/{repo}#commit_{commit_sha[:8]}"

        if commit_key not in commit_details:
            commit_details[commit_key] = {
                "repository": f"{owner}/{repo}",
                "commit_url": commit_url,
                "commit_sha": commit_sha,
                "commit_message": "",
                "status": "pending",
                "buggy_files_downloaded": 0,
                "fixed_files_downloaded": 0,
                "diff_files_saved": 0,
                "full_project_downloaded": False,
            }
            created_commit += 1
        elif commit_details[commit_key].get("status") == "failed":
            commit_details[commit_key]["status"] = "pending"
            reset_failed_commit += 1
            logger.info(f"Reset failed commit to pending: {commit_key}")

    if any([created_commit, reset_failed_commit]):
        logger.info(
            f"Status update: created {created_commit} commits, reset {reset_failed_commit} commits."
        )
        recompute_summary(status_data)
        save_status_file(status_data.get("task_info", {}).get("task_id", "main"), status_data)

    # Keep only keys related to the current task set.
    current_keys = set()
    for task in tasks:
        owner = task.get("owner", "")
        repo = task.get("repo", "")
        commit_sha = task.get("commit_sha")
        if not commit_sha and task.get("commit_url"):
            try:
                _o, _r, _sha = parse_commit_url(task["commit_url"])  # type: ignore[name-defined]
                owner = owner or _o
                repo = repo or _r
                commit_sha = _sha
            except Exception:
                commit_sha = None
        if owner and repo and commit_sha:
            current_keys.add(f"{owner}/{repo}#commit_{commit_sha[:8]}")

    # Prune historical keys that are not part of the active task set.
    if current_keys:
        to_delete = [k for k in list(commit_details.keys()) if k not in current_keys]
        if to_delete:
            for k in to_delete:
                try:
                    del commit_details[k]
                except Exception:
                    pass
            logger.info(f"Pruned {len(to_delete)} historical commit entries.")
            recompute_summary(status_data)
            save_status_file(status_data.get("task_info", {}).get("task_id", "main"), status_data)



## PR helper functions were removed in commit-only mode.

def finalize_status_file(task_id, status_data):
    """Mark the status file as completed and log the final summary."""
    status_data["task_info"]["status"] = "completed"
    status_data["task_info"]["end_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Compute task duration.
    start_time = datetime.fromisoformat(status_data["task_info"]["start_time"].replace(" UTC", "+00:00"))
    end_time = datetime.fromisoformat(status_data["task_info"]["end_time"].replace(" UTC", "+00:00"))
    duration = end_time - start_time
    status_data["task_info"]["duration_seconds"] = int(duration.total_seconds())
    status_data["task_info"]["duration_readable"] = str(duration)
    
    # Force a final flush at task completion.
    flush_status_file(task_id, status_data)
    
    summary = status_data["summary"]
    logger.info("Final summary:")
    logger.info(f"  Total commits: {summary.get('total_commits', 0)}")
    logger.info(f"  Successful commits: {summary.get('successful_commits', 0)}")
    logger.info(f"  Failed commits: {summary.get('failed_commits', 0)}")
    logger.info(f"  Buggy files: {summary.get('total_buggy_files', 0)}")
    logger.info(f"  Fixed files: {summary.get('total_fixed_files', 0)}")
    logger.info(f"  Saved diff files: {summary.get('total_diff_files', 0)}")
    diff_discarded = summary.get('total_diff_files_discarded', 0)
    if diff_discarded > 0:
        logger.info(f"  Discarded empty diff files: {diff_discarded}")
    logger.info(f"  Full projects downloaded: {summary.get('full_projects_downloaded', 0)}")
    logger.info(f"  Commits with downloaded files: {summary.get('downloaded_commits', 0)}")
    logger.info(f"  Duration: {status_data['task_info'].get('duration_readable', '')}")

## PR status helpers were removed in commit-only mode.

def is_commit_completed(status_data, owner: str, repo: str, commit_sha: str):
    """Return True when a commit has already been marked completed."""
    commit_key = f"{owner}/{repo}#commit_{commit_sha[:8]}"
    commit_details = status_data.get("commit_details", {})
    
    if commit_key in commit_details:
        return commit_details[commit_key].get("status") == "completed"
    return False

def update_commit_status(task_id, status_data, commit_info, success=True, **kwargs):
    """Update the processing status for one commit."""
    owner = commit_info['owner']
    repo = commit_info['repo']
    commit_sha = commit_info['commit_sha']
    
    commit_key = f"{owner}/{repo}#commit_{commit_sha[:8]}"
    
    # Initialize the commit status record if it does not exist yet.
    if commit_key not in status_data.get("commit_details", {}):
        if "commit_details" not in status_data:
            status_data["commit_details"] = {}
        status_data["commit_details"][commit_key] = {
            "repository": f"{owner}/{repo}",
            "commit_url": commit_info.get("commit_url", f"https://github.com/{owner}/{repo}/commit/{commit_sha}"),
            "commit_sha": commit_sha,
            "commit_message": commit_info.get("message", ""),
            "status": "processing",
            "buggy_files_downloaded": 0,
            "fixed_files_downloaded": 0,
            "full_project_downloaded": False,
            "diff_files_saved": 0,
            "diff_files_discarded": 0
        }
    
    # Update the final state.
    commit_status = status_data["commit_details"][commit_key]
    status_override = kwargs.get("status")
    if status_override:
        commit_status["status"] = status_override
    else:
        commit_status["status"] = "completed" if success else "failed"
    
    # Update the core identifying fields when provided.
    if "commit_sha" in commit_info:
        commit_status["commit_sha"] = commit_info["commit_sha"]
    if "message" in commit_info and commit_info["message"]:
        commit_status["commit_message"] = commit_info["message"]
    
    # Update counters only when explicit values are provided.
    if "buggy_files_downloaded" in kwargs:
        commit_status["buggy_files_downloaded"] = kwargs["buggy_files_downloaded"]
    if "fixed_files_downloaded" in kwargs:
        commit_status["fixed_files_downloaded"] = kwargs["fixed_files_downloaded"]
    if "full_project_downloaded" in kwargs:
        commit_status["full_project_downloaded"] = bool(kwargs["full_project_downloaded"])
    if "diff_files_saved" in kwargs:
        commit_status["diff_files_saved"] = kwargs["diff_files_saved"]
    if "diff_files_discarded" in kwargs:
        commit_status["diff_files_discarded"] = kwargs["diff_files_discarded"]
    
    # Recompute the summary after every status change.
    recompute_summary(status_data)
    logger.debug(f"Commit status updated: {commit_key} -> {commit_status['status']}")

def contains_framework_import(file_path: Path, framework: str = "qiskit") -> bool:
    """Return True when a local Python file imports the target framework."""
    try:
        # Import here to avoid a circular dependency.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config import contains_framework_import as check_import
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return check_import(content, framework)
    except Exception as e:
        logger.error(f"Failed while checking file {file_path}: {e}")
        return False

# Backward-compatible alias.
def contains_qiskit_import(file_path: Path) -> bool:
    return contains_framework_import(file_path, "qiskit")

# filter_qiskit_files_after_checkout was removed.

# download_full_project was removed.

def _handle_github_rate_limit(headers: dict):
    """Handle GitHub API rate limits by rotating tokens or sleeping."""
    global _global_token_rotator
    
    if 'X-RateLimit-Remaining' in headers and int(headers.get('X-RateLimit-Remaining', 1)) == 0:
        # Prefer token rotation when multiple tokens are available.
        if _global_token_rotator is not None and len(_global_token_rotator.tokens) > 1:
            _global_token_rotator.rotate(reason="RateLimit")
            logger.info("Rate limit detected; switched to the next token.")
            time.sleep(1)
            return
        
        # With a single token, wait for reset.
        reset_timestamp = int(headers.get('X-RateLimit-Reset', 0))
        if reset_timestamp:
            reset_time = datetime.fromtimestamp(reset_timestamp, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            sleep_duration = max(0, (reset_time - now).total_seconds()) + 5
            sleep_duration = min(sleep_duration, 60)
            logger.warning(f"Rate limit reached with a single token. Sleeping for {sleep_duration:.2f}s.")
            time.sleep(sleep_duration)

def _get_current_headers() -> dict:
    """Return headers for the currently active token."""
    global _global_token_rotator
    if _global_token_rotator is not None:
        return _global_token_rotator.get_headers()
    # Fall back to the first available token.
    token_vars = detect_github_token_vars()
    if token_vars:
        token = os.getenv(token_vars[0], '')
        return {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {token}"
        }
    return {"Accept": "application/vnd.github.v3+json"}

# Global retrier for GitHub HTTP requests.
_github_retrier = RetryableRequest(
    max_retries=DEFAULT_MAX_RETRIES,
    timeout=DEFAULT_TIMEOUT,
    rate_limit_handler=_handle_github_rate_limit,
    headers_getter=_get_current_headers
)

def make_api_request(url: str, headers: dict, params: dict = None, max_retries: int = DEFAULT_MAX_RETRIES) -> requests.Response:
    """Issue an API request through the shared retry wrapper."""
    response = _github_retrier.get(url, headers=headers, params=params)
    if response is None:
        raise requests.exceptions.RequestException(f"Request failed: {url}")
    return response



def parse_commit_url(commit_url: str):
    """Parse a commit URL into (owner, repo, commit_sha)."""
    parsed = urlparse(commit_url)
    parts = parsed.path.strip('/').split('/')
    if len(parts) < 4 or parts[2] != 'commit':
        raise ValueError(f"Invalid commit URL format: {commit_url}")
    owner, repo, _, commit_sha = parts[:4]
    return owner, repo, commit_sha

def get_commit_details(owner: str, repo: str, commit_sha: str, token_var: str) -> dict:
    """Fetch detailed information for a commit through the GitHub API."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {os.getenv(token_var, '')}"
    }
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/commits/{commit_sha}"
    try:
        logger.debug(f"Fetching commit details: {owner}/{repo}@{commit_sha[:8]}")
        resp = make_api_request(url, headers=headers)
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch commit details ({commit_sha[:8]}): {e}")
        raise

def get_commit_files(commit_details: dict) -> list:
    """Extract the modified file list from commit details."""
    files = commit_details.get('files', [])
    logger.info(f"Commit {commit_details.get('sha', 'unknown')[:8]} contains {len(files)} modified files.")
    return files

def check_file_contains_framework(owner: str, repo: str, file_path: str, ref: str, token_var: str, framework: str = "qiskit") -> bool:
    """Check whether a remote file imports the target framework."""
    # Import config helpers here to avoid extra module coupling.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import get_import_patterns, get_framework_config
    
    import_patterns = get_import_patterns(framework)
    framework_name = get_framework_config(framework)["name"]
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {os.getenv(token_var, '')}"
    }
    # URL-encode the path so special characters are handled safely.
    encoded_path = quote(file_path, safe='/')
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{encoded_path}"
    # Retry all non-200 outcomes with exponential backoff.
    last_status_code = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            last_status_code = resp.status_code
            
            if resp.status_code != 200:
                logger.warning(f"Framework import check failed for {file_path} (attempt {attempt}/3), HTTP {resp.status_code}.")
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
                    continue
                else:
                    raise FrameworkDetectionError(f"{framework} detection failed for {owner}/{repo}/{ref}/{file_path}: HTTP {resp.status_code} after 3 retries")
            
            content = resp.content.decode('utf-8', errors='ignore')
            has_framework = any(pattern in content for pattern in import_patterns)
            logger.debug(
                f"File contains {framework_name} import: {file_path}" if has_framework else f"File does not contain {framework_name} import: {file_path}"
            )
            return has_framework
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Framework import check failed for {file_path} (attempt {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
            else:
                raise FrameworkDetectionError(f"{framework} detection failed for {owner}/{repo}/{ref}/{file_path}: {e}")
    
    return False


# Backward-compatible alias.
def check_file_contains_qiskit(owner: str, repo: str, file_path: str, ref: str, token_var: str) -> bool:
    """Backward-compatible alias for the qiskit-specific framework check."""
    return check_file_contains_framework(owner, repo, file_path, ref, token_var, "qiskit")

def download_file(owner: str, repo: str, file_path: str, ref: str, token_var: str, commit_info: dict, log_404_as_error: bool = True) -> bytes | None:
    """Download a file with retries and exponential backoff."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {os.getenv(token_var, '')}"
    }
    # URL-encode the path so special characters are handled safely.
    encoded_path = quote(file_path, safe='/')
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{encoded_path}"
    
    logger.debug(f"Downloading file: {file_path} @ {ref[:8]}")
    response = _github_retrier.get(url, headers=headers)
    
    if response is None:
        logger.error(f"Failed to download file: {file_path}")
        return None
    
    if response.status_code != 200:
        if log_404_as_error or response.status_code != 404:
            logger.warning(f"Download failed: {file_path} - HTTP {response.status_code}")
        return None
    
    return response.content

def download_source_file(owner: str, repo: str, full_filename: str, ref: str, token_var: str, commit_info: dict, file_type: str, log_404_as_error: bool = True):
    """Download a file while preserving its original repository-relative path."""
    commit_sha = commit_info['commit_sha']
    base_path = get_commit_source_path(owner, repo, commit_sha, file_type)
    
    original_file_path = Path(full_filename)
    final_path = base_path / original_file_path
    
    final_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Target path: {final_path}")
    
    # Skip the download if the file already exists.
    if final_path.exists():
        logger.debug(f"File already exists, skipping download: {final_path}")
        return final_path

    # Stream directly to disk to avoid keeping the full file in memory.
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {os.getenv(token_var, '')}"
    }
    # URL-encode the file path so special characters are handled safely.
    encoded_path = quote(full_filename, safe='/')
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{encoded_path}"
    max_attempts = 3
    success_written = False
    last_error = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"Streaming file download (attempt {attempt}/{max_attempts}): {full_filename} @ {ref[:8]}")
            with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
                if resp.status_code != 200:
                    error_msg = f"HTTP {resp.status_code}"
                    logger.warning(f"Download failed (attempt {attempt}/{max_attempts}): {full_filename} - {error_msg}")
                    last_error = error_msg
                    if attempt < max_attempts:
                        time.sleep(2 ** (attempt - 1))
                        continue
                    else:
                        break
                
                with open(final_path, 'wb') as f:
                    for chunk in resp.iter_content(64 * 1024):
                        if chunk:
                            f.write(chunk)
            
            logger.debug(f"Saved source file: {final_path}")
            success_written = True
            break
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"File download failed (attempt {attempt}/{max_attempts}) - {full_filename}: {e}")
            last_error = str(e)
            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))
            else:
                logger.error(f"File download failed after {max_attempts} attempts: {full_filename}")
                return None

    if not success_written:
        return None

    # Double-check that the file exists on disk.
    if not final_path.exists():
        logger.error(f"Downloaded file is missing on disk: {final_path}")
        return None

    return final_path

def cleanup_empty_parent_dirs(file_path: Path, stop_at: Path):
    """Recursively delete empty parent directories until stop_at is reached."""
    parent = file_path.parent
    
    # Compare using absolute paths.
    stop_at = stop_at.resolve()
    
    while parent.resolve() != stop_at and parent.exists():
        try:
            if not any(parent.iterdir()):
                logger.debug(f"Removing empty directory: {parent}")
                parent.rmdir()
                parent = parent.parent
            else:
                logger.debug(f"Directory is not empty; stopping cleanup: {parent}")
                break
        except OSError as e:
            logger.debug(f"Could not remove directory {parent}: {e}")
            break
        except Exception as e:
            logger.warning(f"Directory cleanup failed for {parent}: {e}")
            break

def process_commit(commit_url: str, lib_name: str, token_var: str, framework_filter: bool = True) -> dict or None:
    """Process one commit, preserving source layout and detailed status."""
    commit_info = {
        "commit_url": commit_url, 
        "commit_message": None, 
        "files": [], 
        "repository": None, 
        "commit_sha": None
    }
    
    try:
        owner, repo, commit_sha = parse_commit_url(commit_url)
        repository = f"{owner}/{repo}"
        commit_info["repository"] = repository
        commit_info["commit_sha"] = commit_sha
        
        logger.info(f"Starting commit processing for {commit_sha[:8]} @ {repository}")
        

        # Fetch commit metadata.
        commit_details = get_commit_details(owner, repo, commit_sha, token_var)
        commit_info["commit_message"] = commit_details.get("commit", {}).get("message", "No Message")
        
        # Determine the parent commit used for comparison.
        parents = commit_details.get("parents", [])
        if len(parents) == 0:
            logger.warning(f"Commit {commit_sha[:8]} has no parent commit; skipping.")
            commit_info["success"] = False
            commit_info["message"] = "No parent commit - cannot compare"
            return commit_info
        elif len(parents) > 1:
            logger.warning(f"Commit {commit_sha[:8]} has multiple parent commits; skipping.")
            commit_info["success"] = False
            commit_info["message"] = "Multiple parent commits - cannot compare"
            return commit_info
        
        parent_sha = parents[0]["sha"]
        
        # Attach version metadata.
        commit_info["parent_sha"] = parent_sha
        commit_info["commit_sha"] = commit_sha
        commit_info["version_hash"] = commit_sha[:8] if len(commit_sha) >= 8 else commit_sha
        commit_info["owner"] = owner
        commit_info["repo"] = repo
        
        logger.debug(f"Commit message: {commit_info['commit_message']}")
        logger.debug(f"🔗 Parent: {parent_sha[:8]}, Commit: {commit_sha[:8]}")
        
        # Load the changed file list.
        changed_files = get_commit_files(commit_details)

        # Keep modified Python files only, using the minimal fields needed downstream.
        py_files = [
            {"filename": f['filename'], "status": f['status']} for f in changed_files
            if f.get('status') == 'modified' and f.get('filename', '').endswith('.py')
        ]
        
        if not py_files:
            logger.info(f"No matching .py files found in commit {commit_sha[:8]}.")
            commit_info["files"] = []
            commit_info["success"] = True
            commit_info["framework_filtered"] = False
            commit_info["message"] = "No Python files found - skipped"
            return commit_info

        logger.info(f"Found {len(py_files)} candidate Python files in commit {commit_sha[:8]}.")

        # Process files one by one.
        expected_files = 0
        fully_downloaded_files = 0
        diff_saved_count = 0
        diff_generated_count = 0
        diff_empty_count = 0
        framework_matched_files = []
        
        # Optionally filter files by framework imports before downloading.
        global CURRENT_FRAMEWORK
        framework_name = config.get_framework_config(CURRENT_FRAMEWORK)["name"]
        if framework_filter:
            logger.info(f"Checking {len(py_files)} files in parallel for {framework_name} imports...")
            
            with ThreadPoolExecutor(max_workers=min(10, len(py_files))) as framework_executor:
                def check_framework_for_file(file_data):
                    full_filename = file_data['filename']
                    logger.debug(f"Checking file: {full_filename}")
                    
                    # Check the parent and commit versions.
                    try:
                        has_framework_buggy = check_file_contains_framework(owner, repo, full_filename, parent_sha, token_var, CURRENT_FRAMEWORK)
                    except FrameworkDetectionError as e:
                        logger.error(str(e))
                        raise
                    
                    # Only the parent version determines whether the file stays in scope.
                    if has_framework_buggy:
                        try:
                            has_framework_fixed = check_file_contains_framework(owner, repo, full_filename, commit_sha, token_var, CURRENT_FRAMEWORK)
                        except FrameworkDetectionError as e:
                            logger.error(f"Commit-version framework detection failed: {str(e)}")
                            raise
                        
                        logger.debug(f"Parent version contains {framework_name}; keeping file: {full_filename}")
                        return {
                            'file_data': file_data,
                            'has_framework_buggy': True,
                            'has_framework_fixed': has_framework_fixed
                        }
                    else:
                        logger.debug(f"Parent version does not contain {framework_name}; skipping: {full_filename}")
                        return None
                
                framework_futures = {framework_executor.submit(check_framework_for_file, file_data): file_data for file_data in py_files}
                
                framework_files = []
                framework_detection_failed = False
                framework_detection_error = None
                
                for future in as_completed(framework_futures):
                    try:
                        result = future.result()
                        if result:
                            framework_files.append(result)
                    except FrameworkDetectionError as e:
                        logger.error(f"{framework_name} detection failed; commit will be marked as failed: {e}")
                        framework_detection_failed = True
                        framework_detection_error = str(e)
                        break
                    except Exception as e:
                        file_data = framework_futures[future]
                        logger.error(f"Failed while checking {framework_name} imports in {file_data['filename']}: {e}")
                
                if framework_detection_failed:
                    logger.warning(f"Commit {commit_sha[:8]} stopped because {framework_name} detection failed.")
                    commit_info["files"] = []
                    commit_info["success"] = False
                    commit_info["error"] = f"{framework_name} detection failed: {framework_detection_error}"
                    commit_info["framework_filtered"] = True
                    return commit_info
            framework_matched_files = framework_files
        else:
            logger.info(f"Processing {len(py_files)} Python files with {framework_name} filtering disabled...")
            framework_matched_files = []
            for file_data in py_files:
                framework_matched_files.append({
                    'file_data': file_data,
                    'has_framework_buggy': True,
                    'has_framework_fixed': True
                })
        
        if not framework_matched_files:
            if framework_filter:
                logger.info(f"No files with {framework_name} imports were found in commit {commit_sha[:8]}.")
            else:
                logger.info(f"No eligible Python files were found in commit {commit_sha[:8]}.")
            
            message = f"No files with {framework_name} import found - skipped" if framework_filter else "No Python files found - skipped"
            commit_info["files"] = []
            commit_info["success"] = True
            commit_info["framework_filtered"] = framework_filter
            commit_info["message"] = message
            return commit_info
        
        if framework_filter:
            logger.info(f"Files containing {framework_name}: {len(framework_matched_files)}/{len(py_files)}")
        else:
            logger.info(f"Processing {len(framework_matched_files)} Python files.")

        # Download and diff files in parallel.
        def _process_single_file(file_info):
            file_data = file_info['file_data']
            full_filename = file_data['filename']
            has_framework_buggy = file_info.get('has_framework_buggy', True)
            has_framework_fixed = file_info.get('has_framework_fixed', True)

            buggy_path = None
            fixed_path = None

            # Download the parent version first; 404 means the file may be newly added.
            buggy_path = download_source_file(
                owner, repo, full_filename, parent_sha, token_var,
                commit_info, "parent", log_404_as_error=False
            )

            # Download the commit version.
            fixed_path = download_source_file(
                owner, repo, full_filename, commit_sha, token_var,
                commit_info, "commit"
            )

            # Newly added files have no parent version and are skipped.
            if not buggy_path and fixed_path:
                logger.warning(f"Skipping newly added file with no base version: {full_filename}")
                return None

            # Any failed download should fail the whole commit.
            if not buggy_path and not fixed_path:
                error_msg = f"Both versions failed to download: {full_filename}"
                logger.error(error_msg)
                raise FileDownloadError(error_msg)
            elif not buggy_path:
                error_msg = f"Parent version failed to download: {full_filename}"
                logger.error(error_msg)
                raise FileDownloadError(error_msg)
            elif not fixed_path:
                error_msg = f"Commit version failed to download: {full_filename}"
                logger.error(error_msg)
                raise FileDownloadError(error_msg)
            
            expected_inc = 1
            fully_inc = 1

            # Generate and save the local diff once both files exist.
            diff_ok = False
            diff_empty_flag = False
            if fully_inc == 1:
                try:
                    project_root = Path(__file__).resolve().parent.parent
                    buggy_abs_path = project_root / buggy_path
                    fixed_abs_path = project_root / fixed_path
                    
                    if not buggy_abs_path.exists():
                        logger.error(f"Parent file is missing; cannot generate diff: {buggy_abs_path}")
                        raise FileNotFoundError(f"Buggy file not found: {buggy_abs_path}")
                    if not fixed_abs_path.exists():
                        logger.error(f"Commit file is missing; cannot generate diff: {fixed_abs_path}")
                        raise FileNotFoundError(f"Fixed file not found: {fixed_abs_path}")
                    
                    diff_ok, diff_empty_flag = generate_and_save_diff(owner, repo, commit_sha, full_filename, buggy_abs_path, fixed_abs_path)
                except Exception as diff_e:
                    logger.warning(f"Diff generation failed for {full_filename}: {diff_e}")

            # Record file metadata without calling resolve() on paths that may have been deleted.
            project_root = Path(__file__).resolve().parent.parent
            buggy_rel_path = None
            fixed_rel_path = None
            if buggy_path:
                try:
                    buggy_abs = Path(buggy_path) if Path(buggy_path).is_absolute() else (Path.cwd() / buggy_path)
                    buggy_rel_path = str(buggy_abs.relative_to(project_root))
                except (ValueError, OSError):
                    buggy_rel_path = str(buggy_path)
            if fixed_path:
                try:
                    fixed_abs = Path(fixed_path) if Path(fixed_path).is_absolute() else (Path.cwd() / fixed_path)
                    fixed_rel_path = str(fixed_abs.relative_to(project_root))
                except (ValueError, OSError):
                    fixed_rel_path = str(fixed_path)

            file_name = Path(buggy_path if buggy_path else fixed_path if fixed_path else full_filename).name
            record = {
                "file_name": file_name,
                "relative_file_path": full_filename,
                "buggy": buggy_rel_path,
                "fixed": fixed_rel_path,
                "file_type": "python" if full_filename.endswith('.py') else "other",
                "download_status": {
                    "buggy_success": buggy_path is not None,
                    "fixed_success": fixed_path is not None
                },
                "framework_status": {
                    "has_framework_buggy": has_framework_buggy,
                    "has_framework_fixed": has_framework_fixed
                },
                "diff_saved": diff_ok and not diff_empty_flag,
                "diff_empty": diff_ok and diff_empty_flag,
                "diff_generated": diff_ok
            }

            if diff_empty_flag:
                logger.info(f"Empty diff detected; deleting file pair: {full_filename}")
                record["discarded_due_to_empty_diff"] = True
                record["buggy_deleted_path"] = buggy_rel_path
                record["fixed_deleted_path"] = fixed_rel_path
                record["buggy"] = None
                record["fixed"] = None
                # Reset download flags so discarded files do not count toward totals.
                record["download_status"]["buggy_success"] = False
                record["download_status"]["fixed_success"] = False

                # Delete the downloaded source files for empty diffs.
                project_root = Path(__file__).resolve().parent.parent
                for path_obj, label in ((buggy_path, "buggy"), (fixed_path, "fixed")):
                    if path_obj:
                        try:
                            abs_path = project_root / path_obj if not Path(path_obj).is_absolute() else Path(path_obj)
                            if abs_path.exists():
                                abs_path.unlink()
                                logger.debug(f"Deleted {label} file for empty diff: {abs_path}")
                        except FileNotFoundError:
                            pass
                        except Exception as cleanup_err:
                            logger.warning(f"Could not delete {label} file {path_obj}: {cleanup_err}")

                # Delete the empty diff file itself.
                try:
                    diff_base_path = get_commit_source_path(owner, repo, commit_sha, 'diffs')
                    diff_rel = Path(actual_filename)
                    diff_file_path = diff_base_path / diff_rel.parent / f"{diff_rel.name}.diff"
                    if diff_file_path.exists():
                        diff_file_path.unlink()
                        logger.debug(f"Deleted empty diff file: {diff_file_path}")
                    else:
                        logger.debug(f"Empty diff file does not exist (it may already be deleted): {diff_file_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Could not delete empty diff file {actual_filename}: {cleanup_err}")

                # Reset counters because this file was discarded.
                expected_inc = 0
                fully_inc = 0
                diff_ok = False

            if not (buggy_path is not None and fixed_path is not None) and not diff_empty_flag:
                logger.warning(
                    f"File processing was incomplete: {full_filename} (buggy={buggy_path is not None}, fixed={fixed_path is not None})"
                )

            return record, expected_inc, fully_inc, diff_ok, diff_empty_flag

        file_records = []
        download_error = None
        
        if framework_matched_files:
            download_workers = min(10, len(framework_matched_files))
            with ThreadPoolExecutor(max_workers=download_workers) as file_executor:
                future_to_file = {file_executor.submit(_process_single_file, f): f for f in framework_matched_files}
                for future in as_completed(future_to_file):
                    qf = future_to_file[future]
                    try:
                        out = future.result()
                        if out is None:
                            continue
                        record, exp_inc, full_inc, diff_generated, diff_empty = out
                        file_records.append(record)
                        expected_files += exp_inc
                        fully_downloaded_files += full_inc
                        if diff_generated:
                            diff_generated_count += 1
                            if not diff_empty:
                                diff_saved_count += 1
                        if diff_empty:
                            diff_empty_count += 1
                    except FileDownloadError as e:
                        fname = qf.get('file_data', {}).get('filename', 'unknown')
                        logger.error(f"File download failed; aborting commit processing: {fname}: {e}")
                        download_error = str(e)
                        for f in future_to_file:
                            if not f.done():
                                f.cancel()
                        break
                    except Exception as e:
                        fname = qf.get('file_data', {}).get('filename', 'unknown')
                        logger.error(f"Parallel file processing failed; aborting commit processing: {fname}: {e}")
                        download_error = f"File processing error: {str(e)}"
                        for f in future_to_file:
                            if not f.done():
                                f.cancel()
                        break

        # Merge file records into commit_info.
        commit_info["files"].extend(file_records)
        commit_info["diff_saved_count"] = diff_saved_count
        commit_info["diff_generated_count"] = diff_generated_count
        commit_info["diff_empty_count"] = diff_empty_count

        # Clean empty directories after workers finish to avoid contention.
        cleanup_failed = False
        cleanup_errors = []
        if any(r.get("discarded_due_to_empty_diff") for r in file_records):
            logger.info("Running deferred directory cleanup...")
            project_root = Path(__file__).resolve().parent.parent
            
            for record in file_records:
                if not record.get("discarded_due_to_empty_diff"):
                    continue
                
                try:
                    # Clean parent directories.
                    if record.get("buggy_deleted_path"):
                        abs_path = project_root / record["buggy_deleted_path"]
                        stop_at = get_commit_source_path(owner, repo, commit_sha, "parent")
                        cleanup_empty_parent_dirs(abs_path, stop_at)
                    
                    # Clean commit directories.
                    if record.get("fixed_deleted_path"):
                        abs_path = project_root / record["fixed_deleted_path"]
                        stop_at = get_commit_source_path(owner, repo, commit_sha, "commit")
                        cleanup_empty_parent_dirs(abs_path, stop_at)
                        
                    # Clean diff directories inferred from the relative file path.
                    rel_path = Path(record.get("relative_file_path", ""))
                    if rel_path.name:
                        diff_base_path = get_commit_source_path(owner, repo, commit_sha, 'diffs')
                        diff_parent_dir = diff_base_path / rel_path.parent
                        if diff_parent_dir.exists():
                            cleanup_empty_parent_dirs(diff_parent_dir / "dummy", diff_base_path)
                            
                except Exception as cleanup_e:
                    logger.warning(f"Deferred directory cleanup failed: {cleanup_e}")
                    cleanup_failed = True
                    cleanup_errors.append(str(cleanup_e))

        # Fail fast if any file download aborted the commit.
        if download_error:
            logger.error(f"Commit {commit_sha[:8]} failed because of a file download error: {download_error}")
            commit_info["success"] = False
            commit_info["error"] = f"File download failed: {download_error}"
            return commit_info

        # Success requires complete downloads and complete diff generation.
        # Case 1: every in-scope file is newly added and has no parent version.
        if len(framework_matched_files) > 0 and expected_files == 0:
            logger.info(f"Commit {commit_sha[:8]} completed: all files are newly added, so no comparison is needed.")
            commit_info["success"] = True
            commit_info["message"] = "All files are newly added - no comparison needed"
            return commit_info

        # Case 2: there are modified files that need comparison.
        if expected_files > 0:
            # Check 1: every required file pair downloaded successfully.
            files_downloaded_ok = (expected_files == fully_downloaded_files)
            
            # Check 2: every downloaded pair generated a diff, empty or not.
            diffs_generated_ok = (fully_downloaded_files == diff_generated_count)
            
            # Success requires complete downloads, complete diff generation,
            # and no cleanup failures.
            if files_downloaded_ok and diffs_generated_ok and not cleanup_failed:
                if diff_saved_count > 0:
                    logger.info(
                        f"Commit {commit_sha[:8]} succeeded: "
                        f"files {fully_downloaded_files}/{expected_files}, "
                        f"saved diffs {diff_saved_count}/{fully_downloaded_files}, "
                        f"empty diffs {diff_empty_count}"
                    )
                else:
                    logger.info(
                        f"Commit {commit_sha[:8]} succeeded with only empty diffs: "
                        f"files {fully_downloaded_files}/{expected_files}, "
                        f"empty diffs {diff_empty_count}"
                    )
                commit_info["success"] = True
            else:
                error_parts = []
                if not files_downloaded_ok:
                    error_parts.append(f"incomplete file downloads ({fully_downloaded_files}/{expected_files})")
                if not diffs_generated_ok:
                    error_parts.append(f"incomplete diff generation ({diff_generated_count}/{fully_downloaded_files})")
                if cleanup_failed:
                    error_parts.append(f"directory cleanup failed ({len(cleanup_errors)} errors)")
                
                error_message = ", ".join(error_parts) if error_parts else "Unknown processing error"
                logger.warning(f"Commit {commit_sha[:8]} failed: {error_message}")
                commit_info["success"] = False
                commit_info["error"] = error_message
        else:
            logger.info(f"Commit {commit_sha[:8]} completed: there are no modified files to process.")
            commit_info["success"] = True
            commit_info["message"] = "No modified files to process"
        
        return commit_info
            
    except Exception as e:
        logger.error(f"Failed while processing commit {commit_url}: {e}")
        commit_info["success"] = False
        commit_info["error"] = f"Unhandled exception in process_commit: {str(e)}"
        return commit_info

def generate_bugs_json(status_data: dict, output_path: Path) -> dict:
    """Generate bugs.json from status data using the downloaded file artifacts."""
    commit_details = status_data.get('commit_details', {})
    global CURRENT_FRAMEWORK
    source_code_dir = config.get_framework_source_dir(CURRENT_FRAMEWORK)
    
    # Collect file-level bug records.
    bugs_list = []
    commits_count = 0
    
    for commit_key, info in commit_details.items():
        # Keep only completed commits with downloaded files.
        if info.get('status') != 'completed':
            continue
        if info.get('buggy_files_downloaded', 0) == 0:
            continue
        
        repository = info.get('repository', '')
        if not repository or repository == "/":
            continue
        
        # Parse owner and repository name.
        owner, repo_name = None, None
        if '/' in repository:
            parts = repository.split('/')
            if len(parts) == 2:
                owner, repo_name = parts[0], parts[1]
            elif len(parts) >= 3:
                owner, repo_name = parts[-2], parts[-1]
        
        if not owner or not repo_name:
            continue
        
        commit_sha = info.get('commit_sha', '')
        if not commit_sha:
            continue
        
        commits_count += 1
        
        # Scan the diffs directory to recover the file list.
        diffs_dir = source_code_dir / owner / repo_name / commit_sha / "diffs"
        if not diffs_dir.exists():
            continue
        
        # Walk every .diff file.
        parent_dir = source_code_dir / owner / repo_name / commit_sha / "parent"
        commit_dir = source_code_dir / owner / repo_name / commit_sha / "commit"
        
        for diff_file in diffs_dir.rglob("*.diff"):
            # Strip the .diff suffix relative to the diffs directory.
            rel_path = diff_file.relative_to(diffs_dir)
            file_path = str(rel_path)[:-5]
            
            # Build the real artifact paths.
            buggy_file_path = parent_dir / file_path
            fixed_file_path = commit_dir / file_path
            diff_file_path = diff_file
            
            # Convert paths to project-relative form.
            project_root = config.PROJECT_ROOT
            buggy_rel = str(buggy_file_path.relative_to(project_root)) if buggy_file_path.exists() else ""
            fixed_rel = str(fixed_file_path.relative_to(project_root)) if fixed_file_path.exists() else ""
            diff_rel = str(diff_file_path.relative_to(project_root))
            
            bug_entry = {
                'owner': owner,
                'repo_name': repo_name,
                'repository': repository,
                'commit_sha': commit_sha,
                'commit_url': info.get('commit_url', ''),
                'commit_message': info.get('commit_message', ''),
                'buggy_file_path': buggy_rel,
                'fixed_file_path': fixed_rel,
                'diff_file_path': diff_rel,
            }
            bugs_list.append(bug_entry)
    
    # Keep the output ordering deterministic.
    bugs_list.sort(key=lambda x: (x.get('repository', ''), x.get('commit_sha', ''), x.get('buggy_file_path', '')))
    
    # Summary statistics.
    unique_repos = set(b.get('repository', '') for b in bugs_list)
    unique_commits = set((b.get('repository', ''), b.get('commit_sha', '')) for b in bugs_list)
    
    # Build the output payload.
    bugs_data = {
        "metadata": {
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "File-level bug-fix samples, where each record represents one fixed file.",
            "total_bugs": len(bugs_list),
            "total_commits": len(unique_commits),
            "total_repositories": len(unique_repos),
        },
        "bugs": bugs_list
    }
    
    # Save bugs.json.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bugs_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Generated bugs.json: {output_path}")
    logger.info(f"  Total file-level bugs: {len(bugs_list)}")
    logger.info(f"  From {len(unique_commits)} commits across {len(unique_repos)} repositories")
    
    return bugs_data









def load_json_file(file_path):
    """Load a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download files changed by commits while preserving their original layout.")
    
    # Import config here for script-mode execution.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config
    
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
        help="Path to the commit-list JSON file. Defaults to the framework-specific commits.json."
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help="Path to the output JSON file. Defaults to the framework-specific bugs.json."
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=40,
        help='Maximum number of worker threads for parallel downloads. Default: 40.'
    )
    parser.add_argument(
        '--range',
        type=str,
        default=None,
        help='Commit range to process, for example "1,100".'
    )
    parser.add_argument(
        '--task_id',
        type=str,
        default="",
        help='Task identifier used to separate runs and log files.'
    )
    parser.add_argument(
        '--framework_filter',
        action='store_true',
        default=True,
        help='Keep only files that import the target framework. Enabled by default.'
    )
    # Backward-compatible alias.
    parser.add_argument(
        '--qiskit_filter',
        action='store_true',
        dest='framework_filter',
        help='Alias for --framework_filter.'
    )
    args = parser.parse_args()
    
    # Resolve framework-specific input and output defaults.
    framework = args.framework
    framework_config = config.get_framework_config(framework)
    
    if args.input_json is None:
        args.input_json = str(config.get_framework_commits_path(framework))
    
    if args.output_json is None:
        framework_data_dir = config.get_framework_data_dir(framework)
        args.output_json = str(framework_data_dir / "bugs.json")
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    # Keep the active framework available to path helpers.
    global CURRENT_FRAMEWORK
    CURRENT_FRAMEWORK = framework
    ensure_get_bug_paths(framework)

    # Configure logging.
    configure_logging(args.task_id)
    
    paths = get_unified_paths()

    logger.info("=" * 60)
    logger.info(f"Starting bug-file collection for framework {framework_config['name']}.")
    logger.info(f"Input file: {args.input_json}")
    logger.info(f"Output file: {args.output_json}")
    logger.info("=" * 60)
    logger.info(f"Task ID: {args.task_id or 'default'}")
    logger.info("Workflow: download parent/commit files and generate diffs per commit.")
    logger.info(f"Data layout: data/02_source/{framework}/{{owner}}/{{repo}}/{{commit_sha}}/...")
    logger.info(f"Framework filtering: {'enabled' if args.framework_filter else 'disabled'} ({framework_config['name']})")
    logger.info("=" * 80)

    # Validate GitHub tokens through the shared token manager.
    token_env_vars = detect_github_token_vars()
    if not token_env_vars:
        logger.error("No valid GitHub token was found. Check your environment variables or config.py token settings.")
        sys.exit(1)

    logger.info(f"Found {len(token_env_vars)} valid GitHub tokens: {token_env_vars}")
    
    # Create the token rotator used for stats and automatic rate-limit handling.
    global _global_token_rotator
    token_rotator = TokenRotator(auto_detect=True, requests_per_token=100)
    _global_token_rotator = token_rotator

    # Register signal handlers so status can be flushed on interruption.
    status_data_for_signal = {}
    task_id_for_signal = args.task_id or "main"
    
    def signal_handler(signum, frame):
        """Flush status on SIGINT or SIGTERM before exiting."""
        logger.warning("\nInterrupt signal received; saving status...")
        if status_data_for_signal:
            try:
                flush_status_file(task_id_for_signal, status_data_for_signal)
                logger.info("Status saved.")
            except Exception as e:
                logger.error(f"Failed to save status: {e}")
        logger.warning("Program interrupted and exiting.")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Signal handlers registered. Ctrl+C will trigger a status save.")

    # Load commit data.
    try:
        paths = get_unified_paths()
        input_path = Path(args.input_json)
        if not input_path.is_absolute():
            input_path = paths['metadata_dir'] / input_path
            
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded commit data file: {input_path}")
        
        # Support both the new and legacy layouts.
        if isinstance(data, dict) and "commits" in data:
            commit_data = data.get("commits", {})
            summary_info = data.get("summary", {})
            logger.info(f"File metadata timestamp: {summary_info.get('last_updated', 'N/A')}")
            if summary_info.get("keywords"):
                logger.info(f"Search keywords: {', '.join(summary_info.get('keywords', []))}")
        elif isinstance(data, dict):
            commit_data = data
        else:
            commit_data = {}
        
        # Count repositories and commits.
        total_repos = len(commit_data)
        total_commits = sum(len(commits) for commits in commit_data.values())
        logger.info(f"Data summary: {total_repos} repositories, {total_commits} commits")
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load commit file {args.input_json}: {e}")
        return

    # Build the per-commit task list.
    tasks = []
    for repo_name, commits in commit_data.items():
        try:
            owner, repo = repo_name.split('/')
            lib_name = owner.split('-')[0]
        except ValueError:
            logger.warning(f"Invalid repository name format, skipping: {repo_name}")
            continue

        for commit in commits:
            tasks.append({
                "commit_url": commit.get('commit_url'),
                "commit_sha": commit.get('commit_sha'),
                "parent_sha": commit.get('parent_sha'),
                "lib_name": lib_name,
                "owner": owner,
                "repo": repo
            })

    if not tasks:
        logger.warning("There are no commits to process.")
        return

    # Apply the optional commit-range slice.
    original_task_count = len(tasks)
    commit_range_info = None
    if args.range:
        try:
            range_parts = args.range.split(',')
            if len(range_parts) == 2:
                start_index = int(range_parts[0]) - 1
                end_index = int(range_parts[1])
                
                if start_index < 0:
                    start_index = 0
                if end_index > len(tasks):
                    end_index = len(tasks)
                    
                tasks = tasks[start_index:end_index]
                commit_range_info = f"{start_index+1}-{end_index}"
                logger.info(f"Processing commit range {start_index + 1} to {end_index} ({len(tasks)} commits).")
                logger.info(f"Original commit count: {original_task_count}")
        except (ValueError, IndexError):
            logger.warning(f"Invalid range format: {args.range}. All commits will be processed.")
    else:
        logger.info(f"Processing all {original_task_count} commits.")

    if not tasks:
        logger.warning("There are no commits to process within the selected range.")
        return

    # Initialize status-file management.
    task_id = args.task_id or "main"
    
    # Try resume mode first if an existing status file is present.
    existing_status = load_status_file(task_id)
    if existing_status:
        logger.info("Existing status file found; resume mode is available.")
        prev_summary = existing_status.get('summary', {})
        processed = prev_summary.get('successful_commits', 0) + prev_summary.get('failed_commits', 0)
        logger.info(f"Previous run: {processed}/{prev_summary.get('total_commits', 0)} commits processed")
        logger.info(f"Previous success/failure counts: {prev_summary.get('successful_commits', 0)} / {prev_summary.get('failed_commits', 0)}")
        status_data = existing_status
        status_data_for_signal.update(status_data)
        ensure_pending_entries_for_tasks(status_data, tasks)
        
        # Skip commits that were already completed.
        original_task_count_before_filter = len(tasks)
        tasks = [
            task for task in tasks
            if not is_commit_completed(status_data, task.get("owner", ""), task.get("repo", ""), task.get("commit_sha", ""))
        ]
        filtered_count = original_task_count_before_filter - len(tasks)
        
        if filtered_count > 0:
            logger.info(f"Resume mode skipped {filtered_count} completed commits; {len(tasks)} commits remain.")
        
        if not tasks:
            # Reset stale processing states before exiting.
            pending = 0
            for k, v in list(existing_status.get('commit_details', {}).items()):
                if v.get('status') == 'processing':
                    v['status'] = 'failed'
                    pending += 1
            if pending:
                recompute_summary(existing_status)
                save_status_file(task_id, existing_status)
            finalize_status_file(task_id, existing_status)
            logger.info("All commits are already complete.")
            
            # Still regenerate bugs.json so outputs stay aligned.
            try:
                logger.info("Generating bugs.json...")
                bugs_output_path = Path(args.output_json)
                generate_bugs_json(existing_status, bugs_output_path)
                logger.info("bugs.json generation complete.")
            except Exception as e:
                logger.warning(f"bugs.json generation failed: {e}")
            
            return
    else:
        logger.info("Initializing a new status file.")
        status_data = initialize_status_file(task_id, len(tasks), commit_range_info)
        status_data_for_signal.update(status_data)
        ensure_pending_entries_for_tasks(status_data, tasks)

    logger.info("Stage 1: downloading commit files in parallel...")
    logger.info(f"Worker threads: {args.max_workers}")
    logger.info(f"Token rotation pool size: {len(token_env_vars)}")
    
    # Record the initial memory footprint.
    initial_memory = get_memory_usage_mb()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
    processed_commits_count = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {}
        for task in tasks:
            # Get the next token name through the rotator.
            token_name, _ = token_rotator.get_token_name()
            future = executor.submit(process_commit, task["commit_url"], task["lib_name"], token_name, args.framework_filter)
            future_to_task[future] = task
        
        with tqdm(total=len(tasks), desc="Processing commits", unit="commit") as pbar:
            try:
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        if not result:
                            task_info = future_to_task[future]
                            commit_sha = task_info.get("commit_sha", "")
                            if commit_sha:
                                commit_info = {
                                    'commit_sha': commit_sha,
                                    'owner': task_info.get("owner", "unknown"),
                                    'repo': task_info.get("repo", "unknown"),
                                    'message': '', 'parent_sha': '',
                                    'commit_url': task_info.get("commit_url"),
                                }
                                update_commit_status(task_id, status_data, commit_info, False, errors=["Worker returned None"])
                                save_status_file(task_id, status_data)
                            logger.warning(f"A commit worker returned no result and was skipped: {task_info.get('commit_url')}")
                            
                            del future_to_task[future]
                            pbar.update(1)
                            continue

                        processed_commits_count += 1
                        commit_info = {
                            'commit_sha': result.get('commit_sha', ''),
                            'owner': result.get('owner', ''),
                            'repo': result.get('repo', ''),
                            'message': result.get('commit_message', ''),
                            'parent_sha': result.get('parent_sha', ''),
                            'commit_url': result.get('commit_url'),
                        }
                        success = result.get('success', False)
                        
                        # Final status update
                        buggy_count = sum(1 for f in result.get('files', []) if f.get('download_status', {}).get('buggy_success', False))
                        fixed_count = sum(1 for f in result.get('files', []) if f.get('download_status', {}).get('fixed_success', False))
                        diff_saved = result.get('diff_saved_count', 0)
                        diff_discarded = result.get('diff_empty_count', 0)
                        
                        status_info = {
                            'buggy_files_downloaded': buggy_count,
                            'fixed_files_downloaded': fixed_count,
                            'diff_files_saved': diff_saved,
                            'diff_files_discarded': diff_discarded,
                            'framework_filtered': result.get('framework_filtered', False),
                            'status': 'completed' if success else 'failed'
                        }
                        if 'error' in result:
                            status_info['errors'] = [result['error']]
                        
                        # Log per-commit file statistics when empty diffs were discarded.
                        if diff_discarded > 0:
                            logger.info(f"Commit {commit_info['commit_sha'][:8]} file stats: "
                                      f"Buggy={buggy_count}, Fixed={fixed_count}, "
                                      f"Diff saved={diff_saved}, Empty diff discarded={diff_discarded}")
                        
                        update_commit_status(task_id, status_data, commit_info, success, **status_info)
                        save_status_file(task_id, status_data)
                        
                        del future_to_task[future]
                        del result
                        
                        # Run periodic GC during large runs.
                        if processed_commits_count % 100 == 0:
                            current_memory = get_memory_usage_mb()
                            gc.collect()
                            after_gc_memory = get_memory_usage_mb()
                            logger.info(f"Processed {processed_commits_count} commits | "
                                      f"memory: {current_memory:.2f} MB -> {after_gc_memory:.2f} MB (GC freed {current_memory - after_gc_memory:.2f} MB)")

                    except Exception as e:
                        task = future_to_task[future]
                        commit_url = task['commit_url']
                        commit_sha = task.get('commit_sha', '')
                        logger.error(f"Commit worker raised an exception: {commit_url}, error: {e}")
                        if commit_sha:
                            commit_info = {
                                'commit_sha': commit_sha,
                                'owner': task.get("owner", "unknown"),
                                'repo': task.get("repo", "unknown"),
                                'message': '', 'parent_sha': '',
                                'commit_url': task.get('commit_url'),
                            }
                            update_commit_status(task_id, status_data, commit_info, False, errors=[f"Commit worker exception: {str(e)}"])
                            save_status_file(task_id, status_data)
                        
                        del future_to_task[future]
                    
                    pbar.update(1)
            except FuturesTimeout:
                for f, task in list(future_to_task.items()):
                    if not f.done():
                        try:
                            f.cancel()
                        except Exception:
                            pass
                        commit_url = task['commit_url']
                        commit_sha = task.get('commit_sha', '')
                        if commit_sha:
                            commit_info = {
                                'commit_sha': commit_sha,
                                'owner': task.get("owner", "unknown"),
                                'repo': task.get("repo", "unknown"),
                                'message': '', 'parent_sha': '',
                                'commit_url': task.get('commit_url')
                            }
                            update_commit_status(task_id, status_data, commit_info, False, errors=["Processing timed out"])
                            save_status_file(task_id, status_data)
                        pbar.update(1)
                logger.warning("Thread pool wait timed out; unfinished tasks were canceled and marked failed.")
                future_to_task.clear()
                gc.collect()

    logger.info("Running final memory cleanup...")
    before_final_gc = get_memory_usage_mb()
    gc.collect()
    after_final_gc = get_memory_usage_mb()
    logger.info(f"Final memory: {after_final_gc:.2f} MB (freed {before_final_gc - after_final_gc:.2f} MB) | total growth: {after_final_gc - initial_memory:.2f} MB")
    
    # Convert any lingering processing states to failed.
    try:
        pending = 0
        for k, v in list(status_data.get('commit_details', {}).items()):
            if v.get('status') == 'processing':
                v['status'] = 'failed'
                pending += 1
        if pending:
            logger.info(f"Marked {pending} unfinished 'processing' commits as 'failed'.")
            recompute_summary(status_data)
            save_status_file(task_id, status_data)
    except Exception:
        pass

    processing_time = time.time() - start_time
    
    if processed_commits_count > 0:
        total_files = sum(s.get('buggy_files_downloaded', 0) for s in status_data.get('commit_details', {}).values())
        successful_commits_final = sum(1 for s in status_data.get('commit_details', {}).values() if s.get('status') == 'completed')
        total_diffs_saved = sum(s.get('diff_files_saved', 0) for s in status_data.get('commit_details', {}).values())
        total_diffs_discarded = sum(s.get('diff_files_discarded', 0) for s in status_data.get('commit_details', {}).values())

        logger.info("All commit processing is complete.")
        logger.info("=" * 80)
        logger.info("Processing summary:")
        logger.info(f"  Total processing time: {processing_time:.2f}s")
        logger.info(f"  Processed commits: {processed_commits_count} / {len(tasks)}")
        logger.info(f"  Successful commits: {successful_commits_final}")
        logger.info(f"  Downloaded file pairs: {total_files} (parent/commit)")
        logger.info(f"  Saved diff files: {total_diffs_saved}")
        if total_diffs_discarded > 0:
            logger.info(f"  Discarded empty diff files: {total_diffs_discarded}")
        
        logger.info("Saved file layout:")
        logger.info("   data/source_code/{owner}/{repo}/{commit_sha}/")
        logger.info("   ├── parent/        # Parent version, original layout preserved")
        logger.info("   ├── commit/        # Commit version, original layout preserved")
        logger.info("   ├── diffs/         # Locally generated diff files")
        
    else:
        logger.warning("No commits were processed successfully.")

    save_status_file(task_id, status_data)
    finalize_status_file(task_id, status_data)

    # Generate bugs.json at the end of the run.
    try:
        logger.info("Generating bugs.json...")
        bugs_output_path = Path(args.output_json)
        generate_bugs_json(status_data, bugs_output_path)
        logger.info("bugs.json generation complete.")
    except Exception as e:
        logger.warning(f"bugs.json generation failed: {e}")

    logger.info("=" * 80)
    
    # Show token-usage statistics.
    token_rotator.log_stats()
    
    logger.info("Task finished.")

def process_single_commit_diffs(commit_result: dict):
    """Processes a single commit to generate all its diffs."""
    owner = commit_result.get('owner')
    repo = commit_result.get('repo')
    commit_sha = commit_result.get('commit_sha')
    files_to_process = commit_result.get('files', [])
    
    if not all([owner, repo, commit_sha, files_to_process]):
        return False, 0, "Missing key information for diff generation"
    
    diff_saved_count = 0
    all_diffs_ok = True
    
    for file_info in files_to_process:
        if file_info.get('discarded_due_to_empty_diff'):
            continue
        buggy_rel_path = file_info.get('buggy')
        fixed_rel_path = file_info.get('fixed')
        original_filename = file_info.get('relative_file_path')

        if not all([buggy_rel_path, fixed_rel_path, original_filename]):
            all_diffs_ok = False
            continue

        project_root = Path(__file__).resolve().parent.parent
        buggy_abs_path = project_root / buggy_rel_path
        fixed_abs_path = project_root / fixed_rel_path
        
        if generate_and_save_diff(owner, repo, commit_sha, original_filename, buggy_abs_path, fixed_abs_path):
            diff_saved_count += 1
        else:
            all_diffs_ok = False
    
    if not all_diffs_ok:
        return False, diff_saved_count, f"diffs incomplete ({diff_saved_count}/{len(files_to_process)})"
    
    return True, diff_saved_count, ""

def generate_and_save_diff(owner: str, repo: str, commit_sha: str, original_filename: str, buggy_path: Path, fixed_path: Path) -> tuple[bool, bool]:
    """
    Generates a diff and saves it.
    Returns a tuple (success, is_empty).
    """
    try:
        # Define the diff file path
        base_diff_dir = get_commit_source_path(owner, repo, commit_sha, 'diffs')
        file_path_obj = Path(original_filename)
        diff_dir = base_diff_dir / file_path_obj.parent
        diff_dir.mkdir(parents=True, exist_ok=True)
        diff_file_path = diff_dir / f"{file_path_obj.name}.diff"

        if diff_file_path.exists():
            logger.debug(f"Diff file already exists, skipping generation: {diff_file_path}")
            # Assume an existing diff file is valid and non-empty.
            return True, False # Success, not empty

        # Read file contents
        with open(buggy_path, 'r', encoding='utf-8', errors='ignore') as f:
            buggy_lines = f.readlines()
        with open(fixed_path, 'r', encoding='utf-8', errors='ignore') as f:
            fixed_lines = f.readlines()

        # Stream unified diff lines directly to disk.
        with open(diff_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# File: {original_filename}\n")
            f.write(f"# Repository: {owner}/{repo}\n")
            f.write(f"# Commit: {commit_sha}\n")
            f.write(f"# Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
            f.write("\n")
            wrote_any = False
            for line in difflib.unified_diff(
                buggy_lines,
                fixed_lines,
                fromfile=f"a/{original_filename}",
                tofile=f"b/{original_filename}",
            ):
                f.write(line)
                wrote_any = True
        
        if not wrote_any:
            logger.warning(f"Generated diff is empty for {original_filename}. This file pair will be discarded.")
            return True, True # Success, but empty
        
        logger.debug(f"Successfully generated and saved diff: {diff_file_path}")
        return True, False # Success, not empty

    except Exception as e:
        logger.error(f"Failed to generate or save diff for {original_filename}: {e}")
        return False, False

# detect_github_token_vars now lives in util/token_manager.py.

if __name__ == "__main__":
    main()
