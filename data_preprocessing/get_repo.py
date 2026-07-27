import argparse
import os
import sys
import requests
import time
import json
import logging
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

try:
    import config
    from util.retry import RetryableRequest, exponential_backoff, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
    from util.token_manager import TokenRotator, detect_github_tokens, calculate_sleep_time
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    import config
    from util.retry import RetryableRequest, exponential_backoff, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
    from util.token_manager import TokenRotator, detect_github_tokens, calculate_sleep_time

# ==============================================================================
# SCRIPT-SPECIFIC CONFIGURATION
# ==============================================================================
REPO_SEARCH_QUERY = config.get_search_query()

def get_excluded_repos(framework: str | None = None) -> set:
    """Return global and framework-specific repositories to exclude."""
    excluded = set(config.GLOBAL_EXCLUDED_REPOS)
    if framework and framework in config.QUANTUM_FRAMEWORKS:
        fw_config = config.QUANTUM_FRAMEWORKS[framework]
        excluded.update(fw_config.get("exclude_repos", []))
    return excluded

def ensure_get_repo_paths(framework: str | None, output_file: str | Path) -> None:
    """Create only the directories required by get_repo."""
    if hasattr(config, 'ensure_repo_collection_structure'):
        config.ensure_repo_collection_structure(framework)
    else:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)


def get_repo_api_worker_limits(token_rotator: TokenRotator, requested_workers: int) -> tuple[int, int]:
    """Derive safer worker counts for language and commit API requests."""
    token_count = max(1, len(token_rotator))
    language_workers = max(2, min(requested_workers, token_count * 5))
    commit_workers = max(2, min(requested_workers, token_count * 4))
    return language_workers, commit_workers


def _branch_exists(repo_full_name: str, branch: str, token_rotator: TokenRotator) -> bool:
    """Return True if the repository exposes the given branch."""
    url = f"https://api.github.com/repos/{repo_full_name}/branches/{branch}"
    try:
        res = requests.get(url, headers=token_rotator.get_headers(), timeout=15)
        if res.status_code == 403:
            handle_rate_limit_error(res.headers, token_rotator)
            res = requests.get(url, headers=token_rotator.get_headers(), timeout=15)
        if res.status_code == 404:
            return False
        res.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def _pick_branch(repo_full_name: str, token_rotator: TokenRotator) -> str | None:
    """Preserve the original branch priority: main, then master, then default."""
    if _branch_exists(repo_full_name, "main", token_rotator):
        return "main"
    if _branch_exists(repo_full_name, "master", token_rotator):
        return "master"
    info_url = f"https://api.github.com/repos/{repo_full_name}"
    try:
        res = requests.get(info_url, headers=token_rotator.get_headers(), timeout=15)
        if res.status_code == 403:
            handle_rate_limit_error(res.headers, token_rotator)
            res = requests.get(info_url, headers=token_rotator.get_headers(), timeout=15)
        res.raise_for_status()
        data = res.json()
        return data.get("default_branch")
    except requests.exceptions.RequestException:
        return None

def handle_rate_limit_error(headers, token_rotator: TokenRotator | None = None):
    """Handle GitHub rate limits, preferring token rotation when available."""
    if token_rotator is not None and len(token_rotator) > 1:
        old_name = token_rotator.get_current_token_name()
        token_rotator.rotate(reason="rate_limit")
        new_name = token_rotator.get_current_token_name()
        logging.warning(f"Rate limit exceeded for {old_name}. Rotated to {new_name}.")
        time.sleep(1)
        return

    reset_time_str = headers.get("X-RateLimit-Reset")
    if reset_time_str:
        reset_time = datetime.fromtimestamp(int(reset_time_str))
        wait_time = (reset_time - datetime.now()).total_seconds() + 1
        if wait_time > 0:
            logging.warning(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
            time.sleep(wait_time)
            return
    logging.warning("Rate limit exceeded. Waiting for 60 seconds.")
    time.sleep(60)

def fetch_language_data(repo_item, token_rotator: TokenRotator, max_retries=DEFAULT_MAX_RETRIES):
    """Fetch repository language metadata with retry support."""
    lang_url = repo_item.get("languages_url")
    repo_name = repo_item['full_name']
    if not lang_url:
        return repo_name, {}, 0, None

    retrier = RetryableRequest(
        max_retries=max_retries,
        timeout=DEFAULT_TIMEOUT,
        rate_limit_handler=lambda h: handle_rate_limit_error(h, token_rotator),
        headers_getter=token_rotator.get_headers,
    )
    
    response = retrier.get(lang_url, headers=token_rotator.get_headers())
    
    if response is None:
        return repo_name, None, max_retries, "Request failed after retries"
    
    if response.status_code == 404:
        return repo_name, {}, 1, None
    
    try:
        data = response.json()
        if not isinstance(data, dict):
            data = {}
        return repo_name, data, 1, None
    except Exception as e:
        return repo_name, None, 1, str(e)

def get_commit_count(repo_full_name: str, token_rotator: TokenRotator, max_retries=DEFAULT_MAX_RETRIES) -> int | None:
    """Estimate commit count for one repository."""
    url = f"https://api.github.com/repos/{repo_full_name}/commits"
    branch = _pick_branch(repo_full_name, token_rotator)
    params = {"per_page": 100, "until": config.DATA_COLLECTION_CUTOFF}
    if branch:
        params["sha"] = branch
    
    retrier = RetryableRequest(
        max_retries=max_retries,
        timeout=DEFAULT_TIMEOUT,
        rate_limit_handler=lambda h: handle_rate_limit_error(h, token_rotator),
        headers_getter=token_rotator.get_headers,
    )
    
    params["page"] = 1
    res = retrier.get(url, headers=token_rotator.get_headers(), params=params)
    
    if res is None:
        logging.error(f"Failed to get commit count for {repo_full_name} after retries.")
        return None
    
    if res.status_code in (409, 404):
        return 0
    
    try:
        items = res.json()
        if not isinstance(items, list) or len(items) == 0:
            return 0
        
        link = res.headers.get("Link")
        if link:
            m = re.search(r'page=(\d+)>; rel="last"', link)
            if m:
                last_page = int(m.group(1))
                if last_page <= 1:
                    return len(items)
                
                params["page"] = last_page
                res2 = retrier.get(url, headers=token_rotator.get_headers(), params=params)
                
                if res2 is None:
                    return last_page * 100
                
                last_items = res2.json()
                if not isinstance(last_items, list):
                    last_items = []
                return (last_page - 1) * 100 + len(last_items)
        
        return len(items)
    except Exception as e:
        logging.error(f"Error parsing commit count for {repo_full_name}: {e}")
        return None

def search_repositories_single(query, token_rotator: TokenRotator, max_workers, framework=None):
    """Run one GitHub repository search query."""
    repos_dict = {}
    api_url = "https://api.github.com/search/repositories"
    params = {"q": query, "per_page": 100}
    language_workers, commit_workers = get_repo_api_worker_limits(token_rotator, max_workers)
    
    excluded_repos = get_excluded_repos(framework)
    
    page = 1
    while True:
        params["page"] = page
        
        response = None
        last_error = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                logging.info(f"Searching page {page} for query: '{query}' (Attempt {attempt + 1}/{max_retries})")
                response = requests.get(api_url, headers=token_rotator.get_headers(), params=params, timeout=20)

                if response.status_code == 403:
                    handle_rate_limit_error(response.headers, token_rotator)
                    continue

                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                last_error = e
                logging.warning(f"Attempt {attempt + 1}/{max_retries} for page {page} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                response = None

        if response is None:
            logging.error(f"Failed to fetch page {page} after {max_retries} attempts. Last error: {last_error}")
            break

        data = response.json()
        items = data.get("items", [])

        for item in items:
            repo_full_name = item['full_name']
            if repo_full_name in excluded_repos:
                logging.info(f"🚫 Excluded repo: {repo_full_name}")
                continue

            if not item['fork']:
                repos_dict[repo_full_name] = {
                    "url": item['html_url'],
                    "description": item['description'],
                    "stars": item['stargazers_count'],
                    "fork": item['fork'],
                    "created_at": item['created_at'],
                    "languages_url": item['languages_url'],
                    "languages": {}
                }

        if not items or len(items) < params["per_page"]:
            break

        page += 1

    repo_items_to_fetch = [
        {"full_name": name, "languages_url": info["languages_url"]}
        for name, info in repos_dict.items()
    ]
    if repo_items_to_fetch:
        logging.info(f"Using {language_workers} worker(s) for language metadata requests.")
        with ThreadPoolExecutor(max_workers=language_workers) as executor:
            future_to_repo = {executor.submit(fetch_language_data, item, token_rotator, 3): item for item in repo_items_to_fetch}
            for future in tqdm(as_completed(future_to_repo), total=len(repo_items_to_fetch), desc="Fetching language data"):
                repo_name, languages, attempts, err = future.result()
                if repo_name in repos_dict:
                    failed = languages is None
                    lang_dict = languages if isinstance(languages, dict) else {}
                    repos_dict[repo_name]['languages'] = lang_dict
                    repos_dict[repo_name]['_lang_fetch_failed'] = failed
                    repos_dict[repo_name]['_lang_fetch_attempts'] = attempts
                    repos_dict[repo_name]['_lang_empty'] = (not failed and len(lang_dict) == 0)
                    if err:
                        repos_dict[repo_name]['_lang_fetch_error'] = err

    lang_kept = [
        name for name, info in repos_dict.items()
        if isinstance(info.get('languages'), dict) and 'Python' in info['languages']
    ]
    lang_unknown = [
        name for name, info in repos_dict.items()
        if info.get('_lang_fetch_failed') and not info.get('_lang_empty')
    ]
    candidate_names = list({*lang_kept, *lang_unknown})
    if not candidate_names:
        logging.info("⚠️ No repositories with Python or unresolved language metadata were found")
        return {}

    logging.info(f"Using {commit_workers} worker(s) for commit count requests.")
    with ThreadPoolExecutor(max_workers=commit_workers) as executor:
        futures = {executor.submit(get_commit_count, name, token_rotator): name for name in candidate_names}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching commit counts"):
            name = futures[future]
            count = future.result()
            if name in repos_dict:
                repos_dict[name]['commit_count'] = count if count is not None else 0
    
    before = len(repos_dict)
    filtered_repos = {}
    excluded_suspicious_count = 0

    for name, info in repos_dict.items():
        commit_count = info.get('commit_count', 0)
        stars = info.get('stars', 0)

        if commit_count <= 0:
            continue
        
        if commit_count > 500 and stars <= 5:
            logging.info(f"🚫 Suspicious repo excluded (commits>500, stars<=5): {name} ({commit_count} commits, {stars} stars)")
            excluded_suspicious_count += 1
            continue
            
        filtered_repos[name] = info
    
    repos_dict = filtered_repos
    
    logging.info(f"✅ Kept valid repositories: {len(repos_dict)}/{before}")
    if excluded_suspicious_count > 0:
        logging.info(f"   Excluded suspicious manual forks: {excluded_suspicious_count}")

    return repos_dict

def search_repositories(base_query, token_rotator: TokenRotator, max_workers, use_time_segments=True, framework=None):
    """Search repositories, optionally splitting by time segment."""
    if not use_time_segments:
        return search_repositories_single(base_query, token_rotator, max_workers, framework=framework)
    
    all_repos = {}
    time_segments = config.SEARCH_TIME_SEGMENTS
    
    logging.info(f"📅 Starting segmented search across {len(time_segments)} time segment(s)")
    
    for i, (start_date, end_date, description) in enumerate(time_segments, 1):
        time_query = config.get_time_segment_query(start_date, end_date)
        full_query = f"{base_query} {time_query}" if time_query else base_query
        
        logging.info(f"\n{'='*60}")
        logging.info(f"📅 [{i}/{len(time_segments)}] {description}")
        logging.info(f"   Query: {full_query}")
        logging.info(f"{'='*60}")
        
        segment_repos = search_repositories_single(full_query, token_rotator, max_workers, framework=framework)
        
        new_count = 0
        for repo_name, repo_info in segment_repos.items():
            if repo_name not in all_repos:
                all_repos[repo_name] = repo_info
                new_count += 1
        
        logging.info(f"✅ Segment [{description}] complete: fetched {len(segment_repos)} repos, added {new_count}")
        logging.info(f"   Running total: {len(all_repos)} repos")
        
        time.sleep(2)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"🎉 Segmented search complete with {len(all_repos)} unique repositories")
    logging.info(f"{'='*60}")
    
    return all_repos


def save_repositories_to_json(repos_dict, output_file):
    """Merge repositories into the output JSON and refresh summary counts."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_data = {}
    if output_path.exists() and os.path.getsize(output_path) > 0:
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Could not parse existing JSON file {output_path}; creating a new one")

    if isinstance(existing_data, dict) and "repos" in existing_data and isinstance(existing_data["repos"], dict):
        repo_map = existing_data["repos"]
    elif isinstance(existing_data, dict):
        repo_map = existing_data
    else:
        repo_map = {}

    keys_to_remove = [
        "languages_url",
        "_lang_fetch_failed",
        "_lang_fetch_attempts",
        "_lang_empty",
        "_lang_fetch_error"
    ]
    for repo_info in repos_dict.values():
        for key in keys_to_remove:
            repo_info.pop(key, None)

    repo_map.update(repos_dict)
    sorted_repos = sorted(repo_map.items(), key=lambda item: item[1].get('commit_count', 0), reverse=True)
    sorted_repo_map = {k: v for k, v in sorted_repos}

    final_data = {
        "summary": {
            "total_repos": len(sorted_repo_map)
        },
        "repos": sorted_repo_map
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
    logging.info(f"Merged repository info into {output_path}; total repositories: {len(repo_map)}")

def main():
    parser = argparse.ArgumentParser(description="Search GitHub repositories and collect language metadata.")
    parser.add_argument(
        '--query',
        type=str,
        default=REPO_SEARCH_QUERY,
        help=f"GitHub search query. Default: '{REPO_SEARCH_QUERY}'"
    )
    parser.add_argument(
        '--framework',
        type=str,
        default=config.ACTIVE_FRAMEWORK,
        choices=list(config.QUANTUM_FRAMEWORKS.keys()),
        help=f"Quantum framework to collect. Choices: {list(config.QUANTUM_FRAMEWORKS.keys())}. Default: {config.ACTIVE_FRAMEWORK}"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help="Output JSON path. Defaults to the framework-specific repos.json."
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=40,
        help='Maximum number of concurrent API workers.'
    )
    parser.add_argument(
        '--token_env_var',
        type=str,
        default=config.GITHUB_TOKEN_ENV_VARS[0] if config.GITHUB_TOKEN_ENV_VARS else "",
        help='GitHub token environment variable name.'
    )
    parser.add_argument(
        '--no-segment',
        action='store_true',
        help='Disable time-segmented search and keep the standard 1000-result cap.'
    )
    parser.add_argument(
        '--segment',
        action='store_true',
        help='Force time-segmented search, overriding the framework default.'
    )
    args = parser.parse_args()

    framework = args.framework
    framework_config = config.get_framework_config(framework)
    
    if args.query == REPO_SEARCH_QUERY:
        query = config.get_search_query(framework)
    else:
        query = args.query
    
    if args.output_file is None:
        output_file = str(config.get_framework_repos_path(framework))
    else:
        output_file = args.output_file

    ensure_get_repo_paths(framework, output_file)
    log_dir = config.LOG_DIR

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    full_log_handler = logging.FileHandler(log_dir / "get_repo.log", mode='a', encoding='utf-8')
    full_log_handler.setFormatter(formatter)
    logger.addHandler(full_log_handler)

    error_log_handler = logging.FileHandler(log_dir / "get_repo_error.log", mode='a', encoding='utf-8')
    error_log_handler.setLevel(logging.ERROR)
    error_log_handler.setFormatter(formatter)
    logger.addHandler(error_log_handler)

    tokens, token_names = detect_github_tokens()
    
    if not tokens:
        logging.error("❌ No GitHub tokens found. Set environment variables such as GITHUB_TOKEN_1 or GITHUB_TOKEN_2.")
        return
    
    logging.info(f"🔑 Detected {len(tokens)} GitHub token(s): {', '.join(token_names)}")
    
    token_rotator = TokenRotator(tokens, token_names, requests_per_token=100)
    language_workers, commit_workers = get_repo_api_worker_limits(token_rotator, args.max_workers)
    
    logging.info(f"=" * 60)
    logging.info(f"🚀 Collecting repositories for the {framework_config['name']} ecosystem")
    logging.info(f"   Query: {query}")
    logging.info(f"   Output: {output_file}")
    logging.info(f"=" * 60)

    if getattr(args, 'no_segment', False):
        use_time_segments = False
    elif getattr(args, 'segment', False):
        use_time_segments = True
    else:
        use_time_segments = framework_config.get('use_time_segments', False)
    
    logging.info(f"   Segmented search: {'enabled' if use_time_segments else 'disabled'}")
    logging.info(f"   Requested max workers: {args.max_workers}")
    logging.info(f"   Effective language workers: {language_workers}")
    logging.info(f"   Effective commit-count workers: {commit_workers}")
    
    all_repos = search_repositories(query, token_rotator, args.max_workers, use_time_segments=use_time_segments, framework=framework)
    
    if all_repos:
        save_repositories_to_json(all_repos, output_file)
        logging.info(f"✅ {framework_config['name']} repository collection complete: {len(all_repos)} total")
    
    token_rotator.log_stats()

if __name__ == "__main__":
    main()
