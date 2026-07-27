#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central project configuration and path helpers."""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# ==============================================================================
# PROJECT STRUCTURE
# ==============================================================================
# Defines the core directory structure for the project.
PROJECT_ROOT = Path(__file__).resolve().parent

# ==============================================================================
# GITHUB API SETTINGS
# ==============================================================================
# Environment variables that hold GitHub tokens for API access.
# Using multiple tokens can help mitigate rate limiting.
GITHUB_TOKEN_ENV_VARS = ["GITHUB_TOKEN_1", "GITHUB_TOKEN_2", "GITHUB_TOKEN_3", "GITHUB_TOKEN_4", "GITHUB_TOKEN_5","GITHUB_TOKEN_6"]

# ==============================================================================
# KEYWORD LISTS
# ==============================================================================
# Keywords for searching pull requests related to bug fixes.
# These are used to identify PRs that are likely to contain bug-fixing commits.
PR_SEARCH_KEYWORDS = [
    "bug", "fix", "error", "issue", "mistake", "defect",
    "incorrect", "fault", "flaw", "type"
]

# ==============================================================================
# DATA COLLECTION TIME CUTOFF
# ==============================================================================
# Shared data-collection cutoff used by repository search and commit filtering.
DATA_COLLECTION_CUTOFF = "2025-12-01T23:59:59Z"
DATA_COLLECTION_CUTOFF_DATE = "2025-12-01"

# ==============================================================================
# QUANTUM FRAMEWORK CONFIGURATIONS
# ==============================================================================
# Supported quantum framework settings.
QUANTUM_FRAMEWORKS = {
    "qiskit": {
        "name": "Qiskit",
        "search_query": "qiskit",
        "exclude_orgs": ["qiskit", "pennylaneai"],
        "exclude_repos": [],
        "import_patterns": ["import qiskit", "from qiskit"],
        "description": "IBM's open-source quantum computing framework",
        "use_time_segments": True,
    },
    "cirq": {
        "name": "Cirq",
        "search_query": "cirq",
        "exclude_orgs": ["quantumlib", "pennylaneai", "YZNIU", "eric-erki"],
        "exclude_repos": ["siinfante/Cirq"],
        "import_patterns": ["import cirq", "from cirq"],
        "description": "Google's quantum computing framework",
        "use_time_segments": False,
    },
    "pennylane": {
        "name": "PennyLane",
        "search_query": "pennylane",
        "exclude_orgs": ["pennylaneai", "ddinesan"],
        "exclude_repos": [],
        "import_patterns": ["import pennylane", "from pennylane"],
        "description": "Xanadu's quantum machine learning framework",
        "use_time_segments": False,
    },
}

# Default active framework.
ACTIVE_FRAMEWORK = "qiskit"

# ==============================================================================
# GLOBAL EXCLUDED REPOSITORIES
# ==============================================================================
# Global repository exclusions shared by all frameworks.
GLOBAL_EXCLUDED_REPOS = [
    "Z-928/Bugs4Q",
    "indian-institute-of-science-qc/qiskit-aakash",
    "DonPharoah/andersen-lab-qiskit-metal",
    "ivo53/qiskit_experiments",
]

# ==============================================================================
# TIME SEGMENTS FOR GITHUB SEARCH
# ==============================================================================
# Time slices used to bypass the GitHub search API 1000-result cap.
SEARCH_TIME_SEGMENTS = [
    ("2025-09-01", "2025-12-01", "Sep 2025 to Dec 2025"),
    ("2025-06-01", "2025-09-01", "Jun 2025 to Sep 2025"),
    ("2024-12-01", "2025-06-01", "Dec 2024 to Jun 2025"),
    ("2024-03-01", "2024-12-01", "Mar 2024 to Dec 2024"),
    ("2023-04-01", "2024-03-01", "Apr 2023 to Mar 2024"),
    ("2022-04-01", "2023-04-01", "Apr 2022 to Apr 2023"),
    ("2021-04-01", "2022-04-01", "Apr 2021 to Apr 2022"),
    ("2020-04-01", "2021-04-01", "Apr 2020 to Apr 2021"),
    (None, "2020-04-01", "Before Apr 2020"),
]


def get_time_segment_query(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Build a GitHub created-date query fragment."""
    if start_date and end_date:
        return f"created:{start_date}T00:00:00Z..{end_date}T23:59:59Z"
    elif end_date:
        return f"created:<{end_date}T00:00:00Z"
    elif start_date:
        return f"created:>{start_date}T00:00:00Z"
    return ""


def get_framework_config(framework: Optional[str] = None) -> dict:
    """Return the configuration for one framework."""
    fw = framework or ACTIVE_FRAMEWORK
    if fw not in QUANTUM_FRAMEWORKS:
        raise ValueError(f"Unknown framework: {fw}. Available: {list(QUANTUM_FRAMEWORKS.keys())}")
    return QUANTUM_FRAMEWORKS[fw]


def get_search_query(framework: Optional[str] = None) -> str:
    """Build the GitHub search query for one framework."""
    config = get_framework_config(framework)
    query = config["search_query"]
    for org in config["exclude_orgs"]:
        query += f" -org:{org}"
    return query


def get_import_patterns(framework: Optional[str] = None) -> list:
    """Return the import patterns for one framework."""
    return get_framework_config(framework)["import_patterns"]


def contains_framework_import(content: str, framework: Optional[str] = None) -> bool:
    """Return True if content imports the target framework."""
    patterns = get_import_patterns(framework)
    return any(pattern in content for pattern in patterns)

# =============================================================================
# Data layout
# =============================================================================

DATA_ROOT = PROJECT_ROOT / "data"

DATA_PREPROCESSING_ROOT = PROJECT_ROOT / "data_preprocessing"
LOG_DIR = DATA_PREPROCESSING_ROOT / "log"

RAW_DATA_DIR = DATA_ROOT / "01_raw"
SOURCE_CODE_DIR = DATA_ROOT / "02_source"
EXTRACTED_DIR = DATA_ROOT / "03_extracted"
ANALYZED_DIR = DATA_ROOT / "04_analyzed"
DATASETS_DIR = DATA_ROOT / "05_datasets"
CACHE_DIR = DATA_ROOT / "cache"


def get_framework_data_dir(framework: Optional[str] = None, base_dir: Optional[Path] = None) -> Path:
    """Return the raw-data directory for one framework."""
    fw = framework or ACTIVE_FRAMEWORK
    base = base_dir or RAW_DATA_DIR
    return base / fw


def get_framework_repos_path(framework: Optional[str] = None) -> Path:
    """Return the repos.json path for one framework."""
    return get_framework_data_dir(framework) / "repos.json"


def get_framework_commits_path(framework: Optional[str] = None) -> Path:
    """Return the commits.json path for one framework."""
    return get_framework_data_dir(framework) / "commits.json"


def get_framework_source_dir(framework: Optional[str] = None) -> Path:
    """Return the source-code directory for one framework."""
    fw = framework or ACTIVE_FRAMEWORK
    return SOURCE_CODE_DIR / fw

# Common subdirectories.
RAW_STATUS_DIR = RAW_DATA_DIR / "status"
ANALYZED_STAGE1_DIR = ANALYZED_DIR / "stage1"
ANALYZED_STAGE2_DIR = ANALYZED_DIR / "stage2"
ANALYZED_REGISTRIES_DIR = ANALYZED_DIR / "registries"
ANALYZED_ERRORS_DIR = ANALYZED_DIR / "errors"
ANALYZED_EVAL_SAMPLES_DIR = ANALYZED_DIR / "evaluation_samples"
ANALYZED_REPORTS_DIR = ANALYZED_DIR / "reports"
DATASETS_VIS_DIR = DATASETS_DIR / "visualizations"
CACHE_COMMITS_DIR = CACHE_DIR / "commits"

# =============================================================================
# Standard file paths
# =============================================================================

REPOS_JSON_PATH = RAW_DATA_DIR / "repos.json"
COMMITS_JSON_PATH = RAW_DATA_DIR / "commits.json"
BUGS_JSON_PATH = RAW_DATA_DIR / "bugs.json"

METHOD_LEVEL_ALL_JSON = EXTRACTED_DIR / "method_level_all.json"
METHOD_LEVEL_QUANTUM_JSON = EXTRACTED_DIR / "method_level_quantum.json"
METHOD_LEVEL_SINGLE_JSON = EXTRACTED_DIR / "method_level_single.json"

ALL_STAGE1_JSON = ANALYZED_STAGE1_DIR / "all_stage1.json"
ALL_STAGE2_JSON = ANALYZED_STAGE2_DIR / "all_stage2.json"
VALID_STAGE1_JSON = ANALYZED_STAGE1_DIR / "valid_stage1.json"
VALID_STAGE2_JSON = ANALYZED_STAGE2_DIR / "valid_stage2.json"
TEST_STAGE1_JSON = ANALYZED_STAGE1_DIR / "test_stage1.json"
TEST_STAGE2_JSON = ANALYZED_STAGE2_DIR / "test_stage2.json"

# Registry files.
BUG_TYPE_REGISTRY_JSON = ANALYZED_REGISTRIES_DIR / "bug_type_registry.json"
VALID_TYPE_REGISTRY_JSON = ANALYZED_REGISTRIES_DIR / "valid_type_registry.json"
TEST_TYPE_REGISTRY_JSON = ANALYZED_REGISTRIES_DIR / "test_type_registry.json"

# Evaluation sample indexes.
VALID_SAMPLES_CSV = ANALYZED_EVAL_SAMPLES_DIR / "valid.csv"
TEST_SAMPLES_CSV = ANALYZED_EVAL_SAMPLES_DIR / "test.csv"

# Reports and plots.
KEYWORD_MATCHING_SUMMARY = ANALYZED_REPORTS_DIR / "keyword_matching_summary.json"
DATASET_STATISTICS_REPORT = ANALYZED_REPORTS_DIR / "dataset_statistics.json"
BUG_DISTRIBUTION_PLOT = DATASETS_VIS_DIR / "bug_distribution.png"

# Final dataset files.
QUANTUM_BUGS_TRAIN_JSON = DATASETS_DIR / "quantum_bugs_train.json"
QUANTUM_BUGS_VALID_JSON = DATASETS_DIR / "quantum_bugs_valid.json"
QUANTUM_BUGS_TEST_JSON = DATASETS_DIR / "quantum_bugs_test.json"
DATASET_METADATA_JSON = DATASETS_DIR / "metadata.json"

# =============================================================================
# Path helpers
# =============================================================================

def parse_project_name(project_name: str) -> Tuple[str, str]:
    """Parse an ``owner/repo`` project name."""
    if '/' not in project_name:
        raise ValueError(f"Invalid project name format: {project_name}")
    
    parts = project_name.split('/')
    if len(parts) != 2:
        raise ValueError(f"Invalid project name format: {project_name}")
    
    return parts[0], parts[1]

def get_project_identifier(owner: str, repo: str, version_hash: str) -> str:
    """Build a stable project identifier."""
    return f"{owner}-{repo}-{version_hash[:8]}"

def get_commit_source_path(owner: str, repo: str, commit_sha: str, file_type: str) -> Path:
    """Return the per-commit source path for one file type."""
    return SOURCE_CODE_DIR / owner / repo / commit_sha / file_type

def get_full_project_path(owner: str, repo: str, commit_sha: str, version_hash: str) -> Path:
    """Return the full-project checkout path for one version."""
    hash_short = version_hash[:8] if len(version_hash) >= 8 else version_hash
    return SOURCE_CODE_DIR / owner / repo / commit_sha / hash_short

def get_bug_download_status_path(task_id: str = "main") -> Path:
    """Return the bug-download status file path."""
    return RAW_STATUS_DIR / f"bug_download_status_{task_id}.json"

def get_commit_cache_path(owner: str, repo: str, commit_sha: str) -> Path:
    """Return the AST cache path for one commit."""
    safe_owner = str(owner).replace('/', '_')
    safe_repo = str(repo).replace('/', '_')
    return CACHE_COMMITS_DIR / f"{safe_owner}__{safe_repo}__{commit_sha}.json"

# =============================================================================
# Project metadata helpers
# =============================================================================

def get_project_info_from_csv_row(row: Dict[str, Any]) -> Dict[str, str]:
    """Extract normalized project information from one CSV row."""
    project_name = row['project_name']
    version = row['version']
    pr_url = row.get('pr_url', '')
    
    owner, repo = parse_project_name(project_name)
    pr_number = pr_url.split('/')[-1] if '/pull/' in pr_url else 'unknown'
    
    return {
        'project_name': project_name,
        'owner': owner,
        'repo': repo,
        'version': version,
        'version_short': version[:8],
        'pr_number': pr_number,
        'pr_url': pr_url,
        'project_id': get_project_identifier(owner, repo, version)
    }

def extract_pr_number_from_url(pr_url: str) -> str:
    """Extract the PR number from a PR URL."""
    if '/pull/' in pr_url:
        return pr_url.split('/')[-1]
    else:
        return 'unknown'

# =============================================================================
# Directory helpers
# =============================================================================

def ensure_directory_exists(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)

def create_unified_structure() -> None:
    """Create the full project directory layout."""
    directories = [
        RAW_DATA_DIR,
        SOURCE_CODE_DIR,
        EXTRACTED_DIR,
        ANALYZED_DIR,
        DATASETS_DIR,
        CACHE_DIR,
        LOG_DIR,
        RAW_STATUS_DIR,
        ANALYZED_STAGE1_DIR,
        ANALYZED_STAGE2_DIR,
        ANALYZED_REGISTRIES_DIR,
        ANALYZED_ERRORS_DIR,
        ANALYZED_EVAL_SAMPLES_DIR,
        ANALYZED_REPORTS_DIR,
        DATASETS_VIS_DIR,
        CACHE_COMMITS_DIR,
    ]
    
    for directory in directories:
        ensure_directory_exists(directory)


def ensure_repo_collection_structure(framework: Optional[str] = None) -> None:
    """Create only the directories required by repository collection."""
    directories = [
        LOG_DIR,
        get_framework_data_dir(framework),
    ]
    for directory in directories:
        ensure_directory_exists(directory)


def ensure_commit_collection_structure(framework: Optional[str] = None) -> None:
    """Create only the directories required by commit collection."""
    framework_raw_dir = get_framework_data_dir(framework)
    directories = [
        LOG_DIR,
        framework_raw_dir,
        framework_raw_dir / "status",
    ]
    for directory in directories:
        ensure_directory_exists(directory)


def ensure_bug_download_structure(framework: Optional[str] = None) -> None:
    """Create only the directories required by bug file downloading."""
    framework_raw_dir = get_framework_data_dir(framework)
    directories = [
        LOG_DIR,
        framework_raw_dir,
        framework_raw_dir / "status",
        get_framework_source_dir(framework),
    ]
    for directory in directories:
        ensure_directory_exists(directory)


def ensure_method_extraction_structure(framework: Optional[str] = None) -> None:
    """Create only the directories required by method extraction."""
    fw = framework or ACTIVE_FRAMEWORK
    directories = [
        LOG_DIR,
        get_framework_data_dir(fw),
        get_framework_data_dir(fw) / "status",
        get_framework_source_dir(fw),
        EXTRACTED_DIR / fw,
        CACHE_DIR,
        CACHE_COMMITS_DIR,
    ]
    for directory in directories:
        ensure_directory_exists(directory)

# =============================================================================
# Validation and initialization
# =============================================================================

def validate_configuration() -> bool:
    """Validate that the core configuration is usable."""
    try:
        if not PROJECT_ROOT.exists():
            print(f"Project root does not exist: {PROJECT_ROOT}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

def initialize_data_structure():
    """Create the full project directory layout."""
    if validate_configuration():
        create_unified_structure()
        import os
        if not os.getenv('QUIET'):
            print("Data structure initialized.")
            print("  01_raw/")
            print("  02_source/")
            print("  03_extracted/")
            print("  04_analyzed/")
            print("  05_datasets/")
        return True
    else:
        print("Data structure initialization failed.")
        return False


# =============================================================================
# Compatibility aliases
# =============================================================================

METADATA_DIR = RAW_DATA_DIR
ANALYSIS_RESULTS_DIR = ANALYZED_DIR
PROCESSED_DATASETS_DIR = DATASETS_DIR
REPORTS_DIR = ANALYZED_REPORTS_DIR

DATA_DIR = DATA_ROOT
ACQUISITION_OUTPUT_DIR = RAW_DATA_DIR

# =============================================================================
# LLM outputs
# =============================================================================

LLM_OUTPUT_DIR = PROJECT_ROOT / "llm" / "output"

BUG_DETECTION_DATASET = DATASETS_DIR / "dataset_parent_all.json"

BUG_REPAIR_DATASET = DATASETS_DIR / "dataset_parent_all.json"

BUG_REPAIR_AUX_DATASET = EXTRACTED_DIR / "method_level_single_merged.json"

if __name__ == "__main__":
    initialize_data_structure() 
