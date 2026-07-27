#!/usr/bin/env python3
"""
Batch-download repositories used to build negative samples from the positive-sample result file.

Expected directory structure:
    data/02_source/{framework}/{owner}/{repo}/project/

This script only:
  1) Reads samples with change_category="Bug Fix" from the positive result file
  2) Extracts repository and framework information
  3) Clones each unique repository into the structure above, skipping existing ones
  4) Does not perform checkout and keeps the default HEAD; checkout happens later when filtering by parent_sha

For parent negative-sample mining, use:
build_parent_random_func_negatives.py / build_parent_all_func_negatives.py
"""

import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import shutil

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Supported framework directories.
FRAMEWORKS = ["qiskit", "cirq", "pennylane"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class BugSampleMeta:
    sample_id: str
    repository: str
    commit_sha: str
    framework: str = "qiskit"


def load_samples_from_final(final_file: Path) -> List[BugSampleMeta]:
    """
    Load all Bug Fix sample metadata from the positive result file.
    """
    with final_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[BugSampleMeta] = []
    for r in data.get("results", []):
        # Must be a Bug Fix sample.
        if r.get("change_category") != "Bug Fix":
            continue
        
        # Support both id and sample_id field names.
        sid = str(r.get("sample_id") or r.get("id", ""))
        repository = r.get("repository", "")
        commit_sha = r.get("commit_sha", "")
        framework = r.get("framework", "qiskit")
        
        if sid and repository:
            samples.append(
                BugSampleMeta(
                    sample_id=sid,
                    repository=repository,
                    commit_sha=commit_sha,
                    framework=framework,
                )
            )
    
    logging.info("Loaded %d Bug Fix samples from %s", len(samples), final_file.name)
    return samples


def find_existing_repo_dir(repo_root_dir: Path, repository: str) -> Optional[Path]:
    """
    Find an existing repository directory.
    Directory structure: data/02_source/{framework}/{owner}/{repo}/project/
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


def clone_if_missing(
    repo_root_dir: Path,
    repository: str,
    framework: str = "qiskit",
    done_file: Optional[Path] = None,
    done_pairs: Optional[Set[Tuple[str, str]]] = None,
    write_lock: Optional[object] = None,
    reset: bool = False,
) -> None:
    """
    Clone the repository if it is missing.
    Directory structure: data/02_source/{framework}/{owner}/{repo}/project/
    No checkout is performed; the default HEAD is kept.
    
    Args:
        repo_root_dir: Repository root directory.
        repository: Repository name in owner/repo format.
        framework: Framework name (qiskit/cirq/pennylane).
        done_file: Status file path.
        done_pairs: Completed (repository, framework) pairs.
        write_lock: Thread lock used to protect concurrent writes to the status file.
    """
    owner, name = repository.split("/", 1)
    pair = (repository, framework)
    
    # Target path: data/02_source/{framework}/{owner}/{repo}/project/
    repo_dir = repo_root_dir / framework / owner / name / "project"
    
    # Check whether the target directory already exists.
    if repo_dir.is_dir() and (repo_dir / ".git").is_dir() and not reset:
        # Backfill status entry.
        if done_file is not None and done_pairs is not None and write_lock is not None:
            with write_lock:
                if pair not in done_pairs:
                    with done_file.open("a", encoding="utf-8") as f:
                        f.write(f"{repository},{framework}\n")
                    done_pairs.add(pair)
        return

    # If reset was requested, remove the existing directory.
    if reset and repo_dir.exists():
        logging.info("Reset requested: removing %s", repo_dir)
        try:
            shutil.rmtree(repo_dir)
        except Exception as e:
            logging.warning("Failed to remove %s during reset: %s", repo_dir, e)
    
    # Check whether the target directory already exists, possibly without .git.
    if repo_dir.exists() and not reset:
        logging.debug("Directory already exists (possibly incomplete): %s", repo_dir)
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{repository}.git"
    logging.info("Cloning repo %s (%s) to %s", clone_url, framework, repo_dir)
    subprocess.run(
        ["git", "clone", clone_url, str(repo_dir)],
        check=True,
    )

    if done_file is not None and done_pairs is not None and write_lock is not None:
        with write_lock:
            if pair not in done_pairs:
                with done_file.open("a", encoding="utf-8") as f:
                    f.write(f"{repository},{framework}\n")
                done_pairs.add(pair)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download parent repositories used for negative-sample construction (filtered by stage1 Bug Fix + method_level_single)",
    )
    project_root = PROJECT_ROOT

    parser.add_argument(
        "--final-file",
        type=Path,
        default=project_root / "data" / "04_analyzed" / "final" / "bugfix_positive_samples.json",
        help="Path to the positive result file containing repository, framework, change_category, and related fields",
    )
    parser.add_argument(
        "--repo-root-dir",
        type=Path,
        default=project_root / "data" / "02_source",
        help="Root directory for downloaded repositories, using the {framework}/{owner}/{repo}/project layout",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of threads for parallel cloning (default: 8)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="If set, remove existing local directories and force re-cloning, ignoring the status file",
    )
    args = parser.parse_args()

    args.repo_root_dir.mkdir(parents=True, exist_ok=True)

    # Status file path.
    status_dir = project_root / "data" / "01_raw" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    state_file = status_dir / "download_parent_repos_done.txt"

    logging.info("Final file:   %s", args.final_file)
    logging.info("Repo root dir:%s", args.repo_root_dir)

    # Load samples from the positive result file.
    meta_samples = load_samples_from_final(args.final_file)

    # Deduplicate by (repository, framework). The same repository under different frameworks is downloaded separately.
    repo_framework_pairs: Set[Tuple[str, str]] = set()
    for s in meta_samples:
        repo_framework_pairs.add((s.repository, s.framework))
    
    logging.info("Unique (repository, framework) pairs: %d (from %d samples)", len(repo_framework_pairs), len(meta_samples))

    # Filter completed (repository, framework) pairs using the status file to support resuming.
    # Status file format: repository,framework (one entry per line).
    done_pairs: Set[Tuple[str, str]] = set()
    if state_file.exists() and not args.reset:
        try:
            with state_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "," in line:
                        repo, fw = line.split(",", 1)
                        done_pairs.add((repo.strip(), fw.strip()))
                    else:
                        # Backward compatibility: if only repository is stored, assume all frameworks are completed.
                        for fw in FRAMEWORKS:
                            done_pairs.add((line, fw))
            logging.info("Loaded %d completed (repo, framework) pairs from state file", len(done_pairs))
        except OSError:
            pass
    
    if args.reset:
        pending_pairs = sorted(repo_framework_pairs)
        logging.info("Reset requested: will re-clone %d (repo, framework) pairs", len(pending_pairs))
    else:
        # Verify that repositories recorded in the status file actually exist.
        verified_done: Set[Tuple[str, str]] = set()
        for repo, fw in done_pairs:
            try:
                owner, name = repo.split("/", 1)
                repo_dir = args.repo_root_dir / fw / owner / name / "project"
                if repo_dir.is_dir() and (repo_dir / ".git").is_dir():
                    verified_done.add((repo, fw))
            except ValueError:
                continue
        
        pending_pairs = sorted(repo_framework_pairs - verified_done)
        logging.info("Pending (repo, framework) pairs to download: %d", len(pending_pairs))

    # Clone repositories in parallel.
    from threading import Lock
    write_lock = Lock()
    done_pairs_lock = set(done_pairs)  # Used for concurrent checks.
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for repo, framework in pending_pairs:
            fut = executor.submit(
                clone_if_missing, args.repo_root_dir, repo, framework, state_file, done_pairs_lock, write_lock, args.reset
            )
            futures[fut] = (repo, framework)

        # Show progress with a progress bar when available.
        if tqdm:
            pbar = tqdm(total=len(pending_pairs), desc="Cloning repos", unit="repo", mininterval=1.0)
        else:
            pbar = None

        for i, fut in enumerate(as_completed(futures), start=1):
            repo, framework = futures[fut]
            try:
                fut.result()
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"{repo[:30]}...")
                else:
                    logging.info(
                        "[%d/%d] done: %s (%s)",
                        i,
                        len(pending_pairs),
                        repo,
                        framework,
                    )
            except subprocess.CalledProcessError as e:
                if pbar:
                    pbar.update(1)
                logging.warning("Git clone failed for %s (%s): %s", repo, framework, e)
                continue

        if pbar:
            pbar.close()

    logging.info("All candidate repositories processed.")


if __name__ == "__main__":
    main()
