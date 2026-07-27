"""AST helpers for method-level bug extraction."""

import ast
import copy
import gc
import sys
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, List, Tuple

_CURRENT_FRAMEWORK: Optional[str] = None

def set_current_framework(framework: str):
    """Set the active framework used for path generation."""
    global _CURRENT_FRAMEWORK
    _CURRENT_FRAMEWORK = framework

def get_current_framework() -> Optional[str]:
    """Return the active framework name."""
    return _CURRENT_FRAMEWORK

# Compatibility for ast.unparse, which is available in Python 3.9+
try:
    from ast import unparse
except ImportError:
    try:
        from astunparse import unparse
    except ImportError:
        print("Error: 'astunparse' is required for Python < 3.9. Please run 'pip install astunparse'", file=sys.stderr)
        sys.exit(1)

try:
    import config
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    import config


_ast_cache = {}
_cache_hit_count = 0
_cache_miss_count = 0


def get_cache_stats():
    """Return a human-readable AST cache hit-rate summary."""
    global _cache_hit_count, _cache_miss_count
    total = _cache_hit_count + _cache_miss_count
    hit_rate = _cache_hit_count / total * 100 if total > 0 else 0
    return f"AST cache hit rate: {hit_rate:.1f}% ({_cache_hit_count}/{total})"

class GlobalBodyRemover(ast.NodeTransformer):
    """Remove functions, classes, and imports from a module AST."""
    def __init__(self):
        self._node_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)

    def visit_FunctionDef(self, node):
        return None

    def visit_AsyncFunctionDef(self, node):
        return None

    def visit_ClassDef(self, node):
        return None

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


class ImportRemover(ast.NodeTransformer):
    """Remove import statements from a function AST."""
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def fast_read_file(file_path):
    """Read a text file with tolerant UTF-8 decoding."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""


@lru_cache(maxsize=1024)
def cached_ast_parse(code_hash):
    """Check the external AST cache before parsing again."""
    global _cache_hit_count, _cache_miss_count
    if code_hash in _ast_cache:
        _cache_hit_count += 1
        return _ast_cache[code_hash]
    _cache_miss_count += 1
    return None


def smart_ast_parse(code_content):
    """Parse source code into an AST with cache support."""
    global _cache_hit_count, _cache_miss_count
    
    code_hash = hash(code_content)
    cached_result = cached_ast_parse(code_hash)
    if cached_result is not None:
        return cached_result
    
    try:
        tree = ast.parse(code_content)
        _ast_cache[code_hash] = tree
        
        if len(_ast_cache) > 2048:
            keys_to_remove = list(_ast_cache.keys())[:1024]
            for key in keys_to_remove:
                del _ast_cache[key]
            gc.collect()
        
        return tree
    except (SyntaxError, ValueError):
        return None


def extract_global_code(source_code: str) -> str:
    """Extract top-level source code excluding functions, classes, and imports."""
    try:
        tree = smart_ast_parse(source_code)
        if tree is None:
            return source_code
            
        source_lines = source_code.splitlines()
        
        lines_in_structures = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    for line_num in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                        lines_in_structures.add(line_num)
        
        global_lines = [
            line for i, line in enumerate(source_lines, start=1)
            if i not in lines_in_structures
        ]
        
        return '\n'.join(global_lines)
    except Exception:
        return source_code


def unparse_node_without_docstring(node):
    """Unparse an AST node, stripping docstrings from modules and functions."""
    try:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)) and ast.get_docstring(node) is not None:
            node_copy = copy.copy(node)
            node_copy.body = node.body[1:]
            return unparse(node_copy)
        return unparse(node)
    except Exception:
        return ""


def compare_asts_ignoring_trivia(node1, node2):
    """Compare two AST nodes while ignoring docstrings and blank lines."""
    try:
        code1 = unparse_node_without_docstring(node1)
        code2 = unparse_node_without_docstring(node2)
        code1_lines = [line.strip() for line in code1.splitlines() if line.strip()]
        code2_lines = [line.strip() for line in code2.splitlines() if line.strip()]
        return code1_lines != code2_lines
    except Exception:
        return True


def collect_top_level_functions(tree: ast.AST) -> Dict[str, ast.AST]:
    """Return top-level functions and direct class methods."""
    functions: Dict[str, ast.AST] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions[f"{class_name}.{item.name}"] = item
    return functions


def collect_top_level_scope_lines(source_code: str) -> Dict[str, set[int]]:
    """Return source-line sets for top-level scopes and the global scope."""
    tree = smart_ast_parse(source_code)
    if tree is None:
        return {}

    scopes: Dict[str, set[int]] = {}
    for func_name, node in collect_top_level_functions(tree).items():
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            scopes[func_name] = set(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    source_lines = source_code.splitlines()
    lines_in_structures = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                lines_in_structures.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    global_lines = {
        i for i, _line in enumerate(source_lines, start=1)
        if i not in lines_in_structures
    }
    if global_lines:
        scopes["<global>"] = global_lines

    return scopes


def extract_functions_from_source(source_code: str, file_path: str = "") -> List[Dict[str, str]]:
    """Extract top-level functions and direct class methods from source code."""
    try:
        tree = smart_ast_parse(source_code)
        if tree is None:
            return []
    except Exception:
        return []
    
    functions = []
    
    try:
        for func_name, node in collect_top_level_functions(tree).items():
            try:
                func_source = ast.get_source_segment(source_code, node)
                if func_source:
                    functions.append({
                        "name": func_name,
                        "source": func_source
                    })
            except Exception:
                try:
                    func_source = unparse(node)
                    if func_source:
                        functions.append({
                            "name": func_name,
                            "source": func_source
                        })
                except Exception:
                    continue
    except Exception:
        return []
    
    return functions


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


def get_commit_source_path(owner: str, repo_name: str, commit_sha: str, file_type: str, framework: Optional[str] = None) -> Path:
    """Build the source path under ``data/02_source`` for one commit snapshot."""
    try:
        base_dir = config.SOURCE_CODE_DIR
    except Exception:
        project_root = Path(__file__).resolve().parent.parent.parent
        base_dir = project_root / "data" / "02_source"
    
    fw = framework or _CURRENT_FRAMEWORK
    if fw:
        return Path(base_dir) / fw / owner / repo_name / commit_sha / file_type
    else:
        return Path(base_dir) / owner / repo_name / commit_sha / file_type


def process_single_commit(commit_info, commit_info_dict=None):
    """Extract all changed method-level samples for a single commit."""
    all_modified_samples = []
    total_modified_count = 0
    
    global_body_remover = GlobalBodyRemover()

    owner = commit_info.get('owner')
    repository = commit_info.get('repository')
    commit_url = commit_info.get('commit_url', '')
    commit_sha = commit_info.get('commit_sha')
    
    commit_title = commit_info.get('commit_message', '')
    commit_body = ""
    
    if not commit_title and commit_info_dict:
        key = f"{owner}/{repository.split('/')[-1]}"
        if key in commit_info_dict:
            for detail in commit_info_dict[key]:
                if detail.get('commit_sha') == commit_sha:
                    commit_title = detail.get('title', '')
                    commit_body = detail.get('body', '')
                    break
    
    repo_name = repository.split('/')[-1]
    
    if not commit_sha:
        logging.debug(f"Skipping: no commit_sha for {repository}")
        return [], 0
    buggy_base_path = get_commit_source_path(owner, repo_name, commit_sha, 'parent')
    fixed_base_path = get_commit_source_path(owner, repo_name, commit_sha, 'commit')
    
    if not (buggy_base_path.exists() and fixed_base_path.exists()):
        logging.debug(f"Skipping commit {commit_sha}: parent or commit path not found")
        logging.debug(f"  parent exists: {buggy_base_path.exists()}")
        logging.debug(f"  commit exists: {fixed_base_path.exists()}")
        return [], 0
    
    try:
        py_files = list(buggy_base_path.rglob("*.py"))
        if not py_files:
            return [], 0
    except Exception:
        return [], 0

    # The dynamic root calculation is fragile and has been removed.
    # We now consistently use buggy_base_path as the root for normalization,
    # which aligns with how fixed_file_path is calculated and assumes analysis.json
    # paths are relative to the checkout root inside 'buggy'.
    for idx, buggy_file_path in enumerate(py_files):
        try:
            file_relative_path = buggy_file_path.relative_to(buggy_base_path).as_posix()
            normalized_script_path = f"{owner}/{repo_name}/{commit_sha}/{file_relative_path}"

            relative_path_for_fixed = buggy_file_path.relative_to(buggy_base_path)
            fixed_file_path = fixed_base_path / relative_path_for_fixed
            
            if not fixed_file_path.exists():
                continue

            buggy_code_full = fast_read_file(buggy_file_path)
            fixed_code_full = fast_read_file(fixed_file_path)
            
            if not buggy_code_full or not fixed_code_full:
                continue
            
            buggy_tree = smart_ast_parse(buggy_code_full)
            fixed_tree = smart_ast_parse(fixed_code_full)
            
            if buggy_tree is None or fixed_tree is None:
                continue

            buggy_functions_ast = collect_top_level_functions(buggy_tree)
            fixed_functions_ast = collect_top_level_functions(fixed_tree)

            common_functions = set(buggy_functions_ast.keys()) & set(fixed_functions_ast.keys())
            
            added_functions = set(fixed_functions_ast.keys()) - set(buggy_functions_ast.keys())
            deleted_functions = set(buggy_functions_ast.keys()) - set(fixed_functions_ast.keys())
            
            modified_function_count = 0
            file_samples_start_idx = len(all_modified_samples)
            
            for func_name in common_functions:
                buggy_func_ast = buggy_functions_ast[func_name]
                fixed_func_ast = fixed_functions_ast[func_name]
                
                try:
                    buggy_src = ast.get_source_segment(buggy_code_full, buggy_func_ast)
                    fixed_src = ast.get_source_segment(fixed_code_full, fixed_func_ast)
                    
                    if not buggy_src or not fixed_src:
                        continue
                except Exception:
                    continue

                try:
                    cleaned_buggy_func_ast = ImportRemover().visit(copy.deepcopy(buggy_func_ast))
                    cleaned_fixed_func_ast = ImportRemover().visit(copy.deepcopy(fixed_func_ast))
                except Exception:
                    cleaned_buggy_func_ast = buggy_func_ast
                    cleaned_fixed_func_ast = fixed_func_ast

                if compare_asts_ignoring_trivia(cleaned_buggy_func_ast, cleaned_fixed_func_ast):                                                                                  
                    modified_function_count += 1
                    sample_base = {
                        'project_name': repository,
                        'repository': repository,
                        'commit_sha': commit_sha,
                        'commit_url': commit_url,
                        'commit_head': commit_title,
                        'commit_body': commit_body,
                        'file_path': normalized_script_path,
                        'owner': owner,
                        'repo_name': repo_name,
                        'relative_file_path': file_relative_path,
                        'function_name': func_name,
                        'buggy_code': buggy_src,
                        'fixed_code': fixed_src,
                    }
                    all_modified_samples.append(sample_base)
                    total_modified_count += 1

            global_code_changed = False
            try:
                buggy_globals_tree = global_body_remover.visit(copy.copy(buggy_tree))
                fixed_globals_tree = global_body_remover.visit(copy.copy(fixed_tree))

                buggy_global_src = unparse(buggy_globals_tree).strip()
                fixed_global_src = unparse(fixed_globals_tree).strip()
                
                if compare_asts_ignoring_trivia(buggy_globals_tree, fixed_globals_tree):
                    global_code_changed = True
                    global_sample_base = {
                        'project_name': repository,
                        'repository': repository,
                        'commit_sha': commit_sha,
                        'commit_url': commit_url,
                        'commit_head': commit_title,
                        'commit_body': commit_body,
                        'file_path': normalized_script_path,
                        'owner': owner,
                        'repo_name': repo_name,
                        'relative_file_path': file_relative_path,
                        'function_name': '<global>',
                        'buggy_code': buggy_global_src,
                        'fixed_code': fixed_global_src,
                    }
                    all_modified_samples.append(global_sample_base)
                    total_modified_count += 1
            except Exception as e:
                logging.debug(f"Failed to process global code block ({normalized_script_path}): {e}")
            
            file_total_changes = (
                modified_function_count +
                len(added_functions) +
                len(deleted_functions) +
                (1 if global_code_changed else 0)
            )
            
            for i in range(file_samples_start_idx, len(all_modified_samples)):
                all_modified_samples[i]['file_total_changes'] = file_total_changes

        except Exception as e:
            logging.debug(f"File processing failed ({buggy_file_path}): {e}")
            continue
        
        if len(py_files) > 50 and (idx + 1) % 10 == 0:
            gc.collect()

    return all_modified_samples, total_modified_count


__all__ = [
    'GlobalBodyRemover',
    'ImportRemover',
    'fast_read_file',
    'smart_ast_parse',
    'collect_top_level_functions',
    'collect_top_level_scope_lines',
    'extract_global_code',
    'unparse_node_without_docstring',
    'compare_asts_ignoring_trivia',
    'extract_functions_from_source',
    'parse_commit_url',
    'get_commit_source_path',
    'process_single_commit',
    'get_cache_stats',
]
