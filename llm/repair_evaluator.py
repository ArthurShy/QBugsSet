"""
Search/replace-block-based bug repair evaluator.

Input is a full buggy file. The model outputs SEARCH/REPLACE edit blocks in
code fences. The evaluator parses those edits, applies them locally, and uses
an internal diff only for metric computation.
"""

import difflib
import ast
import copy
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from api_clients import BaseClient
from config import BUG_REPAIR_AUX_DATASET
from data_preprocessing.util.ast_method_extractor import collect_top_level_scope_lines, smart_ast_parse


logger = logging.getLogger(__name__)

CHECKPOINT_INTERVAL = 50
RESULT_SCHEMA_VERSION = 14
UNIFIED_DIFF_CONTEXT_LINES = 3


@dataclass
class ParsedPatch:
    old_path: str
    new_path: str
    hunks: List[Dict[str, Any]]


@dataclass
class SearchReplaceEdit:
    search: str
    replace: str


@dataclass
class RepairResult:
    sample_id: str
    success: bool
    generated_patch: Optional[str]
    reference_patch: Optional[str]
    buggy_code: Optional[str]
    exact_match: bool = False
    exact_fixed_match: bool = False
    format_valid: bool = False
    changed: bool = False
    apply_success: bool = False
    valid_edit: bool = False
    edit_parse_success: bool = False
    search_hit_success: bool = False
    unique_hit_success: bool = False
    non_empty_edit: bool = False
    edit_count: int = 0
    function_hit: bool = False
    function_jaccard: float = 0.0
    function_exact_match: bool = False
    line_hit: bool = False
    line_jaccard: float = 0.0
    line_exact_match: bool = False
    patch_similarity: float = 0.0
    fixed_code_similarity: float = 0.0
    syntax_valid: bool = False
    empty_line_only: bool = False
    response_time: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    repair_reason: Optional[str] = None
    parent_file_path: Optional[str] = None
    function_name: Optional[str] = None
    repository: Optional[str] = None
    commit_sha: Optional[str] = None
    framework: Optional[str] = None
    repair_level: Optional[str] = None
    thinking_content: Optional[str] = None
    applied_code: Optional[str] = None
    generated_edits: Optional[List[Dict[str, str]]] = None
    predicted_functions: Optional[List[str]] = None
    reference_functions: Optional[List[str]] = None
    predicted_changed_lines: Optional[List[int]] = None
    reference_changed_lines: Optional[List[int]] = None
    prompt_input_scope: Optional[str] = None


@dataclass
class RepairEvaluationReport:
    model_name: str
    dataset_name: str
    repair_level: str
    total_samples: int
    successful_samples: int
    failed_samples: int
    valid_repair_rate: float
    function_hit_rate: float
    avg_function_jaccard: float
    function_exact_match_rate: float
    line_hit_rate: float
    avg_line_jaccard: float
    line_exact_match_rate: float
    avg_patch_similarity: float
    exact_fixed_match_rate: float
    block_parse_rate: float
    search_hit_rate: float
    unique_hit_rate: float
    apply_rate: float
    non_empty_edit_rate: float
    avg_edit_count: float
    total_time: float
    avg_response_time: float
    results: List[RepairResult]
    token_stats: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "repair_level": self.repair_level,
            "prompt_input_scope": "file",
            "result_representation": ["raw_output", "parsed_output", "post_process", "evaluation"],
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "metrics": {
                "valid_repair_rate": self.valid_repair_rate,
                "function_hit_rate": self.function_hit_rate,
                "avg_function_jaccard": self.avg_function_jaccard,
                "function_exact_match_rate": self.function_exact_match_rate,
                "line_hit_rate": self.line_hit_rate,
                "avg_line_jaccard": self.avg_line_jaccard,
                "line_exact_match_rate": self.line_exact_match_rate,
                "avg_patch_similarity": self.avg_patch_similarity,
                "exact_fixed_match_rate": self.exact_fixed_match_rate,
            },
            "diagnostics": {
                "block_parse_rate": self.block_parse_rate,
                "edit_parse_rate": self.block_parse_rate,
                "search_hit_rate": self.search_hit_rate,
                "unique_hit_rate": self.unique_hit_rate,
                "apply_rate": self.apply_rate,
                "non_empty_edit_rate": self.non_empty_edit_rate,
                "avg_edit_count": self.avg_edit_count,
            },
            "token_usage": self.token_stats,
        }

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print(f"Repair Report - {self.model_name}")
        print("=" * 60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Repair level: {self.repair_level}")
        print(f"Total samples: {self.total_samples}")
        print(f"Successful generations: {self.successful_samples}")
        print(f"Failed generations: {self.failed_samples}")
        print("-" * 40)
        print(f"Valid repair rate: {self.valid_repair_rate:.4f}")
        print(f"Function hit rate: {self.function_hit_rate:.4f}")
        print(f"Avg function Jaccard: {self.avg_function_jaccard:.4f}")
        print(f"Function exact match rate: {self.function_exact_match_rate:.4f}")
        print(f"Line hit rate: {self.line_hit_rate:.4f}")
        print(f"Avg line Jaccard: {self.avg_line_jaccard:.4f}")
        print(f"Line exact match rate: {self.line_exact_match_rate:.4f}")
        print(f"Avg patch similarity: {self.avg_patch_similarity:.4f}")
        print(f"Exact fixed match rate: {self.exact_fixed_match_rate:.4f}")
        print("-" * 40)
        print(f"Block parse rate: {self.block_parse_rate:.4f}")
        print(f"Search hit rate: {self.search_hit_rate:.4f}")
        print(f"Unique hit rate: {self.unique_hit_rate:.4f}")
        print(f"Apply rate: {self.apply_rate:.4f}")
        print(f"Non-empty edit rate: {self.non_empty_edit_rate:.4f}")
        print(f"Avg edit count: {self.avg_edit_count:.4f}")
        print("-" * 40)
        print(f"Total time: {self.total_time:.2f}s")
        print(f"Avg response time: {self.avg_response_time:.2f}s")
        print("-" * 40)
        print("Token stats:")
        for key, value in self.token_stats.items():
            print(f"  {key}: {value:,}")
        print("=" * 60)


class RepairEvaluator:
    SYSTEM_PROMPT_EN = """You are an expert Python bug-fixing assistant.

Return SEARCH/REPLACE edits only.
Return only the final answer.

Every edit must use this exact format inside a ```python``` block:

<<<<<<< SEARCH
exact existing code
=======
replacement code
>>>>>>> REPLACE

Hard rules:
- The SEARCH block must exactly match existing code in the buggy file
- The SEARCH block must be copied verbatim as one contiguous chunk; never skip lines, rewrite comments, or paraphrase code inside SEARCH
- Each SEARCH block must match exactly one location
- Use the smallest possible edit set
- Prefer small local edits over whole-function rewrites
- Avoid replacing an entire function when a smaller SEARCH block can fix the bug
- Edit only one local region unless a second edit is absolutely necessary
- Do not refactor or modernize APIs
- Do not change imports, docstrings, type hints, comments, quote style, or formatting unless required for the fix
- Preserve indentation exactly
- Every SEARCH/REPLACE block must be fully closed with both ======= and >>>>>>> REPLACE
- Do not output any analysis, reasoning, comments, prefaces, follow-up corrections, or text outside the edit blocks
- Do not output anything before the first ```python``` block
- Do not output anything after the last ```python``` block
- Do not output a second attempt, correction, or revised answer
- Do not write words such as "Wait", "Correction", "Updated", "I found", or "Here is"
- Do not output JSON
- Do not output unified diff format
"""

    USER_PROMPT_FILE_EN = """
We are fixing a bug in the following Python file.

Repository: {repository}
File path: {file_path}

Generate the final SEARCH/REPLACE edits to fix it.

Every SEARCH/REPLACE edit must use this format:
1. The line <<<<<<< SEARCH
2. A contiguous chunk of existing code
3. The line =======
4. The replacement code
5. The line >>>>>>> REPLACE

Example:

```python
<<<<<<< SEARCH
def add(a, b):
    return a - b
=======
def add(a, b):
    return a + b
>>>>>>> REPLACE
```

Important rules:
- Wrap every edit block in ```python```
- SEARCH must be copied verbatim from the buggy file
- SEARCH must be one continuous exact excerpt from the file; do not omit lines in the middle
- SEARCH must be unique in the buggy file
- Multi-line SEARCH and REPLACE blocks are allowed
- Prefer the smallest exact SEARCH block that uniquely identifies the bug
- Do not rewrite a whole function unless that is strictly necessary
- Preserve indentation exactly
- Finish every block completely: include <<<<<<< SEARCH, =======, and >>>>>>> REPLACE
- Output only the final edit blocks
- Do not include analysis, reasoning, comments, file summaries, or explanations
- Do not include any text before the first edit block
- Do not include any text between edit blocks except the blocks themselves
- Do not include any text after the last edit block
- Do not output a correction, second attempt, or revised block
- Do not output file paths or directory headers
- Do not output JSON
- If multiple edits are needed, output all of them once in the same final answer
- If you are unsure, still output only the final SEARCH/REPLACE edit blocks

Buggy file:
`````python
{buggy_code}
`````
"""

    HUNK_RE = re.compile(
        r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
    )

    def __init__(
        self,
        client: BaseClient,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        prompt_language: str = "en",
        prompt_version: int = 1,
        repair_level: str = "function",
        max_workers: int = 0,
    ):
        self.client = client
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.prompt_language = "en"
        self.prompt_version = prompt_version
        self.repair_level = repair_level
        self.max_workers = max_workers
        self._display_name = getattr(client, "display_name", None) or client.model_name
        self._repair_aux_lookup: Optional[Dict[str, Dict[str, Any]]] = None

        if (prompt_language or "en").lower() != "en":
            logger.info(
                "Repair evaluation uses English prompt only; ignoring prompt_language=%s",
                prompt_language,
            )

    def _normalize_text(self, text: str) -> str:
        return "\n".join(text.strip("\n").splitlines())

    def _make_reference_patch(self, sample: Dict[str, Any], buggy_code: str, fixed_code: str) -> str:
        file_path = sample.get("parent_file_path") or "target.py"
        diff_lines = difflib.unified_diff(
            buggy_code.splitlines(),
            fixed_code.splitlines(),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=UNIFIED_DIFF_CONTEXT_LINES,
            lineterm="",
        )
        return "\n".join(diff_lines).strip()

    def _make_generated_patch(self, sample: Dict[str, Any], buggy_code: str, applied_code: str) -> str:
        file_path = sample.get("parent_file_path") or "target.py"
        diff_lines = difflib.unified_diff(
            buggy_code.splitlines(),
            applied_code.splitlines(),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=UNIFIED_DIFF_CONTEXT_LINES,
            lineterm="",
        )
        return "\n".join(diff_lines).strip()

    def _similarity(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _has_valid_python_syntax(self, source_code: str) -> bool:
        try:
            ast.parse(source_code)
            return True
        except SyntaxError:
            return False

    def _strip_docstrings_from_ast(self, node: ast.AST) -> ast.AST:
        def is_docstring_expr(stmt: ast.stmt) -> bool:
            if not isinstance(stmt, ast.Expr):
                return False
            value = stmt.value
            if isinstance(value, ast.Constant):
                return isinstance(value.value, str)
            return isinstance(value, ast.Str)

        tree = copy.deepcopy(node)
        for current in ast.walk(tree):
            if isinstance(current, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(current, "body", None)
                if body and is_docstring_expr(body[0]):
                    current.body = body[1:]
        return tree

    def _has_equivalent_python_ast(self, candidate_code: str, reference_code: str) -> bool:
        candidate_tree = smart_ast_parse(candidate_code)
        reference_tree = smart_ast_parse(reference_code)
        if candidate_tree is None or reference_tree is None:
            return False
        normalized_candidate = self._strip_docstrings_from_ast(candidate_tree)
        normalized_reference = self._strip_docstrings_from_ast(reference_tree)
        return ast.dump(normalized_candidate, include_attributes=False) == ast.dump(
            normalized_reference,
            include_attributes=False,
        )

    def _differs_only_by_empty_lines(self, updated_code: str, original_code: str) -> bool:
        if self._normalize_text(updated_code) == self._normalize_text(original_code):
            return False

        def remove_empty_lines(text: str) -> str:
            return "\n".join(line for line in text.splitlines() if line.strip() != "")

        return remove_empty_lines(updated_code) == remove_empty_lines(original_code)

    def _extract_python_blocks(self, content: str) -> List[str]:
        matches = re.findall(r"```(?:python)?\s*\n(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
        return [match.strip() for match in matches if match.strip()]

    def _parse_response(self, content: str) -> Optional[List[SearchReplaceEdit]]:
        blocks = self._extract_python_blocks(content)
        if not blocks:
            logger.warning("Unable to find SEARCH/REPLACE code blocks: %s...", content[:200])
            return None

        edits: List[SearchReplaceEdit] = []
        for block in blocks:
            lines = block.splitlines()
            idx = 0

            while idx < len(lines):
                stripped = lines[idx].strip()
                if not stripped:
                    idx += 1
                    continue
                if stripped.startswith("### "):
                    idx += 1
                    continue
                if stripped != "<<<<<<< SEARCH":
                    idx += 1
                    continue

                idx += 1
                search_lines: List[str] = []
                while idx < len(lines) and lines[idx].strip() != "=======":
                    search_lines.append(lines[idx])
                    idx += 1
                if idx >= len(lines):
                    logger.warning("Malformed SEARCH/REPLACE block, missing =======: %s...", block[:200])
                    return None

                idx += 1
                replace_lines: List[str] = []
                while idx < len(lines) and lines[idx].strip() != ">>>>>>> REPLACE":
                    replace_lines.append(lines[idx])
                    idx += 1
                if idx >= len(lines):
                    logger.warning("Malformed SEARCH/REPLACE block, missing >>>>>>> REPLACE: %s...", block[:200])
                    return None

                edits.append(
                    SearchReplaceEdit(
                        search="\n".join(search_lines),
                        replace="\n".join(replace_lines),
                    )
                )
                idx += 1

        if not edits:
            logger.warning("Unable to parse SEARCH/REPLACE edits: %s...", content[:200])
            return None
        return edits

    def _build_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str, str, str]:
        buggy_code = sample.get("buggy_file") or ""
        fixed_code = sample.get("fixed_file") or ""
        user_prompt = self.USER_PROMPT_FILE_EN.format(
            repository=sample.get("repository", ""),
            file_path=sample.get("parent_file_path", ""),
            buggy_code=buggy_code,
        )
        return self.SYSTEM_PROMPT_EN, user_prompt, buggy_code, fixed_code

    def _get_sample_id(self, sample: Dict[str, Any]) -> str:
        return str(sample.get("id") or sample.get("sample_id") or "unknown")

    def _parse_unified_diff(self, patch_text: str) -> ParsedPatch:
        lines = patch_text.splitlines()
        old_path = ""
        new_path = ""
        hunks: List[Dict[str, Any]] = []
        idx = 0

        while idx < len(lines):
            line = lines[idx]
            if line.startswith("--- "):
                old_path = line[4:].strip()
                idx += 1
                if idx >= len(lines) or not lines[idx].startswith("+++ "):
                    raise ValueError("Missing +++ line after --- line")
                new_path = lines[idx][4:].strip()
                idx += 1
                continue

            match = self.HUNK_RE.match(line)
            if not match:
                idx += 1
                continue

            old_start = int(match.group("old_start"))
            old_count = int(match.group("old_count") or "1")
            new_start = int(match.group("new_start"))
            new_count = int(match.group("new_count") or "1")
            idx += 1
            hunk_lines: List[str] = []

            while idx < len(lines):
                current = lines[idx]
                if current.startswith("@@") or current.startswith(("--- ", "+++ ")):
                    break
                if current.startswith("\\ No newline at end of file"):
                    idx += 1
                    continue
                if current and current[0] in {" ", "+", "-"}:
                    hunk_lines.append(current)
                    idx += 1
                    continue
                raise ValueError(f"Invalid hunk line: {current}")

            hunks.append(
                {
                    "old_start": old_start,
                    "old_count": old_count,
                    "new_start": new_start,
                    "new_count": new_count,
                    "lines": hunk_lines,
                }
            )

        if not old_path or not new_path or not hunks:
            raise ValueError("Patch missing file header or hunks")
        return ParsedPatch(old_path=old_path, new_path=new_path, hunks=hunks)

    def _patch_has_edits(self, parsed_patch: ParsedPatch) -> bool:
        for hunk in parsed_patch.hunks:
            for line in hunk["lines"]:
                if line.startswith(("+", "-")):
                    return True
        return False

    def _apply_search_replace_edits(
        self,
        source_code: str,
        edits: List[SearchReplaceEdit],
    ) -> Tuple[str, bool, bool, bool]:
        updated_code = source_code
        search_hit_success = True
        unique_hit_success = True
        non_empty_edit = False

        for edit in edits:
            if edit.search == edit.replace:
                continue
            occurrences = updated_code.count(edit.search)
            if occurrences == 0:
                search_hit_success = False
                raise ValueError("Search snippet not found in buggy file")
            if occurrences > 1:
                unique_hit_success = False
                raise ValueError("Search snippet matches multiple locations")
            updated_code = updated_code.replace(edit.search, edit.replace, 1)
            non_empty_edit = True

        return updated_code, search_hit_success, unique_hit_success, non_empty_edit

    def _extract_changed_lines(self, parsed_patch: ParsedPatch) -> Set[int]:
        changed: Set[int] = set()
        for hunk in parsed_patch.hunks:
            old_line = hunk["old_start"]
            for line in hunk["lines"]:
                prefix = line[0]
                if prefix == " ":
                    old_line += 1
                elif prefix == "-":
                    changed.add(old_line)
                    old_line += 1
                elif prefix == "+":
                    # Map insertions to the preceding old-file line when possible.
                    # This keeps boundary insertions attached to the region they
                    # extend, instead of incorrectly attributing them to the next
                    # function or block.
                    changed.add(max(old_line - 1, 1))
        return changed

    def _extract_function_spans(self, source_code: str) -> Dict[str, Set[int]]:
        return collect_top_level_scope_lines(source_code)

    def _locate_changed_functions(self, source_code: str, changed_lines: Set[int]) -> Set[str]:
        if not changed_lines:
            return set()

        spans = self._extract_function_spans(source_code)
        matched_functions: Set[str] = set()

        for name, line_span in spans.items():
            if changed_lines & line_span:
                matched_functions.add(name)
        return matched_functions

    def _compute_set_jaccard(self, predicted: Set[Any], reference: Set[Any]) -> float:
        if not predicted and not reference:
            return 1.0
        union = predicted | reference
        if not union:
            return 1.0
        return len(predicted & reference) / len(union)

    def _compute_set_hit(self, predicted: Set[Any], reference: Set[Any]) -> bool:
        if not predicted and not reference:
            return True
        return bool(predicted & reference)

    def _result_from_dict(self, data: Dict[str, Any]) -> RepairResult:
        parsed_output = data.get("parsed_output") or {}
        post_process = data.get("post_process") or {}
        evaluation = data.get("evaluation") or {}
        return RepairResult(
            sample_id=str(data.get("id") or data.get("sample_id") or "unknown"),
            success=data.get("success", False),
            generated_patch=data.get("generated_patch"),
            reference_patch=data.get("reference_patch"),
            buggy_code=post_process.get("original_file_content", data.get("buggy_code")),
            exact_match=evaluation.get("exact_match", data.get("exact_match", data.get("exact_patch_match", False))),
            exact_fixed_match=evaluation.get("exact_fixed_match", data.get("exact_fixed_match", False)),
            format_valid=parsed_output.get("format_valid", data.get("format_valid", False)),
            changed=post_process.get("changed", data.get("changed", False)),
            apply_success=post_process.get("apply_success", data.get("apply_success", False)),
            valid_edit=data.get(
                "valid_edit",
                evaluation.get(
                    "valid_edit",
                    data.get(
                        "valid_patch",
                        parsed_output.get("format_valid", data.get("format_valid", False))
                        and post_process.get("apply_success", data.get("apply_success", False))
                        and post_process.get("changed", data.get("changed", False)),
                    ),
                ),
            ),
            edit_parse_success=parsed_output.get("edit_parse_success", data.get("edit_parse_success", False)),
            search_hit_success=post_process.get("search_hit_success", data.get("search_hit_success", False)),
            unique_hit_success=post_process.get("unique_hit_success", data.get("unique_hit_success", False)),
            non_empty_edit=post_process.get("non_empty_edit", data.get("non_empty_edit", data.get("changed", False))),
            edit_count=int(parsed_output.get("edit_count", data.get("edit_count", 0)) or 0),
            function_hit=evaluation.get(
                "function_hit",
                data.get("function_hit", data.get("function_localization_correct", False)),
            ),
            function_jaccard=evaluation.get(
                "function_jaccard",
                data.get(
                    "function_jaccard",
                    1.0 if data.get("function_localization_correct", False) else 0.0,
                ),
            ),
            function_exact_match=evaluation.get(
                "function_exact_match",
                data.get("function_exact_match", data.get("function_localization_correct", False)),
            ),
            line_hit=evaluation.get("line_hit", data.get("line_hit", False)),
            line_jaccard=evaluation.get(
                "line_jaccard",
                data.get("line_jaccard", data.get("line_localization_f1", data.get("localization_f1", 0.0))),
            ),
            line_exact_match=evaluation.get("line_exact_match", data.get("line_exact_match", False)),
            patch_similarity=evaluation.get("patch_similarity", data.get("patch_similarity", data.get("similarity_to_reference", 0.0))),
            fixed_code_similarity=evaluation.get("fixed_code_similarity", data.get("fixed_code_similarity", 0.0)),
            syntax_valid=post_process.get("syntax_valid", data.get("syntax_valid", False)),
            empty_line_only=post_process.get("empty_line_only", data.get("empty_line_only", False)),
            response_time=data.get("response_time"),
            token_usage=data.get("token_usage"),
            finish_reason=data.get("finish_reason"),
            error=data.get("error"),
            raw_response=data.get("raw_output", data.get("raw_response")),
            system_prompt=data.get("system_prompt"),
            user_prompt=data.get("user_prompt"),
            repair_reason=data.get("repair_reason"),
            parent_file_path=data.get("parent_file_path"),
            function_name=data.get("function_name"),
            repository=data.get("repository"),
            commit_sha=data.get("commit_sha"),
            framework=data.get("framework"),
            repair_level=data.get("repair_level", self.repair_level),
            thinking_content=data.get("thinking_content"),
            applied_code=post_process.get("new_file_content", data.get("applied_code")),
            generated_edits=parsed_output.get("generated_edits", data.get("generated_edits")),
            predicted_functions=evaluation.get("predicted_functions", data.get("predicted_functions")),
            reference_functions=evaluation.get("reference_functions", data.get("reference_functions")),
            predicted_changed_lines=evaluation.get("predicted_changed_lines", data.get("predicted_changed_lines")),
            reference_changed_lines=evaluation.get("reference_changed_lines", data.get("reference_changed_lines")),
            prompt_input_scope=data.get("prompt_input_scope", "file"),
        )

    def _load_existing_results(self, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        model_dir = self.output_dir / self._display_name.replace("/", "_")
        output_path = self._find_existing_output_path(model_dir, dataset_name)
        if output_path is None:
            return {}
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("schema_version") != RESULT_SCHEMA_VERSION:
                logger.info("Ignoring old repair results with incompatible schema: %s", output_path)
                return {}

            existing: Dict[str, Dict[str, Any]] = {}
            for detail in data.get("details", []):
                sample_id = detail.get("id") or detail.get("sample_id")
                has_edits = detail.get("generated_edits") is not None
                if not has_edits and isinstance(detail.get("parsed_output"), dict):
                    has_edits = detail["parsed_output"].get("generated_edits") is not None
                if sample_id and has_edits:
                    existing[str(sample_id)] = detail
            logger.info("Loaded existing repair results: %d", len(existing))
            return existing
        except Exception as exc:
            logger.warning("Failed to load existing repair results: %s", exc)
            return {}

    def _find_existing_output_path(self, model_dir: Path, dataset_name: str) -> Optional[Path]:
        for path in self._get_output_candidates(model_dir, dataset_name):
            if path.exists():
                return path
        return None

    def _get_output_candidates(self, model_dir: Path, dataset_name: str) -> List[Path]:
        version_suffix = f"-v{self.prompt_version}" if self.prompt_version != 1 else ""
        model_suffix = self._display_name.replace("/", "_")
        return [
            model_dir / f"{dataset_name}_repair_{self.repair_level}_{model_suffix}_results-en{version_suffix}.json",
            model_dir / f"{dataset_name}_repair_{self.repair_level}_results-en{version_suffix}.json",
        ]

    def _get_output_path(self, model_dir: Path, dataset_name: str) -> Path:
        return self._get_output_candidates(model_dir, dataset_name)[0]

    def _load_samples(self, dataset_path: Path) -> List[Dict[str, Any]]:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        elif isinstance(data, dict) and "results" in data:
            samples = data["results"]
        elif isinstance(data, list):
            samples = data
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")

        if not samples or not isinstance(samples[0], dict) or "label" not in samples[0]:
            return samples

        aux_lookup = self._load_repair_aux_lookup()
        normalized_samples: List[Dict[str, Any]] = []
        skipped_negative = 0
        skipped_missing_pair = 0

        for sample in samples:
            if sample.get("label") != 1:
                skipped_negative += 1
                continue

            merged = self._merge_repair_sample(sample, aux_lookup.get(str(sample.get("id"))))
            if not merged.get("buggy_file") or not merged.get("fixed_file"):
                skipped_missing_pair += 1
                continue
            normalized_samples.append(merged)

        logger.info(
            "Normalized repair dataset from final benchmark: kept=%d, skipped_negative=%d, skipped_missing_pair=%d",
            len(normalized_samples),
            skipped_negative,
            skipped_missing_pair,
        )
        return normalized_samples

    def _load_repair_aux_lookup(self) -> Dict[str, Dict[str, Any]]:
        if self._repair_aux_lookup is not None:
            return self._repair_aux_lookup

        with open(BUG_REPAIR_AUX_DATASET, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
        self._repair_aux_lookup = {
            str(sample.get("id") or sample.get("sample_id")): sample
            for sample in samples
            if isinstance(sample, dict)
        }
        return self._repair_aux_lookup

    def _merge_repair_sample(
        self,
        sample: Dict[str, Any],
        aux_sample: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = dict(sample)
        if aux_sample:
            for key in [
                "buggy_file",
                "fixed_file",
                "buggy_code",
                "fixed_code",
                "function_name",
                "commit_message",
                "commit_url",
            ]:
                if merged.get(key) in (None, "") and aux_sample.get(key) not in (None, ""):
                    merged[key] = aux_sample[key]

        return merged

    def evaluate_sample(self, sample: Dict[str, Any]) -> RepairResult:
        sample_id = self._get_sample_id(sample)
        system_prompt, user_prompt, buggy_code, fixed_code = self._build_prompt(sample)

        if not buggy_code or not fixed_code:
            missing_field = "buggy_file" if not buggy_code else "fixed_file"
            return RepairResult(
                sample_id=sample_id,
                success=False,
                generated_patch=None,
                reference_patch=None,
                buggy_code=buggy_code,
                error=f"Missing required {missing_field} for file-level repair input",
                parent_file_path=sample.get("parent_file_path"),
                function_name=sample.get("function_name"),
                repository=sample.get("repository"),
                commit_sha=sample.get("commit_sha"),
                framework=sample.get("framework"),
                repair_level=self.repair_level,
                prompt_input_scope="file",
            )

        reference_patch = self._make_reference_patch(sample, buggy_code, fixed_code)
        if not reference_patch:
            return RepairResult(
                sample_id=sample_id,
                success=False,
                generated_patch=None,
                reference_patch="",
                buggy_code=buggy_code,
                error="Reference patch is empty",
                parent_file_path=sample.get("parent_file_path"),
                function_name=sample.get("function_name"),
                repository=sample.get("repository"),
                commit_sha=sample.get("commit_sha"),
                framework=sample.get("framework"),
                repair_level=self.repair_level,
                prompt_input_scope="file",
            )

        try:
            parsed_reference = self._parse_unified_diff(reference_patch)
            reference_changed_lines = self._extract_changed_lines(parsed_reference)
            reference_functions = self._locate_changed_functions(buggy_code, reference_changed_lines)
        except Exception as exc:
            return RepairResult(
                sample_id=sample_id,
                success=False,
                generated_patch=None,
                reference_patch=reference_patch,
                buggy_code=buggy_code,
                error=f"Failed to parse reference patch: {exc}",
                parent_file_path=sample.get("parent_file_path"),
                function_name=sample.get("function_name"),
                repository=sample.get("repository"),
                commit_sha=sample.get("commit_sha"),
                framework=sample.get("framework"),
                repair_level=self.repair_level,
                prompt_input_scope="file",
            )

        response = self.client.complete(user_prompt, system_prompt)
        generated_edits = None
        if response.success:
            generated_edits = self._parse_response(response.content)

        if response is None or not response.success:
            return RepairResult(
                sample_id=sample_id,
                success=False,
                generated_patch=None,
                reference_patch=reference_patch,
                buggy_code=buggy_code,
                error=response.error,
                raw_response=response.content,
                response_time=response.response_time,
                token_usage=response.token_usage,
                finish_reason=response.finish_reason,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parent_file_path=sample.get("parent_file_path"),
                function_name=sample.get("function_name"),
                repository=sample.get("repository"),
                commit_sha=sample.get("commit_sha"),
                framework=sample.get("framework"),
                repair_level=self.repair_level,
                thinking_content=response.thinking_content,
                reference_functions=sorted(reference_functions),
                reference_changed_lines=sorted(reference_changed_lines),
                prompt_input_scope="file",
            )

        if generated_edits is None:
            return RepairResult(
                sample_id=sample_id,
                success=False,
                generated_patch=None,
                reference_patch=reference_patch,
                buggy_code=buggy_code,
                error="Unable to parse patch",
                raw_response=response.content,
                response_time=response.response_time,
                token_usage=response.token_usage,
                finish_reason=response.finish_reason,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parent_file_path=sample.get("parent_file_path"),
                function_name=sample.get("function_name"),
                repository=sample.get("repository"),
                commit_sha=sample.get("commit_sha"),
                framework=sample.get("framework"),
                repair_level=self.repair_level,
                thinking_content=response.thinking_content,
                reference_functions=sorted(reference_functions),
                reference_changed_lines=sorted(reference_changed_lines),
                prompt_input_scope="file",
            )

        result = RepairResult(
            sample_id=sample_id,
            success=True,
            generated_patch=None,
            reference_patch=reference_patch,
            buggy_code=buggy_code,
            response_time=response.response_time,
            token_usage=response.token_usage,
            finish_reason=response.finish_reason,
            raw_response=response.content,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parent_file_path=sample.get("parent_file_path"),
            function_name=sample.get("function_name"),
            repository=sample.get("repository"),
            commit_sha=sample.get("commit_sha"),
            framework=sample.get("framework"),
            repair_level=self.repair_level,
            thinking_content=response.thinking_content,
            reference_functions=sorted(reference_functions),
            reference_changed_lines=sorted(reference_changed_lines),
            generated_edits=[asdict(edit) for edit in generated_edits],
            edit_count=len(generated_edits),
            edit_parse_success=True,
            prompt_input_scope="file",
        )

        try:
            result.format_valid = True
            applied_code, search_hit_success, unique_hit_success, non_empty_edit = self._apply_search_replace_edits(
                buggy_code,
                generated_edits,
            )
            result.search_hit_success = search_hit_success
            result.unique_hit_success = unique_hit_success
            result.non_empty_edit = non_empty_edit
            generated_patch = self._make_generated_patch(sample, buggy_code, applied_code)

            generated_norm = self._normalize_text(generated_patch)
            reference_norm = self._normalize_text(reference_patch)
            result.generated_patch = generated_patch
            result.exact_match = generated_norm == reference_norm
            result.patch_similarity = self._similarity(generated_norm, reference_norm)

            parsed_generated = self._parse_unified_diff(generated_norm)
            result.changed = self._patch_has_edits(parsed_generated)

            predicted_changed_lines = self._extract_changed_lines(parsed_generated)
            predicted_functions = self._locate_changed_functions(buggy_code, predicted_changed_lines)
            result.predicted_changed_lines = sorted(predicted_changed_lines)
            result.predicted_functions = sorted(predicted_functions)

            result.apply_success = True
            result.applied_code = applied_code
            applied_norm = self._normalize_text(applied_code)
            fixed_norm = self._normalize_text(fixed_code)
            result.exact_fixed_match = self._has_equivalent_python_ast(applied_code, fixed_code)
            result.fixed_code_similarity = self._similarity(applied_norm, fixed_norm)
            syntax_valid = self._has_valid_python_syntax(applied_code)
            differs_by_empty_lines_only = self._differs_only_by_empty_lines(applied_code, buggy_code)
            result.syntax_valid = syntax_valid
            result.empty_line_only = differs_by_empty_lines_only
            result.valid_edit = (
                result.format_valid
                and result.apply_success
                and result.non_empty_edit
                and syntax_valid
                and not differs_by_empty_lines_only
            )

            if not syntax_valid:
                result.error = "Generated code has invalid Python syntax"
            elif differs_by_empty_lines_only:
                result.error = "Generated patch only changes empty lines"

            predicted_function_set = set(predicted_functions)
            reference_function_set = set(reference_functions)
            result.function_hit = self._compute_set_hit(predicted_function_set, reference_function_set)
            result.function_jaccard = self._compute_set_jaccard(predicted_function_set, reference_function_set)
            result.function_exact_match = predicted_function_set == reference_function_set
            result.line_hit = self._compute_set_hit(predicted_changed_lines, reference_changed_lines)
            result.line_jaccard = self._compute_set_jaccard(predicted_changed_lines, reference_changed_lines)
            result.line_exact_match = predicted_changed_lines == reference_changed_lines
        except Exception as exc:
            if "not found" in str(exc):
                result.search_hit_success = False
            if "multiple locations" in str(exc):
                result.search_hit_success = True
                result.unique_hit_success = False
            result.error = str(exc)

        return result

    def _compute_metrics(
        self,
        results: List[RepairResult],
        model_name: str,
        dataset_name: str,
        total_time: float,
    ) -> RepairEvaluationReport:
        total = len(results)
        successful = [r for r in results if r.success and r.generated_edits is not None]
        failed = [r for r in results if not r.success or r.generated_edits is None]

        def rate(attr: str) -> float:
            return sum(1 for r in results if getattr(r, attr)) / total if total else 0.0

        def avg(attr: str) -> float:
            return sum(float(getattr(r, attr) or 0.0) for r in results) / total if total else 0.0

        response_times = [r.response_time for r in successful if r.response_time]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        token_stats: Dict[str, int] = {}
        for r in successful:
            if r.token_usage:
                for key, value in r.token_usage.items():
                    if isinstance(value, (int, float)):
                        token_stats[key] = token_stats.get(key, 0) + int(value or 0)

        return RepairEvaluationReport(
            model_name=model_name,
            dataset_name=dataset_name,
            repair_level=self.repair_level,
            total_samples=total,
            successful_samples=len(successful),
            failed_samples=len(failed),
            valid_repair_rate=rate("valid_edit"),
            function_hit_rate=rate("function_hit"),
            avg_function_jaccard=avg("function_jaccard"),
            function_exact_match_rate=rate("function_exact_match"),
            line_hit_rate=rate("line_hit"),
            avg_line_jaccard=avg("line_jaccard"),
            line_exact_match_rate=rate("line_exact_match"),
            avg_patch_similarity=avg("patch_similarity"),
            exact_fixed_match_rate=rate("exact_fixed_match"),
            block_parse_rate=rate("edit_parse_success"),
            search_hit_rate=rate("search_hit_success"),
            unique_hit_rate=rate("unique_hit_success"),
            apply_rate=rate("apply_success"),
            non_empty_edit_rate=rate("non_empty_edit"),
            avg_edit_count=avg("edit_count"),
            total_time=total_time,
            avg_response_time=avg_response_time,
            results=results,
            token_stats=token_stats,
        )

    def _save_results(self, report: RepairEvaluationReport, dataset_name: str) -> None:
        model_dir = self.output_dir / report.model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        details = []
        for r in report.results:
            detail = {
                "id": r.sample_id,
                "success": r.success,
                "framework": r.framework,
                "repository": r.repository,
                "commit_sha": r.commit_sha,
                "parent_file_path": r.parent_file_path,
                "function_name": r.function_name,
                "repair_level": r.repair_level,
                "repair_reason": r.repair_reason,
                "response_time": r.response_time,
                "token_usage": r.token_usage,
                "finish_reason": r.finish_reason,
                "error": r.error,
                "prompt_input_scope": r.prompt_input_scope or "file",
                "raw_output": r.raw_response,
                "parsed_output": {
                    "format_valid": r.format_valid,
                    "edit_parse_success": r.edit_parse_success,
                    "edit_count": r.edit_count,
                    "generated_edits": r.generated_edits,
                },
                "post_process": {
                    "edited_file": r.parent_file_path,
                    "apply_success": r.apply_success,
                    "search_hit_success": r.search_hit_success,
                    "unique_hit_success": r.unique_hit_success,
                    "non_empty_edit": r.non_empty_edit,
                    "changed": r.changed,
                    "syntax_valid": r.syntax_valid,
                    "empty_line_only": r.empty_line_only,
                },
                "evaluation": {
                    "valid_edit": r.valid_edit,
                    "exact_match": r.exact_match,
                    "exact_fixed_match": r.exact_fixed_match,
                    "patch_similarity": r.patch_similarity,
                    "fixed_code_similarity": r.fixed_code_similarity,
                    "function_hit": r.function_hit,
                    "function_jaccard": r.function_jaccard,
                    "function_exact_match": r.function_exact_match,
                    "line_hit": r.line_hit,
                    "line_jaccard": r.line_jaccard,
                    "line_exact_match": r.line_exact_match,
                    "predicted_functions": r.predicted_functions,
                    "reference_functions": r.reference_functions,
                    "predicted_changed_lines": r.predicted_changed_lines,
                    "reference_changed_lines": r.reference_changed_lines,
                },
            }
            # Persist full prompt context for failed cases or debug runs so parser
            # issues can be distinguished from model-formatting issues.
            if self.debug or r.error:
                detail["post_process"]["original_file_content"] = r.buggy_code
                detail["post_process"]["new_file_content"] = r.applied_code
                detail["system_prompt"] = r.system_prompt
                detail["user_prompt"] = r.user_prompt
                detail["thinking_content"] = r.thinking_content
            details.append(detail)

        output = {
            **report.to_dict(),
            "details": details,
        }
        output_path = self._get_output_path(model_dir, dataset_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info("Repair evaluation results saved: %s", output_path)

    def evaluate_dataset(
        self,
        dataset_path: Path,
        limit: Optional[int] = None,
        save_results: bool = True,
        resume: bool = True,
    ) -> RepairEvaluationReport:
        dataset_name = dataset_path.stem
        samples = self._load_samples(dataset_path)
        target_samples = samples[:limit] if limit else samples

        existing_results: Dict[str, Dict[str, Any]] = {}
        if resume:
            existing_results = self._load_existing_results(dataset_name)

        current_results: List[RepairResult] = []
        samples_to_evaluate: List[Dict[str, Any]] = []
        for sample in target_samples:
            sample_id = self._get_sample_id(sample)
            if sample_id in existing_results:
                current_results.append(self._result_from_dict(existing_results[sample_id]))
            else:
                samples_to_evaluate.append(sample)

        logger.info("Target samples in this run: %d", len(target_samples))
        logger.info("  - Reused existing results: %d", len(current_results))
        logger.info("  - Newly evaluated samples: %d", len(samples_to_evaluate))

        start_time = time.time()
        new_results: List[RepairResult] = []
        if samples_to_evaluate:
            if self.max_workers > 0:
                new_results = self._evaluate_parallel(
                    samples_to_evaluate,
                    existing_results=existing_results,
                    dataset_name=dataset_name,
                    save_checkpoint=save_results,
                )
            else:
                new_results = self._evaluate_sequential(
                    samples_to_evaluate,
                    existing_results=existing_results,
                    dataset_name=dataset_name,
                    save_checkpoint=save_results,
                )

        total_time = time.time() - start_time
        current_results.extend(new_results)
        report = self._compute_metrics(
            current_results,
            self._display_name,
            dataset_name,
            total_time,
        )
        if save_results:
            for r in new_results:
                existing_results[r.sample_id] = asdict(r)
            full_results = [self._result_from_dict(d) for d in existing_results.values()]
            full_report = self._compute_metrics(
                full_results,
                self._display_name,
                dataset_name,
                total_time,
            )
            self._save_results(full_report, dataset_name)
        return report

    def _save_checkpoint(
        self,
        pending_results: List[RepairResult],
        existing_results: Dict[str, Dict[str, Any]],
        dataset_name: str,
    ) -> None:
        for result in pending_results:
            existing_results[result.sample_id] = asdict(result)

        checkpoint_report = self._compute_metrics(
            [self._result_from_dict(d) for d in existing_results.values()],
            self._display_name,
            dataset_name,
            0.0,
        )
        self._save_results(checkpoint_report, dataset_name)

    def _build_failed_result(self, sample: Dict[str, Any], error: str) -> RepairResult:
        return RepairResult(
            sample_id=self._get_sample_id(sample),
            success=False,
            generated_patch=None,
            reference_patch=None,
            buggy_code=None,
            error=error,
            repair_level=self.repair_level,
            framework=sample.get("framework"),
            repository=sample.get("repository"),
            commit_sha=sample.get("commit_sha"),
            parent_file_path=sample.get("parent_file_path"),
            function_name=sample.get("function_name"),
            prompt_input_scope="file",
        )

    def _evaluate_parallel(
        self,
        samples: List[Dict[str, Any]],
        existing_results: Dict[str, Dict[str, Any]],
        dataset_name: str,
        save_checkpoint: bool = True,
    ) -> List[RepairResult]:
        results: List[RepairResult] = []
        pending_results: List[RepairResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sample = {
                executor.submit(self.evaluate_sample, sample): sample
                for sample in samples
            }

            for future in tqdm(as_completed(future_to_sample), total=len(samples), desc="Patch repair progress"):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Repair sample failed: %s", exc)
                    result = self._build_failed_result(sample, str(exc))

                results.append(result)
                pending_results.append(result)

                if save_checkpoint and len(pending_results) >= CHECKPOINT_INTERVAL:
                    self._save_checkpoint(pending_results, existing_results, dataset_name)
                    pending_results = []

        if save_checkpoint and pending_results:
            self._save_checkpoint(pending_results, existing_results, dataset_name)

        return results

    def _evaluate_sequential(
        self,
        samples: List[Dict[str, Any]],
        existing_results: Dict[str, Dict[str, Any]],
        dataset_name: str,
        save_checkpoint: bool = True,
    ) -> List[RepairResult]:
        results: List[RepairResult] = []
        pending_results: List[RepairResult] = []

        for sample in tqdm(samples, desc="Patch repair progress"):
            try:
                result = self.evaluate_sample(sample)
            except Exception as exc:
                logger.error("Repair sample failed: %s", exc)
                result = self._build_failed_result(sample, str(exc))

            results.append(result)
            pending_results.append(result)

            if save_checkpoint and len(pending_results) >= CHECKPOINT_INTERVAL:
                self._save_checkpoint(pending_results, existing_results, dataset_name)
                pending_results = []

        if save_checkpoint and pending_results:
            self._save_checkpoint(pending_results, existing_results, dataset_name)

        return results
