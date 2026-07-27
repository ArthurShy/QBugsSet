"""
Bug-detection evaluator.

Evaluates LLM performance on the quantum bug-detection task, with support for:
- Function-level code detection
- File-level code detection
- Batch evaluation
- Result statistics and reporting
"""

import json
import logging
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Project path configuration.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

from api_clients import BaseClient, BatchCompletionRequest, LLMResponse


logger = logging.getLogger(__name__)

# Incremental-save interval, measured in processed samples.
CHECKPOINT_INTERVAL = 50


@dataclass
class EvaluationResult:
    """Evaluation result for a single sample."""
    sample_id: str
    true_label: int              # Ground-truth label (1=buggy, 0=fixed).
    predicted_label: Optional[int]  # Predicted label.
    success: bool                # Whether prediction succeeded.
    response_time: Optional[float] = None
    token_usage: Optional[Dict] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    system_prompt: Optional[str] = None  # Input system prompt.
    user_prompt: Optional[str] = None    # Input user prompt.
    predicted_reason: Optional[str] = None  # Model-provided reasoning for the prediction.
    bug_reason: Optional[str] = None  # Bug reason.
    quantum_reason: Optional[str] = None  # Quantum-related reason.
    lifecycle_reason: Optional[str] = None  # Lifecycle-stage reason.
    lifecycle_stage: Optional[str] = None  # Lifecycle stage.
    submodule: Optional[str] = None  # Submodule.
    sample_category: Optional[str] = None  # Sample category: quantum_related_bug, non_quantum_bug, negative.
    parent_file_path: Optional[str] = None  # Sample file path.
    quantum_specific: Optional[bool] = None  # Whether the sample is quantum-specific.
    commit_sha: Optional[str] = None  # Commit hash.
    repository: Optional[str] = None  # Repository name.
    framework: Optional[str] = None  # Framework (qiskit/cirq/pennylane).
    thinking_content: Optional[str] = None  # Chain-of-thought content from reasoning models.
    
    def is_correct(self) -> bool:
        """Return whether the prediction is correct."""
        return self.success and self.predicted_label == self.true_label


@dataclass
class EvaluationReport:
    """Evaluation report."""
    model_name: str
    dataset_name: str
    total_samples: int
    successful_samples: int
    failed_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]
    total_time: float
    avg_response_time: float
    category_accuracy: Dict[str, float] = field(default_factory=dict)  # Accuracy per sample category.
    results: List[EvaluationResult] = field(default_factory=list)
    # Token stats.
    token_stats: Dict[str, int] = field(default_factory=dict)  # Dynamic token stats across different platforms.
    # Metrics for quantum-related bugs versus negatives.
    quantum_vs_negative: Dict[str, Any] = field(default_factory=dict)
    # Metrics for non-quantum-related bugs versus negatives.
    non_quantum_vs_negative: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "metrics": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1
            },
            "confusion_matrix": {
                "TN_FP": self.confusion_matrix[0],
                "FN_TP": self.confusion_matrix[1]
            },
            "token_usage": self.token_stats
        }
    
    def print_summary(self) -> None:
        """Print the evaluation summary."""
        print("\n" + "=" * 60)
        print(f"Evaluation report - {self.model_name}")
        print("=" * 60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Total samples: {self.total_samples}")
        print(f"Successful predictions: {self.successful_samples}")
        print(f"Failed predictions: {self.failed_samples}")
        print("-" * 40)
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
        print(f"F1 score: {self.f1:.4f}")
        print("-" * 40)
        print("Confusion matrix (rows=true, cols=predicted):")
        print(f"                 Pred:Fixed(0)   Pred:Buggy(1)")
        print(f"  True:Fixed(0)   {self.confusion_matrix[0][0]:^13d}   {self.confusion_matrix[0][1]:^13d}")
        print(f"  True:Buggy(1)   {self.confusion_matrix[1][0]:^13d}   {self.confusion_matrix[1][1]:^13d}")
        print("-" * 40)
        # Category-specific recall.
        if self.quantum_vs_negative or self.non_quantum_vs_negative:
            print("Category recall:")
            if self.quantum_vs_negative:
                print(f"  Quantum-related bugs: {self.quantum_vs_negative.get('recall', 0):.4f}")
            if self.non_quantum_vs_negative:
                print(f"  Non-quantum-related bugs: {self.non_quantum_vs_negative.get('recall', 0):.4f}")
            print("-" * 40)
        print(f"Total time: {self.total_time:.2f}s")
        print(f"Average response time: {self.avg_response_time:.2f}s")
        print("-" * 40)
        print("Token usage:")
        for key, value in self.token_stats.items():
            print(f"  {key}: {value:,}")
        # Show cache hit rate when cache data is present.
        hit = self.token_stats.get('prompt_cache_hit_tokens', 0)
        miss = self.token_stats.get('prompt_cache_miss_tokens', 0)
        if hit + miss > 0:
            hit_rate = hit / (hit + miss) * 100
            print(f"  Cache hit rate: {hit_rate:.1f}%")
        print("=" * 60)


class BugDetectionEvaluator:
    """
    Bug-detection evaluator.
    
    Used to evaluate LLM performance on the quantum bug-detection task.
    """
    
    # =========================================================================
    # System Prompt
    # =========================================================================
    
    SYSTEM_PROMPT = """You are a code reviewer for quantum computing software.

## Task
Determine if the code contains a **definite defect** that causes incorrect behavior.
- label=1: A concrete bug exists (logic error, wrong computation, missing return, API misuse, etc.)
- label=0: No definite bug found, or only code smells/style issues/potential improvements

## Key Rules
1. **High evidence bar**: Report bugs only if you can pinpoint the exact location AND explain why it fails in normal usage.
2. **Assume external correctness**: Libraries, imports, and APIs work as documented.
3. **NOT bugs**: Unused variables, missing docs, suboptimal-but-correct code, deprecated-but-working APIs, test patterns.
4. **Quantum awareness**: Apply domain knowledge; don't flag valid quantum idioms as bugs.

When uncertain → label=0

## Output
```json
{
  "label": 0 or 1,
  "reason": "Brief evidence. If label=1, specify the defect location."
}
```
"""

    # =========================================================================
    # User prompt - review the code file.
    # =========================================================================
    
    USER_PROMPT_PY = """
## Target Code File:
`````python
{code}
`````
Please strictly output JSON result according to the System Prompt requirements.
"""

    def __init__(
        self,
        client: BaseClient,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        max_workers: int = 10,
        use_batch: bool = False,
    ):
        """
        Initialize the evaluator.
        
        Args:
            client: LLM client.
            output_dir: Output directory.
            debug: Whether debug mode is enabled to show raw LLM responses.
            max_workers: Number of parallel worker threads, with 0 meaning serial execution.
        """
        self.client = client
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.max_workers = max_workers
        self.use_batch = use_batch
        # Resolve the display name for output paths, preferring display_name over model_name.
        self._display_name = getattr(client, 'display_name', None) or client.model_name
        
        logger.info("Initializing evaluator")
        logger.info(f"  Client: {client}")
        logger.info(f"  Debug mode: {debug}")
        logger.info(f"  Parallel workers: {max_workers if max_workers > 0 else 'serial'}")
        logger.info("  Detection Prompt: en-v2")
        logger.info(f"  Server-side batch: {'enabled' if self.use_batch else 'disabled'}")
    

    def _parse_prediction(self, content: str) -> tuple:
        """
        Parse model output and extract the predicted label and reason.
        
        Args:
            content: Model output content.
            
        Returns:
            Tuple of (label, reason). Returns (None, None) if parsing fails.
        """
        content = content.strip()
        
        # Remove Markdown code-fence markers.
        if content.startswith("```"):
            lines = content.split("\n")
            # Drop the opening ```json line and the trailing ``` line.
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            content = "\n".join(lines).strip()

        # Prefer the first complete JSON object to avoid surrounding explanatory text.
        if "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}")
            if 0 <= start < end:
                content = content[start:end + 1].strip()

        # Remove invalid control characters while keeping common whitespace.
        content = "".join(
            ch for ch in content
            if ch >= " " or ch in "\n\r\t"
        )
        
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try repairing invalid escapes in LaTeX-like content such as \frac or \pi.
            import re
            fixed_content = re.sub(r'\\([^"\\nrtbfu/])', r'\\\\\1', content)
            try:
                result = json.loads(fixed_content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse prediction result: {content[:200]}...")
                return None, None
        
        label = result.get("label")
        reason = result.get("reason", "")
        
        if label in [0, 1]:
            return label, reason
        if str(label).strip() in ["0", "1"]:
            return int(str(label).strip()), reason
            
        logger.warning(f"Invalid label value: {label}")
        return None, None
    
    def _build_prompt(self, sample: Dict[str, Any]) -> tuple:
        """
        Build prompts based on the prompt mode.
        
        Args:
            sample: Sample data.
            
        Returns:
            Tuple of (system_prompt, user_prompt).
        
        """
        code = sample.get("bug_file_content", "")
        
        return self.SYSTEM_PROMPT, self.USER_PROMPT_PY.format(code=code)

    def _get_sample_id(self, sample: Dict[str, Any]) -> str:
        """
        Get the sample ID while supporting multiple data formats.
        
        Prefer the id field while remaining compatible with the legacy sample_id field.
        """
        if "id" in sample:
            return sample["id"]
        elif "sample_id" in sample:
            return sample["sample_id"]
        else:
            # For negative samples, generate an ID from file_path plus parent_sha or commit_sha.
            file_path = sample.get("file_path", "unknown")
            sha = sample.get("parent_sha", sample.get("commit_sha", "unknown"))
            # Use a short hash fragment to avoid overly long IDs.
            short_sha = sha[:8] if len(sha) > 8 else sha
            return f"neg_{file_path}_{short_sha}"

    def _extract_sample_context(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = self._get_sample_id(sample)
        true_label = sample.get("label")
        quantum_specific = sample.get("quantum_specific", sample.get("is_quantum_related", False))
        if true_label == 0:
            sample_category = "negative"
        elif quantum_specific:
            sample_category = "quantum_related_bug"
        else:
            sample_category = "non_quantum_bug"

        return {
            "sample_id": sample_id,
            "true_label": true_label,
            "code": sample.get("bug_file_content", ""),
            "bug_reason": sample.get("bug_reason"),
            "quantum_reason": sample.get("quantum_reason"),
            "lifecycle_reason": sample.get("lifecycle_reason"),
            "lifecycle_stage": sample.get("lifecycle_stage"),
            "submodule": sample.get("submodule"),
            "parent_file_path": sample.get("parent_file_path") or sample.get("file_path"),
            "commit_sha": sample.get("commit_sha"),
            "repository": sample.get("repository"),
            "framework": sample.get("framework"),
            "quantum_specific": quantum_specific,
            "sample_category": sample_category,
        }

    def _build_response_result(
        self,
        context: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        response: LLMResponse,
        predicted_label: Optional[int],
        predicted_reason: Optional[str],
    ) -> EvaluationResult:
        return EvaluationResult(
            sample_id=context["sample_id"],
            true_label=context["true_label"],
            predicted_label=predicted_label,
            success=predicted_label is not None,
            response_time=response.response_time,
            token_usage=response.token_usage,
            raw_response=response.content,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            predicted_reason=predicted_reason,
            bug_reason=context["bug_reason"],
            quantum_reason=context["quantum_reason"],
            lifecycle_reason=context["lifecycle_reason"],
            lifecycle_stage=context["lifecycle_stage"],
            submodule=context["submodule"],
            sample_category=context["sample_category"],
            parent_file_path=context["parent_file_path"],
            quantum_specific=context["quantum_specific"],
            commit_sha=context["commit_sha"],
            repository=context["repository"],
            framework=context["framework"],
            thinking_content=response.thinking_content,
        )

    def _build_failed_result(
        self,
        context: Dict[str, Any],
        error: str,
        *,
        response: Optional[LLMResponse] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            sample_id=context["sample_id"],
            true_label=context["true_label"],
            predicted_label=None,
            success=False,
            error=error,
            response_time=response.response_time if response else None,
            raw_response=response.content if response else None,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            bug_reason=context["bug_reason"],
            quantum_reason=context["quantum_reason"],
            lifecycle_reason=context["lifecycle_reason"],
            lifecycle_stage=context["lifecycle_stage"],
            submodule=context["submodule"],
            sample_category=context["sample_category"],
            parent_file_path=context["parent_file_path"],
            quantum_specific=context["quantum_specific"],
            commit_sha=context["commit_sha"],
            repository=context["repository"],
            framework=context["framework"],
        )

    def evaluate_sample(self, sample: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate a single sample.
        
        Args:
            sample: Sample data.
            
        Returns:
            EvaluationResult object.
        """
        context = self._extract_sample_context(sample)
        code = context["code"]

        if not code:
            return self._build_failed_result(context, "Code is empty")
        
        # Build prompts based on the selected mode.
        system_prompt, user_prompt = self._build_prompt(sample)
        
        # Call the LLM. If parsing fails, retry up to three times.
        max_parse_retries = 3
        response = None
        predicted_label = None
        predicted_reason = None
        
        for parse_attempt in range(max_parse_retries):
            response = self.client.complete(
                user_prompt, 
                system_prompt,
                response_format={"type": "json_object"}
            )
            
            if not response.success:
                break  # API call failed, so do not retry.
            
            # Parse the prediction result.
            predicted_label, predicted_reason = self._parse_prediction(response.content)
            
            if predicted_label is not None:
                break  # Parsing succeeded.
            
            # Parsing failed, so retry.
            if parse_attempt < max_parse_retries - 1:
                logger.debug(f"Parse failed, retrying ({parse_attempt + 1}/{max_parse_retries})")
        
        # API call failed.
        if not response.success:
            return self._build_failed_result(
                context,
                response.error or "Request failed",
                response=response,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        return self._build_response_result(
            context,
            system_prompt,
            user_prompt,
            response,
            predicted_label,
            predicted_reason,
        )

    def _build_batch_request_items(
        self,
        samples: List[Dict[str, Any]],
    ) -> tuple[List[BatchCompletionRequest], Dict[str, Dict[str, Any]]]:
        requests: List[BatchCompletionRequest] = []
        prepared: Dict[str, Dict[str, Any]] = {}
        for sample in samples:
            context = self._extract_sample_context(sample)
            system_prompt, user_prompt = self._build_prompt(sample)
            custom_id = context["sample_id"]
            prepared[custom_id] = {
                "sample": sample,
                "context": context,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
            requests.append(
                BatchCompletionRequest(
                    custom_id=custom_id,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    response_format={"type": "json_object"},
                )
            )
        return requests, prepared

    def _evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        existing_results: Dict[str, Dict],
        dataset_name: str,
        save_checkpoint: bool = True,
    ) -> List[EvaluationResult]:
        requests, prepared = self._build_batch_request_items(samples)
        batch_responses = self.client.complete_batch(requests)
        results: List[EvaluationResult] = []

        for sample in tqdm(samples, desc="Batch evaluation progress"):
            sample_id = self._get_sample_id(sample)
            item = prepared[sample_id]
            context = item["context"]
            response = batch_responses.get(sample_id)
            if response is None:
                result = self._build_failed_result(
                    context,
                    "Batch response missing",
                    system_prompt=item["system_prompt"],
                    user_prompt=item["user_prompt"],
                )
            elif not response.success:
                result = self._build_failed_result(
                    context,
                    response.error or "Batch request failed",
                    response=response,
                    system_prompt=item["system_prompt"],
                    user_prompt=item["user_prompt"],
                )
            else:
                predicted_label, predicted_reason = self._parse_prediction(response.content)
                if predicted_label is None:
                    logger.warning("Batch output parsing failed and will be recorded as failure: %s", sample_id)
                    result = self._build_failed_result(
                        context,
                        "Batch output parse failed",
                        response=response,
                        system_prompt=item["system_prompt"],
                        user_prompt=item["user_prompt"],
                    )
                else:
                    result = self._build_response_result(
                        context,
                        item["system_prompt"],
                        item["user_prompt"],
                        response,
                        predicted_label,
                        predicted_reason,
                    )

            results.append(result)

        if save_checkpoint and results:
            self._save_checkpoint(results, existing_results, dataset_name)

        return results
    
    def _load_existing_results(self, dataset_name: str) -> Dict[str, Dict]:
        """
        Load existing evaluation results for resume support.
        
        Returns:
            Mapping of evaluated samples in the form {sample_id: result_dict}.
        """
        model_dir = self.output_dir / self._display_name.replace("/", "_")
        output_path = self._get_output_path(model_dir, dataset_name)
        
        if not output_path.exists():
            return {}
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            existing_results = {}
            failed_count = 0
            for detail in data.get("details", []):
                # Support both new and old field names: prefer id but also accept sample_id.
                sample_id = detail.get("id") or detail.get("sample_id")
                # Only keep successful predictions. Failed samples will be retried.
                if sample_id and detail.get("predicted_label") is not None:
                    existing_results[sample_id] = detail
                elif sample_id:
                    failed_count += 1
            
            logger.info(f"Loaded existing results: {len(existing_results)} successful, {failed_count} failed and pending retry")
            return existing_results
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
            return {}
    
    def _result_from_dict(self, d: Dict) -> EvaluationResult:
        """Restore an EvaluationResult from a dictionary while supporting legacy fields."""
        # Support legacy result files that may contain predicted_is_quantum_related.
        if "predicted_is_quantum_related" in d:
            d = {k: v for k, v in d.items() if k != "predicted_is_quantum_related"}
        return EvaluationResult(
            sample_id=d.get("id") or d.get("sample_id"),
            true_label=d.get("true_label"),
            predicted_label=d.get("predicted_label"),
            predicted_reason=d.get("predicted_reason"),
            success=d.get("success", True),
            error=d.get("error"),
            response_time=d.get("response_time"),
            token_usage=d.get("token_usage"),
            system_prompt=d.get("system_prompt"),
            user_prompt=d.get("user_prompt"),
            raw_response=d.get("raw_response"),
            bug_reason=d.get("bug_reason"),
            quantum_reason=d.get("quantum_reason"),
            lifecycle_reason=d.get("lifecycle_reason"),
            lifecycle_stage=d.get("lifecycle_stage"),
            submodule=d.get("submodule"),
            sample_category=d.get("sample_category"),
            parent_file_path=d.get("parent_file_path") or d.get("filepath"),
            quantum_specific=d.get("quantum_specific", d.get("is_quantum_related")),
            commit_sha=d.get("commit_sha"),
            repository=d.get("repository"),
            framework=d.get("framework"),
            thinking_content=d.get("thinking_content")
        )

    def _save_checkpoint(
        self,
        new_results: List[EvaluationResult],
        existing_results: Dict[str, Dict],
        dataset_name: str
    ) -> None:
        """
        Save an incremental checkpoint.
        
        Args:
            new_results: Newly evaluated results.
            existing_results: Existing results dictionary, updated in place.
            dataset_name: Dataset name.
        """
        # Merge the new results into existing_results.
        for r in new_results:
            existing_results[r.sample_id] = asdict(r)
        
        # Rebuild the full result list.
        full_results_list = [self._result_from_dict(d) for d in existing_results.values()]
        
        # Recompute metrics.
        checkpoint_report = self._compute_metrics(
            results=full_results_list,
            model_name=self._display_name,
            dataset_name=dataset_name,
            total_time=0.0  # Checkpoints do not compute total runtime.
        )
        
        # Save results.
        self._save_results(checkpoint_report, dataset_name)
        logger.info(f"Checkpoint saved: {len(existing_results)} total results")


    def evaluate_dataset(
        self, 
        dataset_path: Path, 
        limit: Optional[int] = None,
        save_results: bool = True,
        resume: bool = True
    ) -> EvaluationReport:
        """Evaluate the entire dataset."""
        dataset_name = dataset_path.stem
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        else:
            samples = data
            
        # 1. Try to resume from previous results. If resume is disabled, start from an empty dictionary.
        existing_results = {}
        if resume:
            existing_results = self._load_existing_results(dataset_name)
            
        # 2. Determine the target samples for this run from limit, without truncating the original dataset stream.
        if limit:
            target_samples = samples[:limit]
        else:
            target_samples = samples
            
        # 3. Split samples into already-evaluated and pending groups.
        # Only reuse successful results. Failed samples are evaluated again.
        current_run_results = []
        samples_to_evaluate = []
        
        for sample in target_samples:
            sample_id = self._get_sample_id(sample)
            if sample_id in existing_results:
                existing_record = existing_results[sample_id]
                # Reuse only successful results. Failed samples are evaluated again.
                if existing_record.get("predicted_label") is not None:
                    current_run_results.append(self._result_from_dict(existing_record))
                else:
                    # Remove failed records from existing_results so they can be reevaluated.
                    del existing_results[sample_id]
                    samples_to_evaluate.append(sample)
            else:
                samples_to_evaluate.append(sample)
        
        logger.info(f"Target samples in this run: {len(target_samples)}")
        logger.info(f"  - Reused existing results: {len(current_run_results)}")
        logger.info(f"  - Newly evaluated samples: {len(samples_to_evaluate)}")
        
        start_time = time.time()
        
        # 4. Evaluate new samples. Pass existing_results and dataset_name so incremental saving can work.
        new_results = []
        if samples_to_evaluate:
            if self.use_batch:
                new_results = self._evaluate_batch(
                    samples_to_evaluate,
                    existing_results=existing_results,
                    dataset_name=dataset_name,
                    save_checkpoint=save_results,
                )
            elif self.max_workers and self.max_workers > 1:
                new_results = self._evaluate_parallel(
                    samples_to_evaluate, 
                    existing_results=existing_results,
                    dataset_name=dataset_name,
                    save_checkpoint=save_results
                )
            else:
                new_results = self._evaluate_sequential(
                    samples_to_evaluate,
                    existing_results=existing_results,
                    dataset_name=dataset_name,
                    save_checkpoint=save_results
                )
        
        total_time = time.time() - start_time
        current_run_results.extend(new_results)
        
        # 5. Compute metrics for the current run, used for display and return values.
        current_report = self._compute_metrics(
            results=current_run_results,
            model_name=self._display_name,
            dataset_name=dataset_name,
            total_time=total_time
        )
        
        # 6. Save results by merging old and new entries to support incremental persistence.
        if save_results:
            # Merge new results into existing_results.
            for r in new_results:
                existing_results[r.sample_id] = asdict(r)
                
            # Build the full combined result list for saving.
            full_results_list = [self._result_from_dict(d) for d in existing_results.values()]
            
            # Compute the full merged metrics.
            full_report = self._compute_metrics(
                results=full_results_list,
                model_name=self._display_name,
                dataset_name=dataset_name,
                total_time=total_time # Runtime accounting may be slightly inaccurate in incremental mode, but acceptable here.
            )
            
            self._save_results(full_report, dataset_name)
            
        return current_report

    def _evaluate_parallel(
        self, 
        samples: List[Dict],
        existing_results: Dict[str, Dict],
        dataset_name: str,
        save_checkpoint: bool = True
    ) -> List[EvaluationResult]:
        """Evaluate in parallel and save once every CHECKPOINT_INTERVAL samples."""
        results = []
        pending_results = []  # Results waiting to be saved.
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks.
            future_to_sample = {
                executor.submit(self.evaluate_sample, sample): sample 
                for sample in samples
            }
            
            # Collect results.
            for future in tqdm(as_completed(future_to_sample), total=len(samples), desc="Evaluation progress"):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    results.append(result)
                    pending_results.append(result)
                except Exception as e:
                    logger.error(f"Sample evaluation failed: {e}")
                    # Generate a failed result.
                    failed_result = EvaluationResult(
                        sample_id=self._get_sample_id(sample),
                        true_label=sample.get("label"),
                        predicted_label=None,
                        success=False,
                        error=str(e),
                        framework=sample.get("framework")
                    )
                    results.append(failed_result)
                    pending_results.append(failed_result)
                
                # Save every CHECKPOINT_INTERVAL samples.
                if save_checkpoint and len(pending_results) >= CHECKPOINT_INTERVAL:
                    self._save_checkpoint(pending_results, existing_results, dataset_name)
                    pending_results = []  # Clear the pending-save list.
        
        # Save any remaining results.
        if save_checkpoint and pending_results:
            self._save_checkpoint(pending_results, existing_results, dataset_name)
        
        return results

    def _evaluate_sequential(
        self, 
        samples: List[Dict],
        existing_results: Dict[str, Dict],
        dataset_name: str,
        save_checkpoint: bool = True
    ) -> List[EvaluationResult]:
        """Evaluate serially and save once every CHECKPOINT_INTERVAL samples."""
        results = []
        pending_results = []  # Results waiting to be saved.
        
        for sample in tqdm(samples, desc="Evaluation progress"):
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
                pending_results.append(result)
            except Exception as e:
                logger.error(f"Sample evaluation failed: {e}")
                failed_result = EvaluationResult(
                    sample_id=self._get_sample_id(sample),
                    true_label=sample.get("label"),
                    predicted_label=None,
                    success=False,
                    error=str(e),
                    framework=sample.get("framework")
                )
                results.append(failed_result)
                pending_results.append(failed_result)
            
            # Save every CHECKPOINT_INTERVAL samples.
            if save_checkpoint and len(pending_results) >= CHECKPOINT_INTERVAL:
                self._save_checkpoint(pending_results, existing_results, dataset_name)
                pending_results = []  # Clear the pending-save list.
        
        # Save any remaining results.
        if save_checkpoint and pending_results:
            self._save_checkpoint(pending_results, existing_results, dataset_name)
        
        return results
    
    def _compute_metrics(
        self,
        results: List[EvaluationResult],
        model_name: str,
        dataset_name: str,
        total_time: float
    ) -> EvaluationReport:
        """Compute evaluation metrics."""
        
        # Use the presence of a parsed label as the success criterion to avoid polluted success flags.
        successful_results = [r for r in results if r.predicted_label is not None]
        failed_results = [r for r in results if r.predicted_label is None]
        
        if not successful_results:
            logger.error("No successful predictions were produced; all predicted_label values are empty")
            return EvaluationReport(
                model_name=model_name,
                dataset_name=dataset_name,
                total_samples=len(results),
                successful_samples=0,
                failed_samples=len(results),
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                confusion_matrix=[[0, 0], [0, 0]],
                total_time=total_time,
                avg_response_time=0.0,
                category_accuracy={},
                results=results
            )
        
        y_true = [r.true_label for r in successful_results]
        y_pred = [r.predicted_label for r in successful_results]
        
        # Compute core metrics.
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        # Compute average response time.
        response_times = [r.response_time for r in successful_results if r.response_time]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Compute accuracy by sample category.
        category_accuracy = {}
        category_results = defaultdict(list)
        
        for r in successful_results:
            category = r.sample_category or "unknown"
            category_results[category].append(r)
        
        for category, category_successful_results in category_results.items():
            if len(category_successful_results) > 0:
                category_y_true = [r.true_label for r in category_successful_results]
                category_y_pred = [r.predicted_label for r in category_successful_results]
                category_acc = accuracy_score(category_y_true, category_y_pred)
                category_accuracy[category] = category_acc
        
        # Compute metrics for quantum-related bugs versus negatives.
        quantum_vs_negative = {}
        quantum_results = category_results.get("quantum_related_bug", [])
        negative_results = category_results.get("negative", [])
        
        if quantum_results and negative_results:
            # Combine quantum-related bug results with negative results.
            combined_results = quantum_results + negative_results
            combined_y_true = [r.true_label for r in combined_results]
            combined_y_pred = [r.predicted_label for r in combined_results]
            
            qvn_accuracy = accuracy_score(combined_y_true, combined_y_pred)
            qvn_precision = precision_score(combined_y_true, combined_y_pred, zero_division=0)
            qvn_recall = recall_score(combined_y_true, combined_y_pred, zero_division=0)
            qvn_f1 = f1_score(combined_y_true, combined_y_pred, zero_division=0)
            qvn_cm = confusion_matrix(combined_y_true, combined_y_pred, labels=[0, 1]).tolist()
            
            quantum_vs_negative = {
                "accuracy": qvn_accuracy,
                "precision": qvn_precision,
                "recall": qvn_recall,
                "f1": qvn_f1,
                "confusion_matrix": qvn_cm
            }
        
        # Compute metrics for non-quantum-related bugs versus negatives.
        non_quantum_vs_negative = {}
        non_quantum_results = category_results.get("non_quantum_bug", [])
        
        if non_quantum_results and negative_results:
            # Combine non-quantum-related bug results with negative results.
            combined_results = non_quantum_results + negative_results
            combined_y_true = [r.true_label for r in combined_results]
            combined_y_pred = [r.predicted_label for r in combined_results]
            
            nqvn_accuracy = accuracy_score(combined_y_true, combined_y_pred)
            nqvn_precision = precision_score(combined_y_true, combined_y_pred, zero_division=0)
            nqvn_recall = recall_score(combined_y_true, combined_y_pred, zero_division=0)
            nqvn_f1 = f1_score(combined_y_true, combined_y_pred, zero_division=0)
            nqvn_cm = confusion_matrix(combined_y_true, combined_y_pred, labels=[0, 1]).tolist()
            
            non_quantum_vs_negative = {
                "accuracy": nqvn_accuracy,
                "precision": nqvn_precision,
                "recall": nqvn_recall,
                "f1": nqvn_f1,
                "confusion_matrix": nqvn_cm
            }
        
        # Aggregate token usage dynamically across all available fields.
        token_stats: Dict[str, int] = {}
        for r in results:
            if r.token_usage:
                for key, value in r.token_usage.items():
                    if isinstance(value, (int, float)):
                        token_stats[key] = token_stats.get(key, 0) + int(value or 0)
        
        return EvaluationReport(
            model_name=model_name,
            dataset_name=dataset_name,
            total_samples=len(results),
            successful_samples=len(successful_results),
            failed_samples=len(failed_results),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=cm,
            total_time=total_time,
            avg_response_time=avg_response_time,
            category_accuracy=category_accuracy,
            results=results,
            token_stats=token_stats,
            quantum_vs_negative=quantum_vs_negative,
            non_quantum_vs_negative=non_quantum_vs_negative
        )
    
    def _save_results(self, report: EvaluationReport, dataset_name: str) -> None:
        """Save evaluation results."""
        
        # Create the output directory.
        model_dir = self.output_dir / report.model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the detailed result list, grouped logically by field type.
        details = []
        for r in report.results:
            detail = {
                # Basic identifiers.
                "id": r.sample_id,
                "true_label": r.true_label,
                "predicted_label": r.predicted_label,
                "is_correct": r.is_correct(),
                "success": r.success,
                # Sample metadata.
                "framework": r.framework,
                "repository": r.repository,
                "commit_sha": r.commit_sha,
                "parent_file_path": r.parent_file_path,
                # Classification metadata.
                "quantum_specific": r.quantum_specific,
                "submodule": r.submodule,
                "lifecycle_stage": r.lifecycle_stage,
                "sample_category": r.sample_category,
                # Prediction output.
                "predicted_reason": r.predicted_reason,
                # Reason fields.
                "bug_reason": r.bug_reason,
                "quantum_reason": r.quantum_reason,
                "lifecycle_reason": r.lifecycle_reason,
                # Performance metrics.
                "response_time": r.response_time,
                "token_usage": r.token_usage,
                # Error information.
                "error": r.error,
            }
            # Save raw_response in debug mode or on failure to simplify debugging.
            if self.debug or not r.success:
                detail["raw_response"] = r.raw_response
            if self.debug:
                detail["system_prompt"] = r.system_prompt
                detail["user_prompt"] = r.user_prompt
                detail["thinking_content"] = r.thinking_content
            details.append(detail)
        
        # Combine summary and detailed results into one payload.
        combined_output = {
            **report.to_dict(),
            "details": details
        }
        
        # Save results.
        output_path = self._get_output_path(model_dir, dataset_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation results saved: {output_path}")

    def _get_output_path(self, model_dir: Path, dataset_name: str) -> Path:
        """Detection results always use the fixed en-v2 prompt."""
        return model_dir / f"{dataset_name}_results-en-v2.json"
