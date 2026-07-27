#!/usr/bin/env python3
"""
Main entry point for quantum bug evaluation.

Supported capabilities:
1. Evaluate with local vLLM models
2. Evaluate with the DeepSeek API
3. Batch evaluation and summary reporting
4. Bug-repair evaluation with search/replace edits

Usage:
    # Evaluate with the DeepSeek API
    python llm/run.py --model deepseek-chat --dataset data/05_datasets/dataset_parent_all.json

    # Evaluate with a local vLLM model
    python llm/run.py --model qwen2.5-7b-local --dataset data/05_datasets/dataset_parent_all.json

    # Limit the number of samples for testing
    python llm/run.py --model deepseek-chat --dataset data/05_datasets/dataset_parent_all.json --limit 10

    # Evaluate bug repair (defaults to method_level_single_merged.json)
    # The model outputs search/replace edits and the program builds and evaluates patches locally
    python llm/run.py --task repair --model deepseek-chat

    # List all available models
    python llm/run.py --list-models
"""

import os
import sys
import argparse
import logging
import shlex
from pathlib import Path

# Add project paths.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LLM_OUTPUT_DIR, BUG_DETECTION_DATASET, BUG_REPAIR_DATASET
from api_clients import (
    VLLMClient,
    DeepSeekLLMClient as DeepSeekClient,
    OpenRouterClient,
    AliCloudClient,
    BigModelClient,
    VertexAIClient,
    LLM_MODELS,
    LLMClientType,
    LLMModelConfig,
    get_llm_model_config,
)
from detection_evaluator import BugDetectionEvaluator
from repair_evaluator import RepairEvaluator

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def load_project_env(env_path: Path) -> None:
    """Load the repository-root .env file without overwriting existing environment variables."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):].strip()
        if '=' not in line:
            continue

        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


load_project_env(project_root / '.env')


def load_zshrc_exports(zshrc_path: Path, export_names: set[str]) -> None:
    """Load selected exported variables from ~/.zshrc without overwriting existing environment variables."""
    if not zshrc_path.exists():
        return

    for raw_line in zshrc_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line.startswith('export '):
            continue

        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            continue
        if len(tokens) < 2:
            continue

        assignment = tokens[1]
        if '=' not in assignment:
            continue
        key, value = assignment.split('=', 1)
        if key not in export_names or key in os.environ:
            continue
        os.environ[key] = value


load_zshrc_exports(
    Path.home() / '.zshrc',
    {
        'GoogleVertaxAI_API_KEY',
        'GoogleProjectID',
        'VERTEX_AI_BATCH_GCS_ROOT',
        'VERTEX_AI_PROJECT',
        'VERTEX_AI_LOCATION',
        'VERTEX_AI_ACCESS_TOKEN',
        'GOOGLE_CLOUD_ACCESS_TOKEN',
    },
)


def create_client(
    model_config: LLMModelConfig,
    api_key_override: str | None = None,
    vertex_batch_gcs_root: str | None = None,
    vertex_batch_poll_interval: int | None = None,
    vertex_batch_timeout: int | None = None,
    vertex_batch_job_name: str | None = None,
):
    """
    Create the appropriate client from the model configuration.
    
    Args:
        config: Model configuration.
        
    Returns:
        BaseClient instance.
    """
    if model_config.client_type == LLMClientType.VLLM:
        return VLLMClient(
            model_name=model_config.model_name,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout,
            reasoning_parser=model_config.reasoning_parser
        )
    elif model_config.client_type == LLMClientType.DEEPSEEK:
        api_key = api_key_override or (os.environ.get(model_config.api_key_env) if model_config.api_key_env else None)
        return DeepSeekClient(
            model_name=model_config.model_name,
            api_key=api_key,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout
        )
    elif model_config.client_type == LLMClientType.OPENROUTER:
        api_key = api_key_override or (os.environ.get(model_config.api_key_env) if model_config.api_key_env else None)
        return OpenRouterClient(
            model_name=model_config.model_name,
            api_key=api_key,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout
        )
    elif model_config.client_type == LLMClientType.ALICLOUD:
        api_key = api_key_override or (os.environ.get(model_config.api_key_env) if model_config.api_key_env else None)
        client = AliCloudClient(
            model_name=model_config.model_name,
            api_key=api_key,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout,
            reasoning_parser=model_config.reasoning_parser,
            thinking_budget=model_config.thinking_budget,
        )
        # Use the config name as the display name for output paths while keeping model_name for API calls.
        client.display_name = model_config.name
        return client
    elif model_config.client_type == LLMClientType.BIGMODEL:
        api_key = api_key_override or (os.environ.get(model_config.api_key_env) if model_config.api_key_env else None)
        client = BigModelClient(
            model_name=model_config.model_name,
            api_key=api_key,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout
        )
        client.display_name = model_config.name
        return client
    elif model_config.client_type == LLMClientType.VERTEXAI:
        api_key = api_key_override or (os.environ.get(model_config.api_key_env) if model_config.api_key_env else None)
        client = VertexAIClient(
            model_name=model_config.model_name,
            api_key=api_key,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout,
            batch_gcs_root=vertex_batch_gcs_root,
            batch_poll_interval=vertex_batch_poll_interval,
            batch_timeout=vertex_batch_timeout,
            batch_job_name=vertex_batch_job_name,
        )
        client.display_name = model_config.name
        return client
    else:
        raise ValueError(f"Unsupported client type: {model_config.client_type}")


def print_available_models():
    """Print all available models."""
    print("\n" + "=" * 60)
    print("Available models")
    print("=" * 60)
    
    # Helper function: get model description.
    def get_description(cfg):
        if isinstance(cfg, dict) and 'config' in cfg:
            return cfg['config'].description
        return getattr(cfg, 'description', 'No description')
    
    # Display by category.
    api_models = {k: v for k, v in LLM_MODELS.items() if "local" not in k}
    local_models = {k: v for k, v in LLM_MODELS.items() if "local" in k}
    
    print("\nRemote API models:")
    for name, cfg in api_models.items():
        print(f"  - {name}: {get_description(cfg)}")
    
    print("\nLocal vLLM models:")
    for name, cfg in local_models.items():
        print(f"  - {name}: {get_description(cfg)}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Quantum bug evaluation tool (detection / repair)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run bug detection with the DeepSeek API
    python run.py --model deepseek-chat --dataset ../data/05_datasets/dataset_parent_all.json

    # Run bug detection with local vLLM
    python run.py --model qwen2.5-7b-local --dataset ../data/05_datasets/dataset_parent_all.json

    # Quick test with only 10 samples
    python run.py --model deepseek-chat --dataset ../data/05_datasets/dataset_parent_all.json --limit 10

    # Bug repair evaluation (function-level by default, outputs search/replace edits)
    python run.py --task repair --model deepseek-chat
        """
    )
    
    # Required arguments.
    parser.add_argument(
        "--task",
        type=str,
        choices=["detection", "repair"],
        default="detection",
        help="Evaluation task: detection for bug detection, repair for bug repair"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name. Use --list-models to inspect available models"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Dataset path (JSON file)"
    )
    
    # Optional arguments.
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of evaluation samples for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LLM_OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode and show raw LLM responses"
    )
    parser.add_argument(
        "--prompt-version",
        type=int,
        choices=[1, 2],
        default=1,
        help="Prompt version. Currently only used for repair"
    )
    parser.add_argument(
        "--repair-level",
        type=str,
        choices=["function", "file"],
        default="function",
        help="Repair evaluation granularity: function or file (default: function)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker threads. Use 0 for serial execution"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        nargs='?',
        const=10,
        default=None,
        help="Enable parallel evaluation and optionally specify the worker count (default: 10)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode and restart evaluation from the beginning"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the service-availability check. Recommended for free models to avoid consuming rate-limit quota"
    )
    
    # Platform selection.
    parser.add_argument(
        "--platform",
        type=str,
        choices=["vllm", "deepseek", "alicloud", "openrouter", "bigmodel", "vertexai"],
        default=None,
        help="Specify the platform type: vllm (local deployment), deepseek, alicloud, openrouter, bigmodel, or vertexai"
    )
    
    # vLLM-specific arguments.
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="Custom vLLM service URL that overrides the default configured value"
    )
    
    # API-specific arguments.
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key that overrides the environment variable"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable server-side batch mode. Currently only supported for Vertex AI detection"
    )
    parser.add_argument(
        "--vertex-batch-gcs-root",
        type=str,
        default=None,
        help="GCS root directory used by Vertex AI batch, for example gs://your-bucket/quantum-batch"
    )
    parser.add_argument(
        "--vertex-batch-poll-interval",
        type=int,
        default=None,
        help="Vertex AI batch polling interval in seconds"
    )
    parser.add_argument(
        "--vertex-batch-timeout",
        type=int,
        default=None,
        help="Vertex AI batch total timeout in seconds"
    )
    parser.add_argument(
        "--vertex-batch-job-name",
        type=str,
        default=None,
        help="Continue waiting for an already submitted Vertex batch job, for example projects/.../batchPredictionJobs/..."
    )
    
    args = parser.parse_args()
    
    # List models.
    if args.list_models:
        print_available_models()
        return
    
    # Validate required arguments.
    if not args.model:
        parser.error("Please provide --model or use --list-models to inspect available models")
    if not args.dataset:
        args.dataset = BUG_REPAIR_DATASET if args.task == "repair" else BUG_DETECTION_DATASET
    
    # Validate dataset file.
    if not args.dataset.exists():
        logger.error(f"Dataset file does not exist: {args.dataset}")
        sys.exit(1)
    
    # Load model configuration, optionally filtered by platform.
    try:
        model_config = get_llm_model_config(args.model, platform=args.platform)
    except ValueError as e:
        logger.error(str(e))
        print("\nUse --list-models to inspect available models")
        sys.exit(1)
    
    # Override config values with CLI arguments when provided.
    if args.vllm_url and model_config.client_type == LLMClientType.VLLM:
        model_config.base_url = args.vllm_url
    
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {model_config.name}")
    logger.info(f"Dataset: {args.dataset}")
    if args.limit:
        logger.info(f"Sample limit: {args.limit}")
    # Create the client.
    try:
        client = create_client(
            model_config,
            api_key_override=args.api_key,
            vertex_batch_gcs_root=args.vertex_batch_gcs_root,
            vertex_batch_poll_interval=args.vertex_batch_poll_interval,
            vertex_batch_timeout=args.vertex_batch_timeout,
            vertex_batch_job_name=args.vertex_batch_job_name,
        )
    except ValueError as e:
        logger.error(f"Failed to create client: {e}")
        sys.exit(1)

    if args.batch:
        if model_config.client_type != LLMClientType.VERTEXAI:
            logger.error("--batch currently supports only Vertex AI clients")
            sys.exit(1)
        if args.task != "detection":
            logger.error("--batch is currently integrated only with the detection task")
            sys.exit(1)
        if not client.supports_batch_completion():
            logger.error("Vertex batch GCS root is not configured. Set --vertex-batch-gcs-root or VERTEX_AI_BATCH_GCS_ROOT")
            sys.exit(1)
    
    # Check service availability.
    skip_service_check = args.skip_check
    if args.batch and model_config.client_type == LLMClientType.VERTEXAI:
        skip_service_check = True

    if skip_service_check:
        if args.batch and model_config.client_type == LLMClientType.VERTEXAI and not args.skip_check:
            logger.info("Batch mode: skipping online service-availability check")
        else:
            logger.info("Skipping service-availability check")
    else:
        logger.info("Checking service availability...")
        if not client.is_available():
            logger.error("Service unavailable. Please check:")
            if model_config.client_type == LLMClientType.VLLM:
                logger.error(f"  - Whether the vLLM service is running: {model_config.base_url}")
                logger.error(f"  - Whether the model is loaded: {model_config.model_name}")
            else:
                error_summary = getattr(client, "last_error_summary", None)
                if error_summary:
                    provider_label = {
                        LLMClientType.VERTEXAI: "Vertex AI",
                        LLMClientType.ALICLOUD: "DashScope",
                        LLMClientType.DEEPSEEK: "DeepSeek API",
                        LLMClientType.OPENROUTER: "OpenRouter",
                        LLMClientType.BIGMODEL: "BigModel",
                    }.get(model_config.client_type, "upstream service")
                    logger.error(f"  - {provider_label} returned: {error_summary}")
                    if model_config.client_type == LLMClientType.VERTEXAI:
                        if "429" in error_summary or "RESOURCE_EXHAUSTED" in error_summary:
                            logger.error("  - This is usually rate limiting, insufficient project quota, or temporary model capacity pressure, not a local network issue")
                            logger.error("  - You can try adding --skip-check first. If the real request still returns 429, inspect Vertex AI quota and capacity")
                        elif "403" in error_summary or "PERMISSION_DENIED" in error_summary:
                            logger.error("  - This usually means the project does not have access to the model, the region is unsupported, or IAM permissions are insufficient")
                        elif "401" in error_summary or "UNAUTHENTICATED" in error_summary:
                            logger.error("  - This usually means the bearer token is invalid or expired, or the gcloud login state is broken")
                    elif model_config.client_type == LLMClientType.ALICLOUD:
                        if "product is not activated" in error_summary.lower():
                            logger.error("  - The current AliCloud account has not activated the required DashScope product or model service")
                            logger.error("  - Activate or authorize it in the AliCloud DashScope console before retrying")
                        elif "401" in error_summary or "unauthorized" in error_summary.lower():
                            logger.error("  - This usually means AliCloud_API_KEY is invalid, expired, or missing the required permission")
                        elif "429" in error_summary:
                            logger.error("  - This usually means DashScope is rate-limited or out of quota")
                else:
                    logger.error("  - Whether the API key or access token is configured correctly")
                    logger.error("  - Whether the network connection is working")
            sys.exit(1)
        logger.info("Service available")
    
    # Determine parallelism: --workers > --parallel > serial.
    if args.workers is not None:
        max_workers = args.workers
    elif args.parallel is not None:
        max_workers = args.parallel  # Defaults to 10, or the user-specified value.
        
        # For free models, adapt automatically to the number of available keys.
        if hasattr(client, 'key_pool') and client.key_pool:
            available_keys = client.key_pool.get_available_count()
            if available_keys > 0:
                # Each key allows 20 requests per minute, so the recommended parallelism matches the key count.
                recommended = available_keys
                if max_workers > recommended:
                    logger.info(f"[KeyPool] Detected {available_keys} available keys, adjusting parallelism: {max_workers} -> {recommended}")
                    max_workers = recommended
                else:
                    logger.info(f"[KeyPool] {available_keys} available keys, parallelism: {max_workers}")
        
        logger.info(f"Parallel mode: {max_workers} workers")
    else:
        max_workers = 0  # Serial.
    
    # Determine the output subdirectory from the task type and client type.
    task_output_root = args.output_dir / args.task
    if model_config.client_type == LLMClientType.VLLM:
        output_subdir = task_output_root / "local"
    else:
        output_subdir = task_output_root / "api"
    
    # Create the evaluator and run it.
    if args.task == "repair":
        evaluator = RepairEvaluator(
            client=client,
            output_dir=output_subdir,
            debug=args.debug,
            prompt_language="en",
            prompt_version=args.prompt_version,
            repair_level=args.repair_level,
            max_workers=max_workers,
        )
    else:
        evaluator = BugDetectionEvaluator(
            client=client,
            output_dir=output_subdir,
            debug=args.debug,
            max_workers=max_workers,
            use_batch=args.batch,
        )
    
    report = evaluator.evaluate_dataset(
        dataset_path=args.dataset,
        limit=args.limit,
        save_results=True,
        resume=not args.no_resume
    )
    
    # Print report.
    report.print_summary()
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
