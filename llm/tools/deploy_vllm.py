#!/usr/bin/env python3
"""
Helper script for deploying vLLM services.

Supports:
- Generating vLLM startup commands
- Checking service status
- Listing local model configurations

Usage:
    # Generate a startup command
    python llm/tools/deploy_vllm.py --model qwen2.5-7b-local --generate-cmd

    # Check service status
    python llm/tools/deploy_vllm.py --check

    # Show all local model configs
    python llm/tools/deploy_vllm.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import LLM_MODELS, LLMClientType, get_llm_model_config


# Default vLLM configuration.
VLLM_DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "tensor_parallel_size": 1,  # Number of GPUs.
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.9,
    "dtype": "auto",
}


def generate_vllm_command(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    extra_args: str = ""
) -> str:
    """
    Generate a vLLM startup command.
    
    Args:
        model_path: Model path.
        host: Bind address.
        port: Port number.
        tensor_parallel_size: Tensor parallel size, usually the GPU count.
        max_model_len: Maximum model length.
        gpu_memory_utilization: GPU memory utilization ratio.
        dtype: Data type.
        extra_args: Extra CLI arguments.
        
    Returns:
        vLLM startup command.
    """
    cmd = f"""python -m vllm.entrypoints.openai.api_server \\
    --model {model_path} \\
    --host {host} \\
    --port {port} \\
    --tensor-parallel-size {tensor_parallel_size} \\
    --max-model-len {max_model_len} \\
    --gpu-memory-utilization {gpu_memory_utilization} \\
    --dtype {dtype}"""
    
    if extra_args:
        cmd += f" \\\n    {extra_args}"
    
    return cmd


def check_vllm_service(url: str = "http://localhost:8000") -> bool:
    """
    Check whether the vLLM service is running.
    
    Args:
        url: Service URL.
        
    Returns:
        True if the service is running.
    """
    try:
        import requests
        response = requests.get(f"{url}/v1/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def list_local_models():
    """List all local vLLM model configurations."""
    print("\n" + "=" * 70)
    print("Local vLLM model configurations")
    print("=" * 70)
    
    for name, cfg in LLM_MODELS.items():
        if cfg.client_type == LLMClientType.VLLM:
            print(f"\n📦 {name}")
            print(f"   Model path: {cfg.model_name}")
            print(f"   Service URL: {cfg.base_url}")
            print(f"   Batch size: {cfg.batch_size}")
            print(f"   Timeout: {cfg.timeout}s")
            print(f"   Reasoning parsing: {cfg.reasoning_parser}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM deployment helper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model name, resolved through configuration"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Directly specify the model path and override configuration"
    )
    parser.add_argument(
        "--generate-cmd",
        action="store_true",
        help="Generate the vLLM startup command"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check vLLM service status"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all local model configurations"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="vLLM service URL"
    )
    
    # vLLM config parameters.
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tp", type=int, default=1, help="tensor-parallel-size")
    parser.add_argument("--max-len", type=int, default=32768)
    parser.add_argument("--gpu-util", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--extra", type=str, default="", help="Extra vLLM arguments")
    
    args = parser.parse_args()
    
    # List models.
    if args.list:
        list_local_models()
        return
    
    # Check service.
    if args.check:
        print(f"Checking vLLM service: {args.url}")
        if check_vllm_service(args.url):
            print("vLLM service is running")
            
            # Try to fetch loaded models.
            try:
                import requests
                response = requests.get(f"{args.url}/v1/models", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("data", [])
                    if models:
                        print("Loaded models:")
                        for m in models:
                            print(f"   - {m.get('id', 'unknown')}")
            except Exception:
                pass
        else:
            print("vLLM service is not running or cannot be reached")
        return
    
    # Generate startup command.
    if args.generate_cmd:
        model_path = args.model_path
        
        if not model_path and args.model:
            try:
                cfg = get_llm_model_config(args.model)
                if cfg.client_type != LLMClientType.VLLM:
                    print(f"Error: {args.model} is not a local vLLM model")
                    sys.exit(1)
                model_path = cfg.model_name
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)
        
        if not model_path:
            print("Error: please provide --model or --model-path")
            sys.exit(1)
        
        cmd = generate_vllm_command(
            model_path=model_path,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_len,
            gpu_memory_utilization=args.gpu_util,
            dtype=args.dtype,
            extra_args=args.extra
        )
        
        print("\n" + "=" * 70)
        print("vLLM startup command")
        print("=" * 70)
        print(cmd)
        print("=" * 70)
        
        # Generate nohup background command.
        print("\nBackground run with nohup:")
        print(f"nohup {cmd.replace(chr(10), ' ').replace('    ', '')} > vllm.log 2>&1 &")
        
        # Generate screen command.
        print("\nRun with screen:")
        print(f"screen -S vllm -dm bash -c '{cmd.replace(chr(10), ' ').replace('    ', '')}'")
        
        return
    
    # Show help by default.
    parser.print_help()


if __name__ == "__main__":
    main()
