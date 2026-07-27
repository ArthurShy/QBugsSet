"""
Local vLLM client.

Used to call a locally deployed vLLM service, with support for:
- OpenAI-compatible APIs
- Chain-of-thought parsing for reasoning models such as DeepSeek-R1
- Batch requests
"""

import time
import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

from .base_client import BaseClient, LLMResponse, LLMClientType, LLMModelConfig

logger = logging.getLogger(__name__)


# Local vLLM model configurations.
VLLM_MODELS = {
    "qwen2.5-7b-local": LLMModelConfig(
        name="qwen2.5-7b-local",
        model_name="models/qwen/Qwen2.5-7B",
        client_type=LLMClientType.VLLM,
        base_url="http://localhost:8000/v1",
        batch_size=20,
        description="Local Qwen2.5-7B model"
    ),
    "qwen2.5-14b-local": LLMModelConfig(
        name="qwen2.5-14b-local",
        model_name="models/qwen/Qwen2.5-14B",
        client_type=LLMClientType.VLLM,
        base_url="http://localhost:8000/v1",
        batch_size=10,
        description="Local Qwen2.5-14B model"
    ),
    "qwen2.5-32b-local": LLMModelConfig(
        name="qwen2.5-32b-local",
        model_name="models/qwen/Qwen2.5-32B",
        client_type=LLMClientType.VLLM,
        base_url="http://localhost:8000/v1",
        batch_size=5,
        description="Local Qwen2.5-32B model"
    ),
    "qwen2.5-72b-local": LLMModelConfig(
        name="qwen2.5-72b-local",
        model_name="models/qwen/Qwen2.5-72B",
        client_type=LLMClientType.VLLM,
        base_url="http://localhost:8000/v1",
        batch_size=3,
        timeout=180,
        description="Local Qwen2.5-72B model"
    ),
    "qwen3-14b-local": LLMModelConfig(
        name="qwen3-14b-local",
        model_name="models/qwen/Qwen3-14B",
        client_type=LLMClientType.VLLM,
        base_url="http://localhost:8000/v1",
        batch_size=10,
        reasoning_parser=True,
        description="Local Qwen3-14B model"
    ),
    "qwen3-32b-local": LLMModelConfig(
        name="qwen3-32b-local",
        model_name="models/qwen/Qwen3-32B",
        client_type=LLMClientType.VLLM,
        base_url="http://localhost:8000/v1",
        batch_size=5,
        reasoning_parser=True,
        description="Local Qwen3-32B model"
    ),
    "deepseek-r1-distill-qwen-32b-local": LLMModelConfig(
        name="deepseek-r1-distill-qwen-32b-local",
        model_name="models/deepseek/DeepSeek-R1-Distill-Qwen-32B",
        client_type=LLMClientType.VLLM,
        base_url="http://localhost:8000/v1",
        batch_size=5,
        timeout=180,
        reasoning_parser=True,
        description="DeepSeek-R1-Distill-Qwen-32B"
    ),
    "deepseek-r1-distill-llama-70b-local": LLMModelConfig(
        name="deepseek-r1-distill-llama-70b-local",
        model_name="models/deepseek/DeepSeek-R1-Distill-Llama-70B",
        client_type=LLMClientType.VLLM,
        base_url="http://localhost:8000/v1",
        batch_size=3,
        timeout=180,
        reasoning_parser=True,
        description="DeepSeek-R1-Distill-Llama-70B"
    ),
    "llama3.3-70b-local": LLMModelConfig(
        name="llama3.3-70b-local",
        model_name="models/llama/Llama3.3-70B",
        client_type=LLMClientType.VLLM,
        base_url="http://localhost:8000/v1",
        batch_size=2,
        timeout=180,
        description="Local Llama3.3-70B model"
    ),
}


class VLLMClient(BaseClient):
    """
    Local vLLM client.
    
    Calls a local vLLM service through an OpenAI-compatible API.
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "vllm",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        reasoning_parser: bool = False,
        **kwargs
    ):
        """
        Initialize the vLLM client.
        
        Args:
            model_name: Model name or local path.
            base_url: vLLM service URL.
            api_key: API key. vLLM does not usually require a real key.
            temperature: Generation temperature.
            max_tokens: Maximum output tokens.
            timeout: Request timeout in seconds.
            reasoning_parser: Whether to parse chain-of-thought content for reasoning models such as DeepSeek-R1.
        """
        super().__init__(model_name, temperature, max_tokens, timeout, **kwargs)
        
        self.base_url = base_url
        self.api_key = api_key
        self.reasoning_parser = reasoning_parser
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        
        logger.info(f"  Service URL: {base_url}")
        logger.info(f"  Reasoning parsing: {reasoning_parser}")
    
    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        Send a chat request to the vLLM service.
        
        Args:
            messages: List of messages.
            
        Returns:
            An LLMResponse object.
        """
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_time = time.time() - start_time
            content = response.choices[0].message.content or ""
            
            # Parse chain-of-thought content for reasoning models.
            if self.reasoning_parser:
                content = self._parse_reasoning_content(content)
            
            # Extract token usage details.
            token_usage = None
            if response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return LLMResponse(
                content=content,
                success=True,
                token_usage=token_usage,
                response_time=response_time,
                raw_response=response
            )
            
        except APIConnectionError as e:
            logger.error(f"vLLM connection failed: {e}")
            return LLMResponse(
                content="",
                success=False,
                error=f"Connection failed: {str(e)}",
                response_time=time.time() - start_time
            )
        except APIError as e:
            logger.error(f"vLLM API error: {e}")
            return LLMResponse(
                content="",
                success=False,
                error=f"API error: {str(e)}",
                response_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"vLLM request error: {e}")
            return LLMResponse(
                content="",
                success=False,
                error=f"Request error: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def _parse_reasoning_content(self, content: str) -> str:
        """
        Parse reasoning-model output and extract the final answer.
        
        Reasoning models such as DeepSeek-R1 may emit chain-of-thought
        content inside <think>...</think>, so we extract the final answer
        outside those tags.
        
        Args:
            content: Raw output content.
            
        Returns:
            Parsed final answer.
        """
        import re
        
        # Remove content wrapped in <think>...</think> tags.
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', content, flags=re.DOTALL)
        
        # Trim extra whitespace.
        cleaned = cleaned.strip()
        
        if not cleaned and content:
            # If the parsed output is empty, fall back to the original content.
            logger.warning("Reasoning parser produced empty content; returning original content")
            return content
        
        return cleaned
    
    def is_available(self) -> bool:
        """
        Check whether the vLLM service is available.
        
        Returns:
            True if the service is available.
        """
        try:
            # Try listing models to check service health.
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.warning(f"vLLM service unavailable: {e}")
            return False
    
    def get_loaded_models(self) -> List[str]:
        """
        Return the list of models currently loaded in the vLLM service.
        
        Returns:
            A list of model names.
        """
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"Failed to fetch model list: {e}")
            return []
