#!/usr/bin/env python3
"""
AliCloud DashScope API client.

Supports Qwen-series models through an OpenAI-compatible interface.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI, Stream
from openai import APIError, APIConnectionError, RateLimitError
from openai.types.chat import ChatCompletionChunk

from .base_client import BaseClient, LLMResponse, LLMClientType, LLMModelConfig

logger = logging.getLogger(__name__)


# DashScope API configuration.
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MINIMAX_M25_MODEL_NAME = os.environ.get("ALICLOUD_MINIMAX_M25_MODEL_NAME", "MiniMax-M2.5")

# Model configurations.
ALICLOUD_MODELS = {
    "qwen3-max-2026-01-23": LLMModelConfig(
        name="qwen3-max-2026-01-23",
        model_name="qwen3-max-2026-01-23",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=180,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen3-Max-2026-01-23"
    ),
    "qwen3-max-2026-01-23-thinking": LLMModelConfig(
        name="qwen3-max-2026-01-23-thinking",
        model_name="qwen3-max-2026-01-23",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=3,
        timeout=240,
        max_tokens=32768,
        reasoning_parser=True,
        description="Qwen3-Max-2026-01-23 Thinking (AliCloud DashScope, provider-default thinking)"
    ),
    "qwen3-30b-a3b-instruct-2507": LLMModelConfig(
        name="qwen3-30b-a3b-instruct-2507",
        model_name="qwen3-30b-a3b-instruct-2507",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=150,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen3-30B-A3B-Instruct-2507 (MoE reasoning model)"
    ),
    "qwen3-next-80b-a3b-instruct": LLMModelConfig(
        name="qwen3-next-80b-a3b-instruct",
        model_name="qwen3-next-80b-a3b-instruct",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=180,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen3-Next-80B-A3B-Instruct (MoE reasoning model)"
    ),
    "deepseek-v3.2": LLMModelConfig(
        name="deepseek-v3.2",
        model_name="deepseek-v3.2",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=180,
        max_tokens=8192,
        reasoning_parser=False,
        description="DeepSeek-V3.2 (via AliCloud DashScope)"
    ),
    "deepseek-v3.2-thinking": LLMModelConfig(
        name="deepseek-v3.2-thinking",
        model_name="deepseek-v3.2",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=3,
        timeout=300,
        max_tokens=16000,
        reasoning_parser=True,
        description="DeepSeek-V3.2 Thinking (reasoning mode via AliCloud DashScope)"
    ),
    "qwen3-235b-a22b-instruct-2507": LLMModelConfig(
        name="qwen3-235b-a22b-instruct-2507",
        model_name="qwen3-235b-a22b-instruct-2507",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=180,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen3-235B-A22B-Instruct-2507 (large MoE model)"
    ),
    "qwen3-14b": LLMModelConfig(
        name="qwen3-14b",
        model_name="qwen3-14b",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=120,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen3-14B (14B-parameter model)"
    ),
    "qwen3-8b": LLMModelConfig(
        name="qwen3-8b",
        model_name="qwen3-8b",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=120,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen3-8B (8B-parameter model)"
    ),
    "glm-5": LLMModelConfig(
        name="glm-5",
        model_name="glm-5",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=180,
        max_tokens=8192,
        reasoning_parser=False,
        description="GLM-5 (via AliCloud DashScope)"
    ),
    "minmax-m2.5": LLMModelConfig(
        name="minmax-m2.5",
        model_name=MINIMAX_M25_MODEL_NAME,
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=3,
        timeout=240,
        max_tokens=16384,
        reasoning_parser=True,
        description="MiniMax-M2.5 (AliCloud DashScope reasoning model; override actual model_name with ALICLOUD_MINIMAX_M25_MODEL_NAME)"
    ),
    "minimax-m2.5": LLMModelConfig(
        name="minimax-m2.5",
        model_name=MINIMAX_M25_MODEL_NAME,
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=3,
        timeout=240,
        max_tokens=16384,
        reasoning_parser=True,
        description="MiniMax-M2.5 (AliCloud DashScope reasoning model alias; override actual model_name with ALICLOUD_MINIMAX_M25_MODEL_NAME)"
    ),
    "qwen-3.5-35b-a3b": LLMModelConfig(
        name="qwen-3.5-35b-a3b",
        model_name="qwen3.5-35b-a3b",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=150,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen-3.5-35B-A3B (AliCloud DashScope)"
    ),
    "qwen3.5-35b-a3b": LLMModelConfig(
        name="qwen3.5-35b-a3b",
        model_name="qwen3.5-35b-a3b",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=150,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen3.5-35B-A3B (AliCloud DashScope)"
    ),
    "qwen-3.5-122b-a10b": LLMModelConfig(
        name="qwen-3.5-122b-a10b",
        model_name="qwen3.5-122b-a10b",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=180,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen-3.5-122B-A10B (AliCloud DashScope)"
    ),
    "qwen3.5-122b-a10b": LLMModelConfig(
        name="qwen3.5-122b-a10b",
        model_name="qwen3.5-122b-a10b",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=180,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen3.5-122B-A10B (AliCloud DashScope)"
    ),
    "qwen-3.5-27b": LLMModelConfig(
        name="qwen-3.5-27b",
        model_name="qwen3.5-27b",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=150,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen-3.5-27B (AliCloud DashScope)"
    ),
    "qwen3.5-27b": LLMModelConfig(
        name="qwen3.5-27b",
        model_name="qwen3.5-27b",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=5,
        timeout=150,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen3.5-27B (AliCloud DashScope)"
    ),
    "qwen-3.5-397b-a17b": LLMModelConfig(
        name="qwen-3.5-397b-a17b",
        model_name="qwen3.5-397b-a17b",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=3,
        timeout=240,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen-3.5-397B-A17B (AliCloud DashScope)"
    ),
    "qwen3.5-397b-a17b": LLMModelConfig(
        name="qwen3.5-397b-a17b",
        model_name="qwen3.5-397b-a17b",
        client_type=LLMClientType.ALICLOUD,
        base_url=DASHSCOPE_BASE_URL,
        api_key_env="AliCloud_API_KEY",
        batch_size=3,
        timeout=240,
        max_tokens=8192,
        reasoning_parser=False,
        description="Qwen3.5-397B-A17B (AliCloud DashScope)"
    ),
}


class AliCloudClient(BaseClient):
    """
    AliCloud DashScope API client.
    
    Calls Qwen-series models through an OpenAI-compatible interface.
    """
    
    def __init__(
        self,
        model_name: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: str = DASHSCOPE_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        reasoning_parser: bool = False,
        thinking_budget: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        use_stream: Optional[bool] = None,
        **kwargs
    ):
        """
        Initialize the AliCloud client.
        
        Args:
            model_name: Model name.
            api_key: API key, read from AliCloud_API_KEY by default.
            base_url: API URL.
            temperature: Generation temperature.
            max_tokens: Maximum output tokens.
            timeout: Request timeout in seconds.
            reasoning_parser: Whether to parse chain-of-thought content for reasoning models.
            thinking_budget: Thinking budget. Only applies when thinking is enabled and streaming is used.
            max_retries: Maximum retry count.
            retry_delay: Retry delay in seconds.
        """
        super().__init__(model_name, temperature, max_tokens, timeout, **kwargs)
        
        self.api_key = api_key or os.getenv('AliCloud_API_KEY')
        if not self.api_key:
            raise ValueError("AliCloud API key not provided. Please set the AliCloud_API_KEY environment variable")
        
        self.base_url = base_url
        self.reasoning_parser = reasoning_parser
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_error_summary: Optional[str] = None
        # Reasoning models default to streaming; non-reasoning models default to non-streaming.
        self.use_stream = use_stream if use_stream is not None else reasoning_parser
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            timeout=timeout
        )
        
        logger.info(f"  Service URL: {base_url}")
        logger.info(f"  Streaming: {'enabled' if self.use_stream else 'disabled'}")
        logger.info(f"  Reasoning parsing: {reasoning_parser}")
        logger.info(f"  Thinking budget: {thinking_budget if thinking_budget is not None else 'provider-default'}")
    
    def chat(
        self, 
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None
    ) -> LLMResponse:
        """
        Send a chat request to the DashScope API.
        
        Args:
            messages: List of messages.
            response_format: Response format config (for example {"type": "json_object"}).
            
        Returns:
            An LLMResponse object.
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Build request parameters.
                request_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": self.use_stream
                }
                
                if response_format:
                    request_params["response_format"] = response_format
                
                # Enable thinking mode based on reasoning_parser settings.
                # DashScope requires enable_thinking to be false for non-streaming calls.
                enable_thinking = self.reasoning_parser and self.use_stream
                request_params["extra_body"] = {"enable_thinking": enable_thinking}
                if enable_thinking and self.thinking_budget is not None:
                    request_params["extra_body"]["thinking_budget"] = self.thinking_budget
                
                if self.use_stream:
                    # Streaming request.
                    result = self._stream_request(request_params)
                    response_time = time.time() - start_time
                    return LLMResponse(
                        content=result["content"],
                        success=True,
                        token_usage=result.get("token_usage"),
                        response_time=response_time,
                        thinking_content=result.get("thinking_content")
                    )
                else:
                    # Non-streaming request.
                    response = self.client.chat.completions.create(**request_params)
                    response_time = time.time() - start_time
                    message = response.choices[0].message
                    content = message.content or ""
                    
                    # Extract chain-of-thought content for reasoning models.
                    thinking_content = None
                    if self.reasoning_parser:
                        if hasattr(message, 'model_extra') and message.model_extra:
                            thinking_content = message.model_extra.get('reasoning_content')
                        if not thinking_content:
                            thinking_content = getattr(message, 'reasoning_content', None)
                        if not thinking_content and '<think>' in content:
                            content, thinking_content = self._parse_reasoning_content(content)
                    
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
                        raw_response=response,
                        thinking_content=thinking_content
                    )
                
            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit; retrying after {wait_time}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                    
            except APIConnectionError as e:
                last_error = e
                logger.error(f"DashScope connection failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                    
            except APIError as e:
                last_error = e
                logger.error(f"DashScope API error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                    
            except Exception as e:
                last_error = e
                logger.error(f"DashScope request error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
        
        return LLMResponse(
            content="",
            success=False,
            error=f"Request failed: {str(last_error)}",
            response_time=time.time() - start_time
        )
    
    def _stream_request(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Streaming request implementation.
        
        Args:
            request_params: Request parameters.
            
        Returns:
            A dictionary containing content, thinking_content, and token_usage.
        """
        response = self.client.chat.completions.create(**request_params)
        
        full_content = ""
        full_reasoning = ""
        token_usage = None
        
        for chunk in response:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            # Accumulate answer content.
            if delta.content:
                full_content += delta.content
            
            # Accumulate reasoning content in DashScope format.
            if self.reasoning_parser:
                if hasattr(delta, 'model_extra') and delta.model_extra:
                    reasoning = delta.model_extra.get('reasoning_content', '')
                    if reasoning:
                        full_reasoning += reasoning
            
            # Get token usage information, usually from the final chunk.
            if hasattr(chunk, 'usage') and chunk.usage:
                token_usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens
                }
        
        # If reasoning content is not present in the stream, try parsing <think> tags.
        thinking_content = full_reasoning if full_reasoning else None
        if self.reasoning_parser and not thinking_content and '<think>' in full_content:
            full_content, thinking_content = self._parse_reasoning_content(full_content)
        
        return {
            "content": full_content,
            "thinking_content": thinking_content,
            "token_usage": token_usage
        }
    
    def _parse_reasoning_content(self, content: str) -> tuple:
        """
        Parse reasoning-model output and extract the final answer and chain-of-thought.
        
        Reasoning models such as Qwen3 may emit chain-of-thought content inside
        <think>...</think>. We extract the final answer while preserving the
        thought content separately.
        
        Args:
            content: Raw output content.
            
        Returns:
            Tuple of (final answer, chain-of-thought content).
        """
        import re
        
        # Extract content inside <think>...</think> tags.
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, content, flags=re.DOTALL)
        thinking_content = '\n'.join(think_matches).strip() if think_matches else None
        
        # Remove content inside <think>...</think> tags.
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Trim extra whitespace.
        cleaned = cleaned.strip()
        
        if not cleaned and content:
            logger.warning("Reasoning parser produced empty content; returning original content")
            return content, thinking_content
        
        return cleaned, thinking_content
    
    def is_available(self) -> bool:
        """
        Check whether the DashScope service is available.
        
        Returns:
            True if the service is available.
        """
        try:
            self.last_error_summary = None
            request_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "stream": self.use_stream,
            }
            enable_thinking = self.reasoning_parser and self.use_stream
            request_params["extra_body"] = {"enable_thinking": enable_thinking}

            if self.use_stream:
                result = self._stream_request(request_params)
                return bool(result.get("content"))

            response = self.client.chat.completions.create(**request_params)
            return bool(response.choices[0].message.content)
        except Exception as e:
            self.last_error_summary = str(e)
            logger.warning(f"DashScope service unavailable: {e}")
            return False


# Convenience alias.
DashScopeClient = AliCloudClient
