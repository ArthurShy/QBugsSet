#!/usr/bin/env python3
"""
OpenRouter API client.

Supports calling a variety of LLMs through OpenRouter, including free models
such as Qwen3-Coder.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
import requests

from .base_client import BaseClient, LLMResponse, LLMClientType, LLMModelConfig

# Try importing the key pool (optional dependency).
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llm.key_pool import get_openrouter_key_pool, OpenRouterKeyPool
    KEY_POOL_AVAILABLE = True
except ImportError:
    KEY_POOL_AVAILABLE = False
    OpenRouterKeyPool = None


# OpenRouter API configuration.
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_ENDPOINT = "/chat/completions"

# Default settings.
DEFAULT_MODEL = "qwen/qwen3-coder:free"
DEFAULT_CONNECT_TIMEOUT = 30
DEFAULT_READ_TIMEOUT = 300
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 7000

# Model configurations.
OPENROUTER_MODELS = {
    "qwen3-coder-free": {
        "config": LLMModelConfig(
            name="qwen3-coder-free",
            model_name="qwen/qwen3-coder:free",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=10,
            timeout=120,
            max_tokens=7000,
            description="Qwen3 Coder 480B A35B (via OpenRouter)"
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 131072,
        "is_free": True  # Free model; enable key-pool rotation.
    },
    "gemini-3-flash-preview": {
        "config": LLMModelConfig(
            name="gemini-3-flash-preview",
            model_name="google/gemini-3-flash-preview",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=10,
            timeout=120,
            max_tokens=8192,
            temperature=0.0,
            description="Google Gemini 3 Flash Preview (via OpenRouter)"
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 1000000
    },
    "qwen3-30b-a3b-thinking-2507": {
        "config": LLMModelConfig(
            name="qwen3-30b-a3b-thinking-2507",
            model_name="qwen/qwen3-30b-a3b-thinking-2507",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=10,
            timeout=120,
            max_tokens=8192,
            temperature=0.0,
            description="Qwen3 30B A3B Thinking (via OpenRouter)"
        ),
        "display_name": "qwen3-30b-a3b-thinking-2507",
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 131072
    },
    "qwen3-30b-a3b-instruct-2507": {
        "config": LLMModelConfig(
            name="qwen3-30b-a3b-instruct-2507",
            model_name="qwen/qwen3-30b-a3b-instruct-2507",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=10,
            timeout=120,
            max_tokens=8192,
            temperature=0.0,
            description="Qwen3 30B A3B Instruct (via OpenRouter)"
        ),
        "display_name": "qwen3-30b-a3b-instruct-2507",
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 131072
    },
    "qwen3-235b-a22b-instruct-2507": {
        "config": LLMModelConfig(
            name="qwen3-235b-a22b-instruct-2507",
            model_name="qwen/qwen3-235b-a22b-2507",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=5,
            timeout=180,
            max_tokens=8192,
            temperature=0.0,
            description="Qwen3 235B A22B Instruct (via OpenRouter)"
        ),
        "display_name": "qwen3-235b-a22b-instruct-2507",
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 131072
    },
    "ministral-14b-2512": {
        "config": LLMModelConfig(
            name="ministral-14b-2512",
            model_name="mistralai/ministral-14b-2512",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=10,
            timeout=120,
            max_tokens=8192,
            temperature=0.0,
            description="Mistral Ministral 14B 2512 (via OpenRouter)"
        ),
        "display_name": "Ministral3-14B",
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 131072
    },
    "qwen3-14b": {
        "config": LLMModelConfig(
            name="qwen3-14b",
            model_name="qwen/qwen3-14b",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=10,
            timeout=120,
            max_tokens=8192,
            temperature=0.0,
            description="Qwen3 14B (via OpenRouter)"
        ),
        "display_name": "qwen3-14b",
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 131072
    },
    "gpt-5.2": {
        "config": LLMModelConfig(
            name="gpt-5.2",
            model_name="openai/gpt-5.2",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=5,
            timeout=180,
            max_tokens=8192,
            temperature=0.0,
            description="OpenAI GPT-5.2 (via OpenRouter)"
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 128000,
        "reasoning_effort": "none"
    },
    "gpt-5.3": {
        "config": LLMModelConfig(
            name="gpt-5.3",
            model_name="openai/gpt-5.3-chat",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=5,
            timeout=180,
            max_tokens=8192,
            temperature=0.0,
            description="OpenAI GPT-5.3 Chat (via OpenRouter)"
        ),
        "display_name": "gpt-5.3",
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 128000,
        "reasoning_effort": "none"
    },
    "claude-sonnet-4.6": {
        "config": LLMModelConfig(
            name="claude-sonnet-4.6",
            model_name="anthropic/claude-sonnet-4.6",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=5,
            timeout=180,
            max_tokens=8192,
            temperature=0.0,
            description="Anthropic Claude Sonnet 4.6 (via OpenRouter, reasoning disabled)"
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 1000000,
        "reasoning": {
            "effort": "none"
        }
    },
    "gemini-3-pro-preview": {
        "config": LLMModelConfig(
            name="gemini-3-pro-preview",
            model_name="google/gemini-3-pro-preview",
            client_type=LLMClientType.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            batch_size=5,
            timeout=180,
            max_tokens=8192,
            temperature=0.0,
            description="Google Gemini 3 Pro Preview (via OpenRouter)"
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 1000000,
        "reasoning_effort": "minimal"
    },
}


class OpenRouterClient(BaseClient):
    """
    OpenRouter API client.
    
    Supports a variety of LLMs via OpenRouter and uses streaming responses to
    reduce timeout risk.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        # BaseClient compatibility arguments.
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        # 1. Resolve the final model name.
        actual_model = model_name or model
        
        # 2. Load the predefined configuration, allowing alias lookup.
        model_def = None
        for key, value in OPENROUTER_MODELS.items():
            if value["config"].model_name == actual_model or key == actual_model:
                model_def = value
                break
        
        if not model_def:
            # If the model is unknown, use the default configuration.
            logging.warning(f"Unknown model '{actual_model}', using default configuration")
            model_def = OPENROUTER_MODELS["qwen3-coder-free"]
            
        model_config = model_def["config"]
        retry_config = model_def["retry"]
            
        # 3. Resolve final parameters, preferring explicit arguments.
        final_max_tokens = max_tokens or max_output_tokens or model_config.max_tokens
        final_timeout = timeout or model_config.timeout
        
        # 4. Initialize the parent class.
        super().__init__(
            model_name=actual_model,
            temperature=temperature,
            max_tokens=final_max_tokens,
            timeout=final_timeout,
            **kwargs
        )
        
        # 5. Set OpenRouter-specific attributes, including key-pool support for free models.
        is_free_model = model_def.get("is_free", False)
        self.use_key_pool = kwargs.get("use_key_pool", is_free_model) and KEY_POOL_AVAILABLE and is_free_model
        self.key_pool = None  # type: Any
        
        if self.use_key_pool and KEY_POOL_AVAILABLE:
            # Use the key pool to manage multiple API keys (free models only).
            self.key_pool = get_openrouter_key_pool(
                key_prefix="OPENROUTER_API_KEY",
                daily_limit=kwargs.get("daily_limit", 1000),
                minute_limit=kwargs.get("minute_limit", 20)
            )
            
            # Automatically exclude exhausted keys when enabled.
            if kwargs.get("auto_exclude_exhausted", True):
                min_balance = kwargs.get("min_balance", 0.0)
                excluded = self.key_pool.check_and_exclude_exhausted_keys(min_remaining=min_balance)
                if excluded > 0:
                    logging.info(f"[KeyPool] Excluded {excluded} keys with insufficient balance")
            
            # Fetch a single key for validation during initialization without reserving quota.
            self.api_key = self.key_pool.get_key()
            if not self.api_key:
                # If the key pool is empty or unusable, fall back to a single key.
                self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
                self.use_key_pool = False
                logging.warning("No available key in the key pool; falling back to a single API key")
            else:
                available = self.key_pool.get_available_count()
                logging.info(f"[KeyPool] Using key pool with round-robin rotation; {available} keys available")
        else:
            self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided. Please set OPENROUTER_API_KEY or OPENROUTER_API_KEY_1/2/3...")
            
        self.base_url = OPENROUTER_API_BASE_URL
        self.endpoint = OPENROUTER_API_ENDPOINT
        self.model = actual_model
        self.max_output_tokens = final_max_tokens
        
        # 6. Configure retry behavior.
        self.max_retries = kwargs.get("max_retries", retry_config["max_retries"])
        self.retry_delay = kwargs.get("retry_delay", retry_config["retry_delay"])
        self.max_retries_chunked = kwargs.get("max_retries_chunked", retry_config["max_retries_chunked"])
        
        # 7. Whether to use streaming. Some models such as Gemini have streaming compatibility issues.
        self.use_stream = model_def.get("use_stream", True)
        
        # 8. Set display name for output paths and backward compatibility.
        self.display_name = model_def.get("display_name", model_config.name)
        
        # 9. Reasoning effort level used by models such as o1/o3/gpt-5.2.
        self.reasoning_effort = model_def.get("reasoning_effort", None)
        self.reasoning = model_def.get("reasoning", None)

        # 10. Initialize the session.
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/quantum-research",  # Recommended by OpenRouter.
            "X-Title": "Quantum Bug Detection",  # Application name.
            "Connection": "keep-alive"
        })
        self.headers = dict(self.session.headers)
        
        logging.info(f"  Streaming: enabled")
        logging.info(f"  Retry policy: max={self.max_retries}, delay={self.retry_delay}s")
    
    def _refresh_key_from_pool(self) -> bool:
        """Fetch a new available key from the key pool and return whether it succeeded."""
        if not self.use_key_pool or not self.key_pool:
            return False
        
        new_key = self.key_pool.get_key()
        if new_key and new_key != self.api_key:
            self.api_key = new_key
            self._update_session_auth()
            logging.info(f"[KeyPool] Switched to a new key; {self.key_pool.get_available_count()} still available")
            return True
        return False
    
    def _update_session_auth(self):
        """Update the session authorization header."""
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.headers = dict(self.session.headers)
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        response_format: Optional[Dict[str, str]] = None
    ) -> LLMResponse:
        """
        Send a chat request (BaseClient interface).
        
        Uses streaming to reduce timeout risk and returns an LLMResponse.
        Supports automatic key-pool failover when rate limits or credit exhaustion occur.
        """
        start_time = time.time()
        last_error = None
        current_key = self.api_key
        
        total_attempts = max(self.max_retries, self.max_retries_chunked)
        
        for attempt in range(total_attempts):
            try:
                # Stop retrying if normal retries are exhausted for non-chunked errors.
                if attempt >= self.max_retries and last_error and not isinstance(last_error, requests.exceptions.ChunkedEncodingError):
                    break
                
                # Key pool: acquire a key and reserve quota before each request.
                if self.use_key_pool and self.key_pool:
                    new_key = self.key_pool.acquire_key()  # Reserve quota + automatic rotation.
                    if new_key:
                        if new_key != current_key:
                            self.api_key = new_key
                            current_key = new_key
                            self._update_session_auth()
                    else:
                        # All keys are currently unavailable, so wait briefly.
                        wait_key = self.key_pool.wait_for_available(timeout=30, acquire=True)
                        if wait_key:
                            self.api_key = wait_key
                            current_key = wait_key
                            self._update_session_auth()
                        else:
                            # Timed out while waiting; no key is available.
                            return LLMResponse(
                                content="",
                                success=False,
                                error="All API keys are currently unavailable (per-minute or daily limits reached)",
                                response_time=time.time() - start_time
                            )
                    
                if self.use_stream:
                    result = self._stream_request(messages, response_format=response_format)
                else:
                    result = self._non_stream_request(messages, response_format=response_format)
                response_time = time.time() - start_time
                
                # Quota was already reserved by acquire_key, so no extra accounting is needed.
                
                content = result.get("content", "")
                thinking_content = result.get("thinking_content")
                usage = result.get("usage", {})
                
                # Basic OpenRouter token accounting. Cached-token stats are not supported here.
                token_usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                } if usage else None
                
                return LLMResponse(
                    content=content,
                    success=True,
                    token_usage=token_usage,
                    response_time=response_time,
                    thinking_content=thinking_content
                )
                
            except requests.exceptions.ChunkedEncodingError as e:
                last_error = e
                if attempt < self.max_retries_chunked - 1:
                    wait_time = min((2 ** attempt) * self.retry_delay, 120) 
                    logging.warning(f"Streaming interrupted, retrying ({attempt + 1}/{self.max_retries_chunked}): {e}")
                    time.sleep(wait_time)
                    self._reset_session()
                    continue
                else:
                    break
                    
            except requests.exceptions.HTTPError as e:
                last_error = e
                error_str = str(e)
                
                # Detect rate-limit or credit-exhaustion errors.
                is_rate_limit = "429" in error_str or "rate" in error_str.lower()
                is_credit_exhausted = "402" in error_str or "credit" in error_str.lower()
                
                if self.use_key_pool and self.key_pool and (is_rate_limit or is_credit_exhausted):
                    self.key_pool.record_error(current_key, error_str)
                    
                    # Try switching to a new key.
                    if self._refresh_key_from_pool():
                        current_key = self.api_key
                        logging.info(f"[KeyPool] Switching keys due to {'rate limit' if is_rate_limit else 'credit exhaustion'}")
                        continue  # Retry with the new key.
                
                logging.warning(f"HTTP error, retrying ({attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    self._reset_session()
                else:
                    break
                    
            except Exception as e:
                last_error = e
                logging.warning(f"Request failed, retrying ({attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    self._reset_session()
                else:
                    break
        
        return LLMResponse(
            content="",
            success=False,
            error=f"Request failed: {str(last_error)}",
            response_time=time.time() - start_time
        )
    
    def _stream_request(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Internal streaming request implementation."""
        url = f"{self.base_url}{self.endpoint}"
        
        # Parameter priority: method arguments > instance attributes.
        final_temp = temperature if temperature is not None else self.temperature
        final_max_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": final_temp,
            "max_tokens": final_max_tokens,
            "stream": True,
            **kwargs
        }

        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        elif self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        if response_format:
            payload["response_format"] = response_format
        
        start_time = time.time()
        timeout_tuple = (DEFAULT_CONNECT_TIMEOUT, max(self.timeout, DEFAULT_READ_TIMEOUT))
        
        response = self.session.post(url, json=payload, timeout=timeout_tuple, stream=True)
        response.raise_for_status()
        
        # Process the streaming response.
        full_content = ""
        full_reasoning = ""
        finish_reason = None
        usage = None
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line_text = line.decode('utf-8')
            if line_text.startswith("data: "):
                data_str = line_text[6:]
                if data_str.strip() == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data_str)
                    
                    if "usage" in chunk and chunk["usage"]:
                        usage = chunk["usage"]
                    
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            full_content += delta["content"]
                        # Handle reasoning fields produced by reasoning-capable models such as Gemini Pro.
                        if delta.get("reasoning"):
                            full_reasoning += delta["reasoning"]
                        if choices[0].get("finish_reason"):
                            finish_reason = choices[0]["finish_reason"]
                except json.JSONDecodeError:
                    continue
        
        # Strict separation: content is the final answer, reasoning is the thought process.
        final_content = full_content
        thinking_content = full_reasoning if full_reasoning else None
        
        return {
            "content": final_content,
            "thinking_content": thinking_content,
            "finish_reason": finish_reason or "stop",
            "usage": usage or {},
            "response_time": time.time() - start_time
        }
    
    def _non_stream_request(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Non-streaming request implementation for models with streaming issues, such as Gemini."""
        url = f"{self.base_url}{self.endpoint}"
        
        final_temp = temperature if temperature is not None else self.temperature
        final_max_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": final_temp,
            "max_tokens": final_max_tokens,
            "stream": False,
            **kwargs
        }

        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        elif self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        if response_format:
            payload["response_format"] = response_format
        
        start_time = time.time()
        timeout_tuple = (DEFAULT_CONNECT_TIMEOUT, max(self.timeout, DEFAULT_READ_TIMEOUT))
        
        response = self.session.post(url, json=payload, timeout=timeout_tuple)
        response.raise_for_status()
        
        data = response.json()
        
        content = ""
        finish_reason = None
        usage = data.get("usage", {})
        
        thinking_content = None
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content", "")
            reasoning = msg.get("reasoning", "")
            
            # Strict separation: content is the final answer, reasoning is the thought process.
            thinking_content = reasoning if reasoning else None
            
            finish_reason = data["choices"][0].get("finish_reason")
        
        return {
            "content": content,
            "thinking_content": thinking_content,
            "finish_reason": finish_reason or "stop",
            "usage": usage,
            "response_time": time.time() - start_time
        }
    
    def _reset_session(self):
        """Reset the HTTP session."""
        try:
            self.session.close()
        except:
            pass
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def is_available(self) -> bool:
        """Check whether the OpenRouter API is available, with optional key-pool rotation."""
        max_attempts = self.key_pool.get_available_count() if self.use_key_pool and self.key_pool else 1
        max_attempts = min(max_attempts, 3)  # Try at most 3 keys.
        
        for attempt in range(max_attempts):
            try:
                # Rotate through the key pool.
                if self.use_key_pool and self.key_pool and attempt > 0:
                    new_key = self.key_pool.get_key()
                    if new_key and new_key != self.api_key:
                        self.api_key = new_key
                        self._update_session_auth()
                        logging.info(f"[KeyPool] is_available switching keys for retry ({attempt + 1}/{max_attempts})")
                
                url = f"{self.base_url}{self.endpoint}"
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 50
                }
                response = self.session.post(url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                msg = result.get("choices", [{}])[0].get("message", {})
                # Thinking-model output may appear in the reasoning field instead of content.
                return bool(msg.get("content") or msg.get("reasoning"))
            except requests.exceptions.HTTPError as e:
                # On HTTP 429, try switching keys.
                if "429" in str(e) and self.use_key_pool and self.key_pool:
                    logging.warning(f"[KeyPool] Key rate-limited; trying another key ({attempt + 1}/{max_attempts})")
                    time.sleep(1)  # Brief pause before retrying.
                    continue
                logging.warning(f"OpenRouter API unavailable: {e}")
                return False
            except Exception as e:
                logging.warning(f"OpenRouter API unavailable: {e}")
                return False
        
        logging.warning("OpenRouter API unavailable: all keys were rate-limited")
        return False
