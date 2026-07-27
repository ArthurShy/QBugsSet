#!/usr/bin/env python3
"""
DeepSeek API client.

Supports the deepseek-chat and deepseek-reasoner models and uses
streaming responses to reduce timeout risk.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
import requests

from .base_client import BaseClient, LLMResponse, LLMClientType, LLMModelConfig


# DeepSeek API configuration.
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_ENDPOINT = "/chat/completions"

# Default settings.
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_CONNECT_TIMEOUT = 30
DEFAULT_READ_TIMEOUT = 300
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 8000
DEFAULT_SEED = 42

# Model configurations.
DEEPSEEK_MODELS = {
    "deepseek-chat": {
        "config": LLMModelConfig(
            name="deepseek-chat",
            model_name="deepseek-chat",
            client_type=LLMClientType.DEEPSEEK,
            api_key_env="DEEPSEEK_API_KEY",
            batch_size=10,
            timeout=60,
            max_tokens=8000,
            description="DeepSeek Chat API"
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6
        },
        "context_window": 128000
    },
    "deepseek-reasoner": {
        "config": LLMModelConfig(
            name="deepseek-reasoner",
            model_name="deepseek-reasoner",
            client_type=LLMClientType.DEEPSEEK,
            api_key_env="DEEPSEEK_API_KEY",
            batch_size=3,
            timeout=150,
            max_tokens=32000,
            reasoning_parser=True,
            description="DeepSeek R1 reasoning model"
        ),
        "retry": {
            "max_retries": 3,
            "retry_delay": 2,
            "max_retries_chunked": 8
        },
        "context_window": 128000
    },
}


# Backward-compatible constant.
MODEL_CONTEXT_WINDOW = {
    k: v["context_window"] 
    for k, v in DEEPSEEK_MODELS.items()
}


class DeepSeekClient(BaseClient):
    """
    DeepSeek API client.
    
    Supports the deepseek-chat and deepseek-reasoner models and uses
    streaming responses to reduce timeout risk.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        seed: int = 42,
        # BaseClient compatibility arguments.
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        # 1. Resolve the final model name.
        actual_model = model_name or model
        
        # 2. Load the predefined configuration.
        model_def = DEEPSEEK_MODELS.get(actual_model)
        if not model_def:
            # If the model is unknown, fall back to the default chat config but keep the new name.
            logging.warning(f"Unknown model '{actual_model}', using default configuration")
            model_config = DEEPSEEK_MODELS[DEFAULT_MODEL]["config"]
            retry_config = DEEPSEEK_MODELS[DEFAULT_MODEL]["retry"]
        else:
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
        
        # 5. Set DeepSeek-specific attributes.
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided. Please set the DEEPSEEK_API_KEY environment variable")
            
        self.base_url = DEEPSEEK_API_BASE_URL
        self.endpoint = DEEPSEEK_API_ENDPOINT
        self.seed = seed
        self.model = actual_model  # Important: set self.model for later use.
        self.is_reasoner = "reasoner" in actual_model.lower()
        self.max_output_tokens = final_max_tokens  # Alias kept for compatibility.
        
        # 6. Configure retry behavior.
        # Explicit retry arguments in kwargs override defaults.
        self.max_retries = kwargs.get("max_retries", retry_config["max_retries"])
        self.retry_delay = kwargs.get("retry_delay", retry_config["retry_delay"])
        self.max_retries_chunked = kwargs.get("max_retries_chunked", retry_config["max_retries_chunked"])

        # 7. Initialize the session.
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        })
        self.headers = dict(self.session.headers)
        
        logging.info(f"  Reasoning model: {self.is_reasoner}")
        logging.info(f"  Streaming: enabled")
        logging.info(f"  Retry policy: max={self.max_retries}, delay={self.retry_delay}s")
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        response_format: Optional[Dict[str, str]] = None
    ) -> LLMResponse:
        """
        Send a chat request (BaseClient interface).
        
        Uses streaming to reduce timeout risk and returns an LLMResponse.
        """
        start_time = time.time()
        last_error = None
        
        # Determine retry behavior (normal errors vs chunked transfer errors).
        # Everything is handled in one loop, but ChunkedEncodingError gets more attempts.
        total_attempts = max(self.max_retries, self.max_retries_chunked)
        
        for attempt in range(total_attempts):
            try:
                # Stop retrying if we already exhausted normal retries for non-chunked errors.
                if attempt >= self.max_retries and last_error and not isinstance(last_error, requests.exceptions.ChunkedEncodingError):
                    break
                    
                result = self._stream_request(messages, response_format=response_format)
                response_time = time.time() - start_time
                
                content = result.get("content", "")
                reasoning = result.get("reasoning_content", "")
                usage = result.get("usage", {})
                
                # Debugging: empty-response diagnostics.
                if not content:
                    logging.warning("Received empty response content")
                    logging.warning(f"  - reasoning_content length: {len(reasoning)}")
                    logging.warning(f"  - finish_reason: {result.get('finish_reason', 'N/A')}")
                    logging.warning(f"  - usage: {usage}")
                
                token_usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    # DeepSeek-specific fields.
                    "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
                    "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens", 0)
                } if usage else None
                
                return LLMResponse(
                    content=content,
                    success=True,
                    token_usage=token_usage,
                    response_time=response_time,
                    raw_response={"reasoning_content": reasoning} if reasoning else None
                )
                
            except requests.exceptions.ChunkedEncodingError as e:
                last_error = e
                # This is a common error, so we allow additional retries.
                if attempt < self.max_retries_chunked - 1:
                    wait_time = min((2 ** attempt) * self.retry_delay, 120) 
                    logging.warning(f"Streaming interrupted, retrying ({attempt + 1}/{self.max_retries_chunked}): {e}")
                    time.sleep(wait_time)
                    self._reset_session()
                    continue
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
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat request using the original interface and return a Dict.
        
        This method is retained for compatibility with data_preprocessing.
        """
        result = self._stream_request(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            **kwargs
        )
        
        # Build a response structure consistent with the non-streaming format.
        message_obj = {"role": "assistant", "content": result.get("content", "")}
        if result.get("reasoning_content"):
            message_obj["reasoning_content"] = result["reasoning_content"]
        
        return {
            "id": "streamed_response",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{"index": 0, "message": message_obj, "finish_reason": result.get("finish_reason", "stop")}],
            "usage": result.get("usage", {}),
            "_response_time": result.get("response_time", 0)
        }
    
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
        
        if self.seed is not None:
            payload["seed"] = self.seed
        if response_format:
            payload["response_format"] = response_format
        
        start_time = time.time()
        # Use max(configured, default) for read timeout to ensure enough headroom.
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
                        if delta.get("reasoning_content"):
                            full_reasoning += delta["reasoning_content"]
                        if choices[0].get("finish_reason"):
                            finish_reason = choices[0]["finish_reason"]
                except json.JSONDecodeError:
                    continue
        
        return {
            "content": full_content,
            "reasoning_content": full_reasoning,
            "finish_reason": finish_reason or "stop",
            "usage": usage or {},
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
        """Check whether the DeepSeek API is available."""
        try:
            # Use a simple hello-world test.
            test_model = "deepseek-chat" # Always use the chat model for connectivity checks to save cost/time.
            url = f"{self.base_url}{self.endpoint}"
            payload = {
                "model": test_model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            }
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return bool(result.get("choices", [{}])[0].get("message", {}).get("content"))
        except Exception as e:
            logging.warning(f"DeepSeek API unavailable: {e}")
            return False
    
    def analyze_with_retry(
        self,
        prompt: str,
        max_retries: Optional[int] = None,
        use_json_mode: bool = True
    ) -> Dict[str, Any]:
        """Retrying API call kept for data_preprocessing compatibility."""
        # Reuse chat_completion but add an extra retry layer for application-level errors.
        # Note: BaseClient does not define this method; it is kept for legacy compatibility.
        
        target_retries = max_retries if max_retries is not None else self.max_retries
        last_error = None
        
        for attempt in range(target_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                response_format = {'type': 'json_object'} if use_json_mode else None
                result = self.chat_completion(messages, response_format=response_format)
                
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                if use_json_mode and not content.strip():
                    raise ValueError("API returned empty content")
                
                return result
                
            except Exception as e:
                last_error = e
                logging.warning(f"Application-level API retry {attempt + 1}/{target_retries}: {e}")
                if attempt < target_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        
        raise RuntimeError(f"API call failed after {target_retries} retries: {last_error}")


# Backward-compatible alias.
DeepSeekLLMClient = DeepSeekClient
