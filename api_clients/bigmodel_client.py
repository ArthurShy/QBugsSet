#!/usr/bin/env python3
"""
Zhipu BigModel API client.

Supports GLM-series models and uses streaming responses to reduce timeout risk.
API docs: https://open.bigmodel.cn/dev/api
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
import requests

from .base_client import BaseClient, LLMResponse, LLMClientType, LLMModelConfig


# BigModel API configuration.
BIGMODEL_API_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
BIGMODEL_API_ENDPOINT = "/chat/completions"

# Default settings.
DEFAULT_MODEL = "glm-4-flash"
DEFAULT_CONNECT_TIMEOUT = 30
DEFAULT_READ_TIMEOUT = 300
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# Model configurations.
BIGMODEL_MODELS = {
    "glm-4-flash": {
        "config": LLMModelConfig(
            name="glm-4-flash",
            model_name="glm-4-flash",
            client_type=LLMClientType.BIGMODEL,
            api_key_env="BIGMODEL_API_KEY",
            batch_size=10,
            timeout=60,
            max_tokens=4096,
            description="Zhipu GLM-4-Flash model"
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
        },
        "context_window": 128000
    },
    "glm-4.7-flash": {
        "config": LLMModelConfig(
            name="glm-4.7-flash",
            model_name="glm-4-flash",  # Actual model name used by the API.
            client_type=LLMClientType.BIGMODEL,
            api_key_env="BIGMODEL_API_KEY",
            batch_size=10,
            timeout=60,
            max_tokens=4096,
            description="Zhipu GLM-4.7-Flash model"
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
        },
        "context_window": 128000
    },
}


class BigModelClient(BaseClient):
    """
    Zhipu BigModel API client.
    
    Supports GLM-series models and uses streaming responses to reduce timeout risk.
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
        
        # 2. Load the predefined configuration.
        model_def = BIGMODEL_MODELS.get(actual_model)
        if not model_def:
            logging.warning(f"Unknown model '{actual_model}', using default configuration")
            model_config = BIGMODEL_MODELS[DEFAULT_MODEL]["config"]
            retry_config = BIGMODEL_MODELS[DEFAULT_MODEL]["retry"]
            api_model_name = actual_model
        else:
            model_config = model_def["config"]
            retry_config = model_def["retry"]
            api_model_name = model_config.model_name
            
        # 3. Resolve final parameters.
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
        
        # 5. Set BigModel-specific attributes.
        self.api_key = api_key or os.getenv('BIGMODEL_API_KEY')
        if not self.api_key:
            raise ValueError("BigModel API key not provided. Please set the BIGMODEL_API_KEY environment variable")
            
        self.base_url = BIGMODEL_API_BASE_URL
        self.endpoint = BIGMODEL_API_ENDPOINT
        self.model = api_model_name  # Actual model name used by the API.
        self.display_name = actual_model  # Display name.
        self.max_output_tokens = final_max_tokens
        
        # 6. Configure retry behavior.
        self.max_retries = kwargs.get("max_retries", retry_config["max_retries"])
        self.retry_delay = kwargs.get("retry_delay", retry_config["retry_delay"])

        # 7. Initialize the session.
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        })
        self.headers = dict(self.session.headers)
        
        logging.info(f"  Streaming: enabled")
        logging.info(f"  Retry policy: max={self.max_retries}, delay={self.retry_delay}s")
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        response_format: Optional[Dict[str, str]] = None
    ) -> LLMResponse:
        """
        Send a chat request (BaseClient interface).
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = self._stream_request(messages, response_format=response_format)
                response_time = time.time() - start_time
                
                content = result.get("content", "")
                usage = result.get("usage", {})
                
                if not content:
                    logging.warning("Received empty response content")
                    logging.warning(f"  - finish_reason: {result.get('finish_reason', 'N/A')}")
                    logging.warning(f"  - usage: {usage}")
                
                token_usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                } if usage else None
                
                return LLMResponse(
                    content=content,
                    success=True,
                    token_usage=token_usage,
                    response_time=response_time
                )
                
            except requests.exceptions.ChunkedEncodingError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = min((2 ** attempt) * self.retry_delay, 120) 
                    logging.warning(f"Streaming interrupted, retrying ({attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(wait_time)
                    self._reset_session()
                    continue
                    
            except Exception as e:
                last_error = e
                logging.warning(f"Request failed, retrying ({attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    self._reset_session()
        
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
        
        final_temp = temperature if temperature is not None else self.temperature
        final_max_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": final_temp,
            "max_tokens": final_max_tokens,
            "stream": True,
            "enable_thinking": False,  # Explicitly disable thinking mode.
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        start_time = time.time()
        content_chunks = []
        usage = {}
        finish_reason = None
        response = None
        
        try:
            response = self.session.post(
                url,
                json=payload,
                stream=True,
                timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str.strip() == '[DONE]':
                        break
                        
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get('choices', [])
                        if choices:
                            delta = choices[0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                content_chunks.append(delta['content'])
                            
                            if choices[0].get('finish_reason'):
                                finish_reason = choices[0]['finish_reason']
                        
                        if 'usage' in chunk:
                            usage = chunk['usage']
                            
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if response is not None:
                try:
                    error_detail = response.json()
                    error_msg = f"{e}: {error_detail}"
                except:
                    pass
            raise Exception(error_msg)
        
        return {
            "content": "".join(content_chunks),
            "usage": usage,
            "finish_reason": finish_reason,
            "response_time": time.time() - start_time
        }
    
    def _reset_session(self):
        """Reset the HTTP session."""
        self.session.close()
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def is_available(self) -> bool:
        """Check whether the service is available."""
        try:
            test_messages = [{"role": "user", "content": "Hi"}]
            response = self.chat(test_messages)
            return response.success
        except Exception as e:
            logging.warning(f"BigModel service check failed: {e}")
            return False
