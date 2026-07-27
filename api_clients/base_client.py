#!/usr/bin/env python3
"""
Base classes for LLM clients.

Defines the shared client interface, response schema, and model config types.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


class LLMClientType(Enum):
    """LLM client types."""
    VLLM = "vllm"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ALICLOUD = "alicloud"
    BIGMODEL = "bigmodel"
    VERTEXAI = "vertexai"


@dataclass
class LLMModelConfig:
    """LLM model configuration."""
    name: str
    model_name: str
    client_type: LLMClientType
    base_url: str = ""
    api_key_env: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 120
    batch_size: int = 5
    reasoning_parser: bool = False
    thinking_budget: Optional[int] = None
    description: str = ""


@dataclass
class LLMResponse:
    """LLM response data structure."""
    content: str
    success: bool
    error: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    response_time: Optional[float] = None
    raw_response: Optional[Any] = None
    finish_reason: Optional[str] = None
    thinking_content: Optional[str] = None  # Chain-of-thought content for reasoning models.
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "success": self.success,
            "error": self.error,
            "token_usage": self.token_usage,
            "response_time": self.response_time,
            "finish_reason": self.finish_reason,
        }


@dataclass
class BatchCompletionRequest:
    """Batch completion request."""

    custom_id: str
    prompt: str
    system_prompt: Optional[str] = None
    response_format: Optional[Dict[str, str]] = None


class BaseClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_config = kwargs
        
        logging.info(f"Initializing client: {self.__class__.__name__}")
        logging.info(f"  Model: {model_name}")
        logging.info(f"  Temperature: {temperature}")
        logging.info(f"  Max output tokens: {max_tokens}")
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        response_format: Optional[Dict[str, str]] = None
    ) -> LLMResponse:
        """
        Send a chat request and return an LLMResponse.
        
        Args:
            messages: List of messages.
            response_format: Response format config (for example {"type": "json_object"}).
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the service is available."""
        pass
    
    def complete(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> LLMResponse:
        """
        Simplified completion interface.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt for system/user separation.
            response_format: Response format config (for example {"type": "json_object"}).
            
        Returns:
            An LLMResponse object.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, response_format=response_format)

    def supports_batch_completion(self) -> bool:
        """Whether server-side batch completion is supported."""
        return False

    def complete_batch(self, requests: List[BatchCompletionRequest]) -> Dict[str, LLMResponse]:
        """Run server-side batch completion."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support complete_batch()"
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
