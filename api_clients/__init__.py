"""
Unified LLM client module.

Client classes:
- BaseClient: abstract base class
- DeepSeekClient: DeepSeek API
- VLLMClient: local vLLM service
- OpenRouterClient: OpenRouter API

Data structures:
- LLMResponse: response data structure
- LLMClientType: client type enum
- LLMModelConfig: model configuration dataclass

Model configs:
- DEEPSEEK_MODELS: DeepSeek model configs
- VLLM_MODELS: local vLLM model configs
- OPENROUTER_MODELS: OpenRouter model configs
"""

from typing import Optional
from .base_client import (
    BaseClient,
    BatchCompletionRequest,
    LLMResponse,
    LLMClientType,
    LLMModelConfig,
)
from .deepseek_client import DeepSeekClient, DeepSeekLLMClient, DEEPSEEK_MODELS
from .vllm_client import VLLMClient, VLLM_MODELS
from .openrouter_client import OpenRouterClient, OPENROUTER_MODELS
from .alicloud_client import AliCloudClient, DashScopeClient, ALICLOUD_MODELS
from .bigmodel_client import BigModelClient, BIGMODEL_MODELS
from .vertexai_client import VertexAIClient, VERTEX_MODELS


# Merge all model configurations.
def get_all_models():
    """Return all available model configurations."""
    all_models = {}
    all_models.update(DEEPSEEK_MODELS)
    all_models.update(VLLM_MODELS)
    all_models.update(OPENROUTER_MODELS)
    all_models.update(ALICLOUD_MODELS)
    all_models.update(BIGMODEL_MODELS)
    all_models.update(VERTEX_MODELS)
    return all_models


LLM_MODELS = get_all_models()


# Mapping from platform name to client type.
PLATFORM_TO_CLIENT_TYPE = {
    "vllm": LLMClientType.VLLM,
    "deepseek": LLMClientType.DEEPSEEK,
    "alicloud": LLMClientType.ALICLOUD,
    "openrouter": LLMClientType.OPENROUTER,
    "bigmodel": LLMClientType.BIGMODEL,
    "vertexai": LLMClientType.VERTEXAI,
}


def get_llm_model_config(model_name: str, platform: Optional[str] = None) -> LLMModelConfig:
    """
    Get the LLM model configuration.
    
    Args:
        model_name: Model name.
        platform: Optional platform type (vllm, deepseek, alicloud, openrouter, bigmodel).
    """
    # If a platform is provided, search that platform's model list first.
    if platform:
        platform_models = {
            "deepseek": DEEPSEEK_MODELS,
            "vllm": VLLM_MODELS,
            "openrouter": OPENROUTER_MODELS,
            "alicloud": ALICLOUD_MODELS,
            "bigmodel": BIGMODEL_MODELS,
            "vertexai": VERTEX_MODELS,
        }
        models = platform_models.get(platform, {})
        if model_name in models:
            model_def = models[model_name]
            if isinstance(model_def, dict) and 'config' in model_def:
                return model_def['config']
            return model_def
        # Raise an error if the model cannot be found on the requested platform.
        available = ", ".join(models.keys())
        raise ValueError(f"Model {model_name} was not found on platform {platform}. Available: {available}")
    
    # If no platform is specified, search the merged model registry.
    if model_name in VERTEX_MODELS:
        return VERTEX_MODELS[model_name]["config"]

    if model_name not in LLM_MODELS:
        available = ", ".join(LLM_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    model_def = LLM_MODELS[model_name]
    if isinstance(model_def, dict) and 'config' in model_def:
        return model_def['config']
    return model_def


__all__ = [
    # Clients
    'BaseClient',
    'BatchCompletionRequest',
    'DeepSeekClient',
    'DeepSeekLLMClient',
    'VLLMClient',
    'OpenRouterClient',
    'AliCloudClient',
    'DashScopeClient',
    'BigModelClient',
    'VertexAIClient',
    # Data structures
    'LLMResponse',
    'LLMClientType',
    'LLMModelConfig',
    # Model configurations
    'LLM_MODELS',
    'DEEPSEEK_MODELS',
    'VLLM_MODELS',
    'OPENROUTER_MODELS',
    'ALICLOUD_MODELS',
    'BIGMODEL_MODELS',
    'VERTEX_MODELS',
    'get_llm_model_config',
]
