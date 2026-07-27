#!/usr/bin/env python3
"""
Vertex AI REST API client.

Supports two model families:
1. Google Gemini models
2. Anthropic Claude partner models

Supports two authentication modes:
1. Vertex AI Express Mode API key (Google Gemini only)
2. Standard Vertex AI bearer token
"""

import logging
import os
import shutil
import subprocess
import time
import json
import hashlib
import tempfile
import uuid
from textwrap import shorten
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .base_client import (
    BaseClient,
    BatchCompletionRequest,
    LLMClientType,
    LLMModelConfig,
    LLMResponse,
)


DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_BATCH_POLL_INTERVAL = 15
DEFAULT_BATCH_TIMEOUT = 60 * 60 * 6
DEFAULT_NETWORK_RETRIES = 5
DEFAULT_GEMINI_25_FLASH_THINKING_BUDGET = 0
TERMINAL_BATCH_STATES = {
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_FAILED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_SUCCEEDED",
}
SUCCESS_BATCH_STATES = {
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_SUCCEEDED",
}


VERTEX_MODELS = {
    "gemini-3.1-flash-lite-preview": {
        "config": LLMModelConfig(
            name="gemini-3.1-flash-lite-preview",
            model_name="gemini-3.1-flash-lite-preview",
            client_type=LLMClientType.VERTEXAI,
            api_key_env="VERTEX_AI_API_KEY",
            batch_size=10,
            timeout=60,
            max_tokens=8192,
            reasoning_parser=False,
            description="Google Gemini 3.1 Flash Lite Preview via Vertex AI",
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6,
        },
        "context_window": 1048576,
        "publisher": "google",
        "endpoint": "generateContent",
    },
    "gemini-3.1-flash-lite": {
        "config": LLMModelConfig(
            name="gemini-3.1-flash-lite",
            model_name="gemini-3.1-flash-lite-preview",
            client_type=LLMClientType.VERTEXAI,
            api_key_env="VERTEX_AI_API_KEY",
            batch_size=10,
            timeout=60,
            max_tokens=8192,
            reasoning_parser=False,
            description="Google Gemini 3.1 Flash Lite via Vertex AI",
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6,
        },
        "context_window": 1048576,
        "publisher": "google",
        "endpoint": "generateContent",
    },
    "gemini-2.5-flash": {
        "config": LLMModelConfig(
            name="gemini-2.5-flash",
            model_name="gemini-2.5-flash",
            client_type=LLMClientType.VERTEXAI,
            api_key_env="VERTEX_AI_API_KEY",
            batch_size=10,
            timeout=60,
            max_tokens=8192,
            reasoning_parser=False,
            description="Google Gemini 2.5 Flash via Vertex AI",
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6,
        },
        "context_window": 1048576,
        "publisher": "google",
        "endpoint": "generateContent",
    },
    "gemini-3-flash-preview": {
        "config": LLMModelConfig(
            name="gemini-3-flash-preview",
            model_name="gemini-3-flash-preview",
            client_type=LLMClientType.VERTEXAI,
            api_key_env="VERTEX_AI_API_KEY",
            batch_size=10,
            timeout=60,
            max_tokens=8192,
            reasoning_parser=False,
            description="Google Gemini 3 Flash Preview via Vertex AI",
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6,
        },
        "context_window": 1048576,
        "publisher": "google",
        "endpoint": "generateContent",
    },
    "gemini-3.1-pro-preview": {
        "config": LLMModelConfig(
            name="gemini-3.1-pro-preview",
            model_name="gemini-3.1-pro-preview",
            client_type=LLMClientType.VERTEXAI,
            api_key_env="VERTEX_AI_API_KEY",
            batch_size=4,
            timeout=120,
            max_tokens=65536,
            reasoning_parser=False,
            description="Google Gemini 3.1 Pro Preview via Vertex AI (thinking low)",
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6,
        },
        "context_window": 1048576,
        "publisher": "google",
        "endpoint": "generateContent",
    },
    "claude-sonnet-4-6": {
        "config": LLMModelConfig(
            name="claude-sonnet-4-6",
            model_name="claude-sonnet-4-6",
            client_type=LLMClientType.VERTEXAI,
            api_key_env="VERTEX_AI_API_KEY",
            batch_size=4,
            timeout=120,
            max_tokens=8192,
            reasoning_parser=False,
            description="Anthropic Claude Sonnet 4.6 via Vertex AI (thinking disabled)",
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6,
        },
        "context_window": 200000,
        "publisher": "anthropic",
        "endpoint": "rawPredict",
        "anthropic_version": "vertex-2023-10-16",
    },
    "claude-haiku-4-5": {
        "config": LLMModelConfig(
            name="claude-haiku-4-5",
            model_name="claude-haiku-4-5@20251001",
            client_type=LLMClientType.VERTEXAI,
            api_key_env="VERTEX_AI_API_KEY",
            batch_size=8,
            timeout=120,
            max_tokens=8192,
            reasoning_parser=False,
            description="Anthropic Claude Haiku 4.5 via Vertex AI (thinking disabled)",
        ),
        "retry": {
            "max_retries": 5,
            "retry_delay": 2,
            "max_retries_chunked": 6,
        },
        "context_window": 200000,
        "publisher": "anthropic",
        "endpoint": "rawPredict",
        "anthropic_version": "vertex-2023-10-16",
    },
}


class VertexAIClient(BaseClient):
    """Vertex AI REST client."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        thinking_level: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        batch_gcs_root: Optional[str] = None,
        batch_poll_interval: Optional[int] = None,
        batch_timeout: Optional[int] = None,
        batch_job_name: Optional[str] = None,
        **kwargs: Any,
    ):
        requested_model = model or model_name or DEFAULT_MODEL
        model_def = VERTEX_MODELS.get(requested_model)
        if not model_def and model_name:
            for candidate_name, candidate_def in VERTEX_MODELS.items():
                if candidate_def["config"].model_name == model_name:
                    requested_model = candidate_name
                    model_def = candidate_def
                    break
        if not model_def:
            raise ValueError(f"Unknown Vertex AI model: {requested_model}")

        model_config = model_def["config"]
        retry_config = model_def["retry"]
        self.publisher = model_def.get("publisher", "google")
        self.endpoint = model_def.get("endpoint", "generateContent")
        self.anthropic_version = model_def.get("anthropic_version", "vertex-2023-10-16")
        final_max_tokens = max_tokens or max_output_tokens or model_config.max_tokens
        final_timeout = timeout or model_config.timeout

        super().__init__(
            model_name=requested_model,
            temperature=temperature,
            max_tokens=final_max_tokens,
            timeout=final_timeout,
            **kwargs,
        )

        self.project = (
            project
            or os.getenv("VERTEX_AI_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCLOUD_PROJECT")
            or os.getenv("PROJECT_ID")
            or os.getenv("GoogleProjectID")
        )
        self.location = (
            location
            or os.getenv("VERTEX_AI_LOCATION")
            or os.getenv("GOOGLE_CLOUD_LOCATION")
            or os.getenv("LOCATION")
            or DEFAULT_LOCATION
        )
        self.api_model_name = model_config.model_name
        self.max_output_tokens = final_max_tokens
        self.thinking_level = thinking_level or self._default_thinking_level(requested_model)
        self.thinking_budget = self._resolve_thinking_budget(requested_model, thinking_budget)
        self.max_retries = kwargs.get("max_retries", retry_config["max_retries"])
        self.retry_delay = kwargs.get("retry_delay", retry_config["retry_delay"])
        self.batch_gcs_root = (
            batch_gcs_root
            or os.getenv("VERTEX_AI_BATCH_GCS_ROOT")
            or os.getenv("GOOGLE_CLOUD_BATCH_GCS_ROOT")
        )
        self.batch_poll_interval = int(
            batch_poll_interval
            or os.getenv("VERTEX_AI_BATCH_POLL_INTERVAL")
            or DEFAULT_BATCH_POLL_INTERVAL
        )
        self.batch_timeout = int(
            batch_timeout
            or os.getenv("VERTEX_AI_BATCH_TIMEOUT")
            or DEFAULT_BATCH_TIMEOUT
        )
        self.batch_job_name = (
            batch_job_name
            or os.getenv("VERTEX_AI_BATCH_JOB_NAME")
            or None
        )
        self.network_retries = int(
            kwargs.get("network_retries")
            or os.getenv("VERTEX_AI_NETWORK_RETRIES")
            or DEFAULT_NETWORK_RETRIES
        )
        self.api_key = (
            api_key
            or os.getenv("VERTEX_AI_API_KEY")
            or os.getenv("API_KEY")
            or os.getenv("GoogleVertaxAI_API_KEY")
        )
        self.use_express_mode = bool(self.api_key) and self.publisher == "google"
        self.access_token = os.getenv("VERTEX_AI_ACCESS_TOKEN") or os.getenv("GOOGLE_CLOUD_ACCESS_TOKEN")
        self._token_loaded_from_gcloud = False
        self.last_error_summary: Optional[str] = None
        if not self.project:
            raise ValueError(
                "Vertex AI project not provided. Please set VERTEX_AI_PROJECT or GOOGLE_CLOUD_PROJECT."
            )
        api_host = "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"
        self.base_url = (
            f"https://{api_host}/v1/projects/"
            f"{self.project}/locations/{self.location}"
        )

        logging.info("  Vertex AI: enabled")
        logging.info("  Auth mode: %s", "api-key" if self.use_express_mode else "bearer-token")
        logging.info("  Project: %s", self.project)
        logging.info("  Location: %s", self.location)
        logging.info("  Publisher: %s", self.publisher)
        logging.info("  Thinking level: %s", self.thinking_level or "provider-default")
        logging.info(
            "  Thinking budget: %s",
            self.thinking_budget if self.thinking_budget is not None else "provider-default",
        )
        logging.info("  Retry policy: max=%s, delay=%ss", self.max_retries, self.retry_delay)
        logging.info("  Network retries: %s", self.network_retries)
        if self.batch_gcs_root:
            logging.info("  Batch GCS Root: %s", self.batch_gcs_root)
        if self.batch_job_name:
            logging.info("  Resume Batch Job: %s", self.batch_job_name)

    def _default_thinking_level(self, model_name: str) -> Optional[str]:
        if model_name.startswith("gemini-3"):
            return "low"
        return None

    def _resolve_thinking_budget(
        self,
        model_name: str,
        thinking_budget: Optional[int],
    ) -> Optional[int]:
        if thinking_budget is not None:
            return thinking_budget

        env_budget = os.getenv("VERTEX_AI_THINKING_BUDGET")
        if env_budget is not None and env_budget != "":
            return int(env_budget)

        # Vertex AI docs note that Gemini 2.5 Flash defaults to automatic
        # thinking up to 8,192 tokens; setting budget to 0 disables thinking.
        # For strict JSON classification output, disabling thinking avoids
        # consuming the entire output budget on thoughts.
        if model_name == "gemini-2.5-flash":
            return DEFAULT_GEMINI_25_FLASH_THINKING_BUDGET

        return None

    def _get_access_token(self) -> str:
        if self.access_token:
            return self.access_token

        if not shutil.which("gcloud"):
            raise ValueError(
                "Vertex AI access token not provided. Please set VERTEX_AI_ACCESS_TOKEN "
                "or install and log in to gcloud."
            )

        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        token = result.stdout.strip()
        if not token:
            raise ValueError("gcloud did not return a valid access token")
        self.access_token = token
        self._token_loaded_from_gcloud = True
        return token

    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        system_prompt = None
        contents: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        return system_prompt, contents

    def _convert_messages_anthropic(
        self,
        messages: List[Dict[str, str]],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        system_prompt = None
        converted: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role in {"user", "assistant"}:
                converted.append({"role": role, "content": content})

        return system_prompt, converted

    def _build_request_payload(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if self.publisher == "anthropic":
            system_prompt, anthropic_messages = self._convert_messages_anthropic(messages)
            payload: Dict[str, Any] = {
                "anthropic_version": self.anthropic_version,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if system_prompt:
                payload["system"] = system_prompt
            return payload

        system_prompt, contents = self._convert_messages(messages)
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if self.thinking_budget is not None:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": self.thinking_budget,
            }
        elif self.thinking_level:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": self.thinking_level,
            }
        if response_format and response_format.get("type") == "json_object":
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    def _extract_text_and_thinking(self, response_json: Dict[str, Any]) -> tuple[str, Optional[str]]:
        if self.publisher == "anthropic":
            blocks = response_json.get("content") or []
            texts: List[str] = []
            thoughts: List[str] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                text = block.get("text", "")
                if not text:
                    continue
                if block.get("type") == "thinking":
                    thoughts.append(text)
                else:
                    texts.append(text)
            thinking_content = "".join(thoughts).strip() or None
            return "".join(texts).strip(), thinking_content

        candidates = response_json.get("candidates") or []
        if not candidates:
            return "", None
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts: List[str] = []
        thoughts: List[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text", "")
            if not text:
                continue
            if part.get("thought") is True:
                thoughts.append(text)
            else:
                texts.append(text)
        thinking_content = "".join(thoughts).strip() or None
        return "".join(texts).strip(), thinking_content

    def _extract_finish_reason(self, response_json: Dict[str, Any]) -> Optional[str]:
        if self.publisher == "anthropic":
            return response_json.get("stop_reason")
        candidates = response_json.get("candidates") or []
        if not candidates:
            return None
        return candidates[0].get("finishReason")

    def _unwrap_batch_response_json(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        current: Any = response_json
        visited = set()

        while isinstance(current, dict):
            marker = id(current)
            if marker in visited:
                break
            visited.add(marker)

            if current.get("candidates"):
                return current
            if isinstance(current.get("content"), list):
                return current

            if isinstance(current.get("body"), dict):
                current = current["body"]
                continue

            if isinstance(current.get("prediction"), dict):
                current = current["prediction"]
                continue

            predictions = current.get("predictions")
            if isinstance(predictions, list) and predictions and isinstance(predictions[0], dict):
                current = predictions[0]
                continue

            if isinstance(current.get("generateContentResponse"), dict):
                current = current["generateContentResponse"]
                continue

            if isinstance(current.get("response"), dict):
                current = current["response"]
                continue

            break

        return current if isinstance(current, dict) else response_json

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if not self.use_express_mode:
            headers["Authorization"] = f"Bearer {self._get_access_token()}"
        return headers

    def _build_url(self) -> str:
        model_resource = self._build_batch_model_resource()
        if self.use_express_mode:
            return f"{self.base_url}/{model_resource}:{self.endpoint}?key={self.api_key}"
        return f"{self.base_url}/{model_resource}:{self.endpoint}"

    def _build_availability_url(self) -> str:
        return self._build_url()

    def _summarize_http_error(self, response: requests.Response) -> str:
        status = response.status_code
        summary = f"HTTP {status}"
        body = ""
        try:
            payload = response.json()
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                code = error.get("code")
                message = error.get("message")
                status_text = error.get("status")
                detail_parts = [
                    str(part)
                    for part in [code, status_text, message]
                    if part not in (None, "")
                ]
                if detail_parts:
                    summary = " | ".join(detail_parts)
                    return shorten(summary, width=400, placeholder="...")
            body = json.dumps(payload, ensure_ascii=False)
        except ValueError:
            body = response.text

        if body:
            summary = f"{summary} | {body}"
        return shorten(summary, width=400, placeholder="...")

    def supports_batch_completion(self) -> bool:
        return bool(self.batch_gcs_root)

    def _build_batch_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }

    def _batch_jobs_url(self) -> str:
        api_host = "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"
        return f"https://{api_host}/v1/projects/{self.project}/locations/{self.location}/batchPredictionJobs"

    def _batch_job_url(self, job_name: str) -> str:
        api_host = "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"
        return f"https://{api_host}/v1/{job_name}"

    def _build_batch_display_name(self) -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_token = self.model_name.replace(".", "-").replace("/", "-")
        return f"quantum-{model_token}-batch-{timestamp}"

    def _build_batch_model_resource(self) -> str:
        if self.api_model_name.startswith("publishers/"):
            return self.api_model_name
        return f"publishers/{self.publisher}/models/{self.api_model_name}"

    def _normalize_gcs_uri(self, uri: str) -> str:
        return uri.rstrip("/")

    def _build_batch_input_uri(self) -> str:
        batch_id = uuid.uuid4().hex
        root = self._normalize_gcs_uri(self.batch_gcs_root or "")
        return f"{root}/inputs/{self.model_name}/{batch_id}.jsonl"

    def _build_batch_output_prefix(self) -> str:
        batch_id = uuid.uuid4().hex
        root = self._normalize_gcs_uri(self.batch_gcs_root or "")
        return f"{root}/outputs/{self.model_name}/{batch_id}"

    def _storage_command_prefix(self) -> List[str]:
        if shutil.which("gcloud"):
            return ["gcloud", "storage"]
        if shutil.which("gsutil"):
            return ["gsutil"]
        raise ValueError("Vertex AI batch requires the gcloud storage or gsutil CLI")

    def _run_storage_command(self, args: List[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
        prefix = self._storage_command_prefix()
        result = subprocess.run(
            prefix + args,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result

    def _upload_to_gcs(self, local_path: Path, gcs_uri: str) -> None:
        logging.info("Uploading batch input file to %s", gcs_uri)
        self._run_storage_command(["cp", str(local_path), gcs_uri], timeout=600)

    def _list_gcs_jsonl_files(self, gcs_prefix: str) -> List[str]:
        prefix = self._storage_command_prefix()
        if prefix == ["gcloud", "storage"]:
            result = self._run_storage_command(["ls", "--recursive", gcs_prefix], timeout=300)
        else:
            result = self._run_storage_command(["ls", "-r", gcs_prefix], timeout=300)
        files = [line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".jsonl")]
        return files

    def _download_from_gcs(self, gcs_uri: str, local_dir: Path) -> Path:
        destination = local_dir / Path(gcs_uri).name
        self._run_storage_command(["cp", gcs_uri, str(destination)], timeout=600)
        return destination

    def _request_with_bearer(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(self.network_retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self._build_batch_headers(),
                    json=json_body,
                    timeout=timeout or self.timeout,
                )
                if response.status_code == 401 and self._token_loaded_from_gcloud:
                    self.access_token = None
                    self._token_loaded_from_gcloud = False
                    response = requests.request(
                        method=method,
                        url=url,
                        headers=self._build_batch_headers(),
                        json=json_body,
                        timeout=timeout or self.timeout,
                    )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt >= self.network_retries - 1:
                    break
                backoff = min(30, 2 ** attempt)
                logging.warning(
                    "Vertex batch network request failed, retrying (%s/%s): %s",
                    attempt + 1,
                    self.network_retries,
                    exc,
                )
                time.sleep(backoff)

        if last_error:
            raise last_error

        response = requests.request(
            method=method,
            url=url,
            headers=self._build_batch_headers(),
            json=json_body,
            timeout=timeout or self.timeout,
        )
        if response.status_code == 401 and self._token_loaded_from_gcloud:
            self.access_token = None
            self._token_loaded_from_gcloud = False
            response = requests.request(
                method=method,
                url=url,
                headers=self._build_batch_headers(),
                json=json_body,
                timeout=timeout or self.timeout,
            )
        response.raise_for_status()
        return response

    def _make_request_hash(self, payload: Dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _make_prompt_signature(self, system_prompt: Optional[str], prompt: str) -> str:
        normalized = json.dumps(
            {
                "system_prompt": system_prompt or "",
                "prompt": prompt,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _build_batch_label_id(self, index: int) -> str:
        return f"q{index}"

    def _write_batch_requests(
        self,
        requests_list: List[BatchCompletionRequest],
        input_path: Path,
    ) -> Dict[str, Dict[str, List[str]]]:
        request_hash_to_ids: Dict[str, List[str]] = {}
        prompt_signature_to_ids: Dict[str, List[str]] = {}
        label_to_ids: Dict[str, List[str]] = {}
        with input_path.open("w", encoding="utf-8") as handle:
            for index, item in enumerate(requests_list):
                label_id = self._build_batch_label_id(index)
                payload = self._build_batch_request_payload(item, label_id=label_id)
                request_hash = self._make_request_hash(payload)
                request_hash_to_ids.setdefault(request_hash, []).append(item.custom_id)
                prompt_signature = self._make_prompt_signature(item.system_prompt, item.prompt)
                prompt_signature_to_ids.setdefault(prompt_signature, []).append(item.custom_id)
                label_to_ids.setdefault(label_id, []).append(item.custom_id)
                handle.write(json.dumps({"request": payload}, ensure_ascii=False))
                handle.write("\n")
        return {
            "request_hash_to_ids": request_hash_to_ids,
            "prompt_signature_to_ids": prompt_signature_to_ids,
            "label_to_ids": label_to_ids,
        }

    def _build_batch_request_payload(
        self,
        item: BatchCompletionRequest,
        *,
        label_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._build_request_payload(
            messages=[
                *([{"role": "system", "content": item.system_prompt}] if item.system_prompt else []),
                {"role": "user", "content": item.prompt},
            ],
            response_format=item.response_format,
        )
        if label_id and self.publisher == "google":
            payload["labels"] = {"id": label_id}
        return payload

    def _build_request_hash_mapping(
        self,
        requests_list: List[BatchCompletionRequest],
    ) -> Dict[str, Dict[str, List[str]]]:
        request_hash_to_ids: Dict[str, List[str]] = {}
        prompt_signature_to_ids: Dict[str, List[str]] = {}
        label_to_ids: Dict[str, List[str]] = {}
        for index, item in enumerate(requests_list):
            label_id = self._build_batch_label_id(index)
            payload = self._build_batch_request_payload(item, label_id=label_id)
            request_hash = self._make_request_hash(payload)
            request_hash_to_ids.setdefault(request_hash, []).append(item.custom_id)
            prompt_signature = self._make_prompt_signature(item.system_prompt, item.prompt)
            prompt_signature_to_ids.setdefault(prompt_signature, []).append(item.custom_id)
            label_to_ids.setdefault(label_id, []).append(item.custom_id)
        return {
            "request_hash_to_ids": request_hash_to_ids,
            "prompt_signature_to_ids": prompt_signature_to_ids,
            "label_to_ids": label_to_ids,
        }

    def _extract_output_prompt_signature(self, request_payload: Dict[str, Any]) -> Optional[str]:
        if self.publisher == "anthropic":
            prompt_parts: List[str] = []
            for message in request_payload.get("messages") or []:
                if not isinstance(message, dict):
                    continue
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    prompt_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("text"):
                            prompt_parts.append(block["text"])
            if not prompt_parts:
                return None
            return self._make_prompt_signature(request_payload.get("system", ""), "".join(prompt_parts))

        contents = request_payload.get("contents") or []
        prompt_parts: List[str] = []
        for content in contents:
            if not isinstance(content, dict):
                continue
            for part in content.get("parts") or []:
                if isinstance(part, dict) and part.get("text"):
                    prompt_parts.append(part["text"])
        system_instruction = request_payload.get("systemInstruction") or {}
        system_parts = []
        for part in system_instruction.get("parts") or []:
            if isinstance(part, dict) and part.get("text"):
                system_parts.append(part["text"])
        if not prompt_parts:
            return None
        return self._make_prompt_signature("".join(system_parts), "".join(prompt_parts))

    def _pop_custom_id_for_output(
        self,
        payload: Dict[str, Any],
        mappings: Dict[str, Dict[str, List[str]]],
    ) -> Optional[str]:
        request_payload = payload.get("request") or {}

        labels = request_payload.get("labels") or {}
        label_id = labels.get("id")
        if label_id:
            custom_ids = mappings["label_to_ids"].get(label_id)
            if custom_ids:
                return custom_ids.pop(0)

        prompt_signature = self._extract_output_prompt_signature(request_payload)
        if prompt_signature:
            custom_ids = mappings["prompt_signature_to_ids"].get(prompt_signature)
            if custom_ids:
                return custom_ids.pop(0)

        request_hash = self._make_request_hash(request_payload)
        custom_ids = mappings["request_hash_to_ids"].get(request_hash)
        if custom_ids:
            return custom_ids.pop(0)

        return None

    def _create_batch_job(self, input_uri: str, output_prefix: str) -> Dict[str, Any]:
        body = {
            "displayName": self._build_batch_display_name(),
            "model": self._build_batch_model_resource(),
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {
                    "uris": [input_uri],
                },
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {
                    "outputUriPrefix": output_prefix,
                },
            },
        }
        response = self._request_with_bearer(
            "POST",
            self._batch_jobs_url(),
            json_body=body,
            timeout=self.timeout,
        )
        return response.json()

    def _wait_for_batch_job(self, job_name: str) -> Dict[str, Any]:
        start_time = time.time()
        while True:
            response = self._request_with_bearer(
                "GET",
                self._batch_job_url(job_name),
                timeout=min(self.timeout, 60),
            )
            job = response.json()
            state = job.get("state")
            if state in TERMINAL_BATCH_STATES:
                if state not in SUCCESS_BATCH_STATES:
                    error = job.get("error") or {}
                    message = error.get("message") or json.dumps(error, ensure_ascii=False)
                    raise RuntimeError(f"Vertex batch job failed: {state} - {message}")
                return job

            if time.time() - start_time > self.batch_timeout:
                raise TimeoutError(f"Vertex batch job timed out (>{self.batch_timeout}s): {job_name}")

            elapsed = int(time.time() - start_time)
            logging.info("Waiting for Vertex batch completion: %s (%s, elapsed %ss)", job_name, state or "UNKNOWN", elapsed)
            time.sleep(self.batch_poll_interval)

    def _parse_batch_response_line(self, line: Dict[str, Any]) -> LLMResponse:
        status = line.get("status") or {}
        status_code = status.get("code", 0) if isinstance(status, dict) else 0
        if status_code:
            message = status.get("message") or json.dumps(status, ensure_ascii=False)
            return LLMResponse(
                content="",
                success=False,
                error=f"Batch item failed: {message}",
                raw_response=line,
            )

        response_json = line.get("response")
        if not isinstance(response_json, dict):
            return LLMResponse(
                content="",
                success=False,
                error="Batch item missing response body",
                raw_response=line,
            )

        response_json = self._unwrap_batch_response_json(response_json)

        content, thinking_content = self._extract_text_and_thinking(response_json)
        finish_reason = self._extract_finish_reason(response_json)
        token_usage = None
        if self.publisher == "anthropic":
            usage = response_json.get("usage") or {}
            if usage:
                prompt_tokens = usage.get("input_tokens", 0) or 0
                completion_tokens = usage.get("output_tokens", 0) or 0
                token_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
        else:
            usage_metadata = response_json.get("usageMetadata") or {}
            if usage_metadata:
                token_usage = {
                    "prompt_tokens": usage_metadata.get("promptTokenCount", 0) or 0,
                    "completion_tokens": usage_metadata.get("candidatesTokenCount", 0) or 0,
                    "total_tokens": usage_metadata.get("totalTokenCount", 0) or 0,
                }
                if usage_metadata.get("thoughtsTokenCount") is not None:
                    token_usage["thoughts_tokens"] = usage_metadata.get("thoughtsTokenCount", 0) or 0
                if usage_metadata.get("cachedContentTokenCount") is not None:
                    token_usage["cached_tokens"] = usage_metadata.get("cachedContentTokenCount", 0) or 0

        return LLMResponse(
            content=content,
            success=True,
            token_usage=token_usage,
            response_time=None,
            raw_response=response_json,
            finish_reason=finish_reason,
            thinking_content=thinking_content,
        )

    def complete_batch(self, requests_list: List[BatchCompletionRequest]) -> Dict[str, LLMResponse]:
        if not requests_list:
            return {}
        if not self.batch_gcs_root:
            raise ValueError("Vertex batch output directory is not configured. Set VERTEX_AI_BATCH_GCS_ROOT or pass batch_gcs_root")
        if not self.batch_gcs_root.startswith("gs://"):
            raise ValueError(f"Vertex batch GCS root must start with gs://: {self.batch_gcs_root}")

        input_uri = self._build_batch_input_uri()
        output_prefix = self._build_batch_output_prefix()
        results: Dict[str, LLMResponse] = {}

        with tempfile.TemporaryDirectory(prefix="vertex-batch-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            job_name = self.batch_job_name
            if job_name:
                request_mappings = self._build_request_hash_mapping(requests_list)
                logging.info("Continuing to wait for existing Vertex batch: %s", job_name)
            else:
                input_path = tmp_path / "requests.jsonl"
                request_mappings = self._write_batch_requests(requests_list, input_path)
                self._upload_to_gcs(input_path, input_uri)

                job = self._create_batch_job(input_uri, output_prefix)
                job_name = job.get("name")
                if not job_name:
                    raise RuntimeError(f"Vertex batch creation failed; no job name returned: {job}")
                logging.info("Vertex batch submitted: %s", job_name)

            completed_job = self._wait_for_batch_job(job_name)
            output_dir = (
                (completed_job.get("outputInfo") or {}).get("gcsOutputDirectory")
                or output_prefix
            )
            logging.info("Vertex batch completed, output directory: %s", output_dir)

            for gcs_file in self._list_gcs_jsonl_files(output_dir):
                local_file = self._download_from_gcs(gcs_file, tmp_path)
                with local_file.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        payload = json.loads(raw_line)
                        custom_id = self._pop_custom_id_for_output(payload, request_mappings)
                        if not custom_id:
                            logging.warning("Received unmatched batch output item: %s", gcs_file)
                            continue
                        results[custom_id] = self._parse_batch_response_line(payload)

        for item in requests_list:
            if item.custom_id not in results:
                results[item.custom_id] = LLMResponse(
                    content="",
                    success=False,
                    error="Batch output missing for request",
                )

        return results

    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        start_time = time.time()
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                payload = self._build_request_payload(messages, response_format)
                response = requests.post(
                    self._build_url(),
                    headers=self._build_headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                response_time = time.time() - start_time
                response.raise_for_status()
                response_json = response.json()

                content, thinking_content = self._extract_text_and_thinking(response_json)
                finish_reason = self._extract_finish_reason(response_json)

                token_usage = None
                if self.publisher == "anthropic":
                    usage = response_json.get("usage") or {}
                    if usage:
                        prompt_tokens = usage.get("input_tokens", 0) or 0
                        completion_tokens = usage.get("output_tokens", 0) or 0
                        token_usage = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        }
                else:
                    usage_metadata = response_json.get("usageMetadata") or {}
                    if usage_metadata:
                        token_usage = {
                            "prompt_tokens": usage_metadata.get("promptTokenCount", 0) or 0,
                            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0) or 0,
                            "total_tokens": usage_metadata.get("totalTokenCount", 0) or 0,
                        }
                        if usage_metadata.get("thoughtsTokenCount") is not None:
                            token_usage["thoughts_tokens"] = usage_metadata.get("thoughtsTokenCount", 0) or 0
                        if usage_metadata.get("cachedContentTokenCount") is not None:
                            token_usage["cached_tokens"] = usage_metadata.get("cachedContentTokenCount", 0) or 0

                return LLMResponse(
                    content=content,
                    success=True,
                    token_usage=token_usage,
                    response_time=response_time,
                    raw_response=response_json,
                    finish_reason=finish_reason,
                    thinking_content=thinking_content,
                )
            except Exception as exc:
                last_error = exc
                if self._token_loaded_from_gcloud:
                    self.access_token = None
                    self._token_loaded_from_gcloud = False
                logging.warning("Vertex AI request failed, retrying (%s/%s): %s", attempt + 1, self.max_retries, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

        return LLMResponse(
            content="",
            success=False,
            error=f"Request failed: {last_error}",
            response_time=time.time() - start_time,
        )

    def is_available(self) -> bool:
        try:
            response = requests.post(
                self._build_url(),
                headers=self._build_headers(),
                json=self._build_request_payload(
                    messages=[{"role": "user", "content": "ping"}],
                ),
                timeout=min(self.timeout, 10),
            )
            self.last_error_summary = None
            if response.status_code >= 400:
                self.last_error_summary = self._summarize_http_error(response)
            response.raise_for_status()
            return True
        except Exception as exc:
            if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                self.last_error_summary = self._summarize_http_error(exc.response)
                logging.warning("Vertex AI API unavailable: %s", self.last_error_summary)
            else:
                self.last_error_summary = str(exc)
                logging.warning("Vertex AI API unavailable: %s", exc)
            return False
