"""
Local LLM provider using transformers for inference.

This provider runs a model available on the local filesystem using the
`transformers` library. The path to the model directory is specified by the
`LOCAL_LLM_MODEL_PATH` environment variable or directly in the configuration
file via `api_endpoint_env`.
"""

import asyncio
import json
import re
import threading
from typing import Any, Dict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config.config import CONFIG
from llm.llm_provider import LLMProvider
from utils.logging_config_helper import get_configured_logger

logger = get_configured_logger("local_llm")


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


class LocalLLMProvider(LLMProvider):
    """Implementation of LLMProvider for a local model."""

    _init_lock = threading.Lock()
    _pipeline = None

    @classmethod
    def get_model_path(cls) -> str:
        provider_config = CONFIG.llm_endpoints.get("local")
        if provider_config and provider_config.endpoint:
            return provider_config.endpoint
        raise ConfigurationError(
            "LOCAL_LLM_MODEL_PATH is not configured in config_llm.yaml or .env"
        )

    @classmethod
    def get_client(cls):
        with cls._init_lock:
            if cls._pipeline is None:
                model_path = cls.get_model_path()
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                cls._pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device="cpu",
                )
        return cls._pipeline

    @classmethod
    def clean_response(cls, content: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip()
        match = re.search(r"(\{.*\})", cleaned, re.S)
        if not match:
            logger.error("Failed to parse JSON from content: %r", content)
            return {}
        return json.loads(match.group(1))

    async def get_completion(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        **kwargs,
    ) -> Dict[str, Any]:
        pipe = self.get_client()
        system_prompt = (
            f"Provide a valid JSON response matching this schema: {json.dumps(schema)}\n"
            f"{prompt}"
        )
        try:
            outputs = await asyncio.wait_for(
                asyncio.to_thread(
                    pipe,
                    system_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout,
            )
        except asyncio.TimeoutError:
            logger.error("Completion request timed out after %s seconds", timeout)
            return {}

        if not outputs:
            return {}
        content = outputs[0]["generated_text"]
        return self.clean_response(content)


provider = LocalLLMProvider()
