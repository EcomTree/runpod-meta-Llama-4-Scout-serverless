"""
Central configuration module for RunPod serverless deployment.
Handles all environment variables and configuration settings.
"""

import os
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    
    # Model identification
    model_id: str = os.getenv(
        "MODEL_ID", 
        "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )
    
    # Hugging Face authentication
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    
    # Device configuration
    device_map: str = os.getenv("DEVICE_MAP", "auto")
    
    # Memory optimization
    torch_dtype: str = os.getenv("TORCH_DTYPE", "bfloat16")  # auto, float16, bfloat16
    load_in_8bit: bool = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
    load_in_4bit: bool = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
    
    # Attention optimization
    use_flash_attention: bool = os.getenv("ENABLE_FLASH_ATTENTION", "true").lower() == "true"
    
    # Memory limits (in GB, optional)
    max_memory: Optional[dict] = None
    
    # Cache directory
    cache_dir: str = os.getenv("HF_HOME", "/root/.cache/huggingface")
    
    # Trust remote code
    trust_remote_code: bool = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"


@dataclass
class InferenceConfig:
    """Default configuration for text generation."""
    
    # Generation parameters
    max_new_tokens: int = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("DEFAULT_TOP_P", "0.9"))
    top_k: int = int(os.getenv("DEFAULT_TOP_K", "50"))
    repetition_penalty: float = float(os.getenv("DEFAULT_REPETITION_PENALTY", "1.1"))
    
    # Limits
    max_input_tokens: int = int(os.getenv("MAX_INPUT_TOKENS", "4096"))
    max_total_tokens: int = int(os.getenv("MAX_TOTAL_TOKENS", "8192"))
    
    # Character-to-token ratio estimate (approximate, varies by tokenizer and language)
    chars_per_token_estimate: int = 4
    
    # Timeout settings
    inference_timeout_seconds: int = int(os.getenv("INFERENCE_TIMEOUT", "120"))
    
    # Streaming
    enable_streaming: bool = os.getenv("ENABLE_STREAMING", "false").lower() == "true"


@dataclass
class ServerConfig:
    """Configuration for the FastAPI health check server."""
    
    host: str = os.getenv("HEALTH_CHECK_HOST", "0.0.0.0")
    port: int = int(os.getenv("HEALTH_CHECK_PORT", "8000"))
    
    # Health check settings
    model_warmup: bool = os.getenv("MODEL_WARMUP", "true").lower() == "true"
    warmup_prompt: str = "Hello, how are you?"


@dataclass
class LogConfig:
    """Logging configuration."""
    
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = os.getenv("LOG_FORMAT", "json")  # json or text
    
    # Performance logging
    log_metrics: bool = os.getenv("LOG_METRICS", "true").lower() == "true"
    log_requests: bool = os.getenv("LOG_REQUESTS", "true").lower() == "true"


@dataclass
class RunPodConfig:
    """RunPod-specific configuration."""
    
    api_key: Optional[str] = os.getenv("RUNPOD_AI_API_KEY")
    endpoint_id: Optional[str] = os.getenv("RUNPOD_ENDPOINT_ID")
    pod_id: Optional[str] = os.getenv("RUNPOD_POD_ID")
    
    # Worker configuration
    worker_id: Optional[str] = os.getenv("RUNPOD_WORKER_ID")


# Global configuration instances
model_config = ModelConfig()
inference_config = InferenceConfig()
server_config = ServerConfig()
log_config = LogConfig()
runpod_config = RunPodConfig()


def validate_config() -> Tuple[bool, List[str]]:
    """
    Validate configuration and return status with any error messages.
    
    Returns:
        Tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check required HF token
    if not model_config.hf_token:
        errors.append("HF_TOKEN environment variable is required for model access")
    
    # Check for conflicting quantization settings
    if model_config.load_in_8bit and model_config.load_in_4bit:
        errors.append("Cannot use both 8-bit and 4-bit quantization simultaneously")
    
    # Validate numeric ranges
    if not 0.0 <= inference_config.temperature <= 2.0:
        errors.append(f"Temperature must be between 0.0 and 2.0, got {inference_config.temperature}")
    
    if not 0.0 <= inference_config.top_p <= 1.0:
        errors.append(f"top_p must be between 0.0 and 1.0, got {inference_config.top_p}")
    
    if inference_config.top_k < 0:
        errors.append(f"top_k must be non-negative, got {inference_config.top_k}")
    
    if inference_config.max_new_tokens <= 0:
        errors.append(f"max_new_tokens must be positive, got {inference_config.max_new_tokens}")
    
    return len(errors) == 0, errors


def get_config_summary() -> dict:
    """
    Get a summary of current configuration (safe for logging).
    
    Returns:
        dict: Configuration summary without sensitive data
    """
    return {
        "model_id": model_config.model_id,
        "device_map": model_config.device_map,
        "torch_dtype": model_config.torch_dtype,
        "use_flash_attention": model_config.use_flash_attention,
        "load_in_8bit": model_config.load_in_8bit,
        "load_in_4bit": model_config.load_in_4bit,
        "max_new_tokens": inference_config.max_new_tokens,
        "temperature": inference_config.temperature,
        "log_level": log_config.level,
        "hf_token_configured": model_config.hf_token is not None,
        "runpod_endpoint_id": runpod_config.endpoint_id,
    }

