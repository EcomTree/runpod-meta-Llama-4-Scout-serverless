"""
Utility functions, custom exceptions, and logging configuration.
"""

import logging
import json
import sys
import time
import traceback
import uuid
from typing import Any, Dict, Optional
from datetime import datetime
from functools import wraps

from src.config import log_config, inference_config


# Custom Exceptions
class ModelLoadError(Exception):
    """Raised when model fails to load."""
    pass


class InferenceError(Exception):
    """Raised when inference fails."""
    pass


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class OperationTimeoutError(Exception):
    """Raised when operation times out."""
    pass


class GPUMemoryError(Exception):
    """Raised when GPU runs out of memory."""
    pass


# Logging Configuration
class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        # Add extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
        
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Custom text formatter for readable logging."""
    
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logging() -> logging.Logger:
    """
    Configure logging based on configuration settings.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("runpod_llama4")
    logger.setLevel(getattr(logging, log_config.level.upper(), logging.INFO))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter based on config
    if log_config.format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())
    
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Global logger instance
logger = setup_logging()


# Helper Functions
def generate_request_id() -> str:
    """
    Generate a unique request ID for tracking.
    Uses UUID4 for guaranteed uniqueness across distributed systems.
    
    Returns:
        str: Unique request identifier
    """
    return f"req_{uuid.uuid4().hex[:16]}"


def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input text.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length (defaults to config-based value)
        
    Returns:
        str: Sanitized text
        
    Raises:
        ValidationError: If input is invalid
    """
    if not isinstance(text, str):
        raise ValidationError(f"Input must be string, got {type(text).__name__}")
    
    if len(text) == 0:
        raise ValidationError("Input text cannot be empty")
    
    # Use config-based max length if not provided
    if max_length is None:
        max_length = inference_config.max_input_tokens * inference_config.chars_per_token_estimate
    
    if len(text) > max_length:
        raise ValidationError(f"Input text exceeds maximum length of {max_length} characters")
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Validate that text is not empty after sanitization
    if len(text) == 0 or not text.strip():
        raise ValidationError("Input text cannot be empty after sanitization")
    
    return text


def format_error_response(
    error: Exception,
    request_id: Optional[str] = None,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Format error as standardized response.
    
    Args:
        error: Exception that occurred
        request_id: Optional request identifier
        include_traceback: Whether to include full traceback
        
    Returns:
        dict: Formatted error response
    """
    error_response = {
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
        }
    }
    
    if request_id:
        error_response["request_id"] = request_id
    
    if include_traceback:
        error_response["error"]["traceback"] = traceback.format_exc()
    
    return error_response


def format_success_response(
    generated_text: str,
    metrics: Dict[str, Any],
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format successful inference result.
    
    Args:
        generated_text: Generated output text
        metrics: Performance metrics
        request_id: Optional request identifier
        
    Returns:
        dict: Formatted success response
    """
    response = {
        "output": {
            "generated_text": generated_text,
            **metrics
        }
    }
    
    if request_id:
        response["request_id"] = request_id
    
    return response


def log_metrics(logger_instance: logging.Logger, metrics: Dict[str, Any], request_id: Optional[str] = None):
    """
    Log performance metrics.
    
    Args:
        logger_instance: Logger to use
        metrics: Metrics dictionary
        request_id: Optional request identifier
    """
    if not log_config.log_metrics:
        return
    
    log_record = logger_instance.makeRecord(
        logger_instance.name,
        logging.INFO,
        "(metrics)",
        0,
        f"Performance metrics: {json.dumps(metrics)}",
        (),
        None,
    )
    
    if request_id:
        log_record.request_id = request_id
    
    log_record.metrics = metrics
    logger_instance.handle(log_record)


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"{func.__name__} completed in {elapsed_ms:.2f}ms")
            return result
        except Exception:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.exception(f"{func.__name__} failed after {elapsed_ms:.2f}ms")
            raise
    
    return wrapper


def validate_generation_params(
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    repetition_penalty: float
) -> None:
    """
    Validate generation parameters.
    
    Args:
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_new_tokens: Maximum tokens to generate
        repetition_penalty: Repetition penalty factor
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    if not 0.0 < temperature <= 2.0:
        raise ValidationError(f"temperature must be greater than 0.0 and at most 2.0, got {temperature}")
    
    if not 0.0 <= top_p <= 1.0:
        raise ValidationError(f"top_p must be between 0.0 and 1.0, got {top_p}")
    
    if top_k < 0:
        raise ValidationError(f"top_k must be non-negative, got {top_k}")
    
    if max_new_tokens <= 0 or max_new_tokens > 8192:
        raise ValidationError(f"max_new_tokens must be between 1 and 8192, got {max_new_tokens}")
    
    if repetition_penalty < 1.0 or repetition_penalty > 2.0:
        raise ValidationError(f"repetition_penalty must be between 1.0 and 2.0, got {repetition_penalty}")


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get current GPU memory usage information.
    
    Returns:
        dict: GPU memory statistics
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"available": False}
        
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        
        return {
            "available": True,
            "device": torch.cuda.get_device_name(device),
            "total_gb": total_memory / (1024**3),
            "allocated_gb": allocated_memory / (1024**3),
            "reserved_gb": reserved_memory / (1024**3),
            "free_gb": (total_memory - allocated_memory) / (1024**3),
        }
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e!s}")
        return {"available": False, "error": str(e)}


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU cache: {e!s}")

