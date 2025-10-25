"""
RunPod serverless handler for Llama-4-Scout-17B-16E-Instruct.
Main entry point for inference requests.
"""

import time
import torch
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

from src.model_loader import get_model_loader, ModelLoader
from src.config import inference_config, log_config, model_config
from src.utils import (
    logger,
    generate_request_id,
    sanitize_input,
    format_error_response,
    format_success_response,
    validate_generation_params,
    log_metrics,
    clear_gpu_cache,
    InferenceError,
    ValidationError,
    GPUMemoryError,
    TimeoutError as CustomTimeoutError,
)


# Input validation with Pydantic
class InferenceInput(BaseModel):
    """Input schema for inference requests."""
    
    prompt: str = Field(..., description="Input text prompt for generation")
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=0,
        description="Top-k sampling parameter"
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=2.0,
        description="Repetition penalty factor"
    )
    do_sample: Optional[bool] = Field(
        default=True,
        description="Whether to use sampling"
    )
    
    @field_validator("prompt")
    def validate_prompt(cls, v):
        """Validate and sanitize prompt."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return sanitize_input(v, max_length=inference_config.max_input_tokens * 4)
    
    class Config:
        """Pydantic config."""
        extra = "forbid"  # Reject unknown fields


def validate_input(input_data: Dict[str, Any]) -> InferenceInput:
    """
    Validate input data against schema.
    
    Args:
        input_data: Raw input dictionary
        
    Returns:
        InferenceInput: Validated input object
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        return InferenceInput(**input_data)
    except Exception as e:
        raise ValidationError(f"Input validation failed: {e!s}") from e


def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    do_sample: bool = True,
) -> tuple[str, Dict[str, Any]]:
    """
    Generate text using the loaded model.
    
    Args:
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        do_sample: Whether to use sampling
        
    Returns:
        tuple: (generated_text, metrics_dict)
        
    Raises:
        InferenceError: If generation fails
        GPUMemoryError: If GPU runs out of memory
    """
    try:
        # Get model and tokenizer
        model = ModelLoader.get_model()
        tokenizer = ModelLoader.get_tokenizer()
        
        # Start timing
        start_time = time.time()
        
        # Tokenize input
        tokenization_start = time.time()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=inference_config.max_input_tokens,
        ).to(model.device)
        tokenization_time = time.time() - tokenization_start
        
        input_token_count = inputs.input_ids.shape[1]
        logger.info(f"Input tokenized: {input_token_count} tokens")
        
        # Check total token limit
        if input_token_count + max_new_tokens > inference_config.max_total_tokens:
            raise InferenceError(
                f"Total tokens ({input_token_count} + {max_new_tokens}) "
                f"exceeds limit of {inference_config.max_total_tokens}"
            )
        
        # Generate
        generation_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - generation_start
        
        # Decode output
        decoding_start = time.time()
        generated_text = tokenizer.decode(
            outputs[0][input_token_count:],  # Only decode new tokens
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        decoding_time = time.time() - decoding_start
        
        # Calculate metrics
        total_time = time.time() - start_time
        output_token_count = outputs.shape[1] - input_token_count
        tokens_per_second = output_token_count / generation_time if generation_time > 0 else 0
        
        metrics = {
            "tokens_generated": output_token_count,
            "input_tokens": input_token_count,
            "total_tokens": outputs.shape[1],
            "generation_time_ms": int(generation_time * 1000),
            "total_time_ms": int(total_time * 1000),
            "tokenization_time_ms": int(tokenization_time * 1000),
            "decoding_time_ms": int(decoding_time * 1000),
            "tokens_per_second": round(tokens_per_second, 2),
        }
        
        logger.info(f"Generation complete: {output_token_count} tokens in {generation_time:.2f}s "
                   f"({tokens_per_second:.2f} tokens/s)")
        
        return generated_text, metrics
        
    except torch.cuda.OutOfMemoryError as e:
        logger.exception("GPU out of memory during generation")
        clear_gpu_cache()
        raise GPUMemoryError("GPU out of memory. Try reducing max_new_tokens or input length.") from e
    
    except Exception as e:
        logger.exception("Generation failed")
        clear_gpu_cache()
        raise InferenceError(f"Text generation failed: {e!s}") from e


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.
    This is the main entry point called by RunPod for each request.
    
    Args:
        event: RunPod event dictionary with 'input' field
        
    Returns:
        dict: Response dictionary with 'output' or 'error' field
    """
    request_id = generate_request_id()
    request_start_time = time.time()
    
    try:
        logger.info(f"Handler called with request_id: {request_id}")
        
        # Log request if enabled
        if log_config.log_requests:
            logger.debug(f"Request data: {event}")
        
        # Extract input from event
        input_data = event.get("input", {})
        if not input_data:
            raise ValidationError("Missing 'input' field in request")
        
        # Validate input
        validated_input = validate_input(input_data)
        
        # Apply defaults from config
        max_new_tokens = (
            validated_input.max_new_tokens
            if validated_input.max_new_tokens is not None
            else inference_config.max_new_tokens
        )
        temperature = (
            validated_input.temperature
            if validated_input.temperature is not None
            else inference_config.temperature
        )
        top_p = (
            validated_input.top_p
            if validated_input.top_p is not None
            else inference_config.top_p
        )
        top_k = (
            validated_input.top_k
            if validated_input.top_k is not None
            else inference_config.top_k
        )
        repetition_penalty = (
            validated_input.repetition_penalty
            if validated_input.repetition_penalty is not None
            else inference_config.repetition_penalty
        )
        
        # Validate generation parameters
        validate_generation_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        
        # Check if model is loaded
        if not ModelLoader.is_loaded():
            logger.warning("Model not loaded, attempting to load now...")
            loader = get_model_loader()
            loader.load_model()
        
        # Generate text
        generated_text, metrics = generate_text(
            prompt=validated_input.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=validated_input.do_sample,
        )
        
        # Add request-level metrics
        total_request_time = time.time() - request_start_time
        metrics["request_time_ms"] = int(total_request_time * 1000)
        
        # Log metrics
        log_metrics(logger, metrics, request_id)
        
        # Clear GPU cache for next request
        clear_gpu_cache()
        
        # Format and return response
        response = format_success_response(generated_text, metrics, request_id)
        
        logger.info(f"Request {request_id} completed successfully in {total_request_time:.2f}s")
        
        return response
        
    except ValidationError as e:
        logger.warning(f"Validation error for request {request_id}: {e!s}")
        return format_error_response(e, request_id, include_traceback=False)
    
    except GPUMemoryError as e:
        logger.error(f"GPU memory error for request {request_id}: {e!s}")
        return format_error_response(e, request_id, include_traceback=False)
    
    except InferenceError as e:
        logger.error(f"Inference error for request {request_id}: {e!s}")
        return format_error_response(e, request_id, include_traceback=False)
    
    except Exception as e:
        logger.exception(f"Unexpected error for request {request_id}")
        return format_error_response(e, request_id, include_traceback=True)


# Health check function for manual testing
def health_check() -> Dict[str, Any]:
    """
    Perform a health check.
    
    Returns:
        dict: Health status information
    """
    try:
        from src.utils import get_gpu_memory_info
        
        status = {
            "status": "healthy" if ModelLoader.is_loaded() else "initializing",
            "model_loaded": ModelLoader.is_loaded(),
            "model_id": model_config.model_id,  # Safe to expose
        }
        
        if ModelLoader.is_loaded():
            load_time = ModelLoader.get_load_time()
            if load_time:
                status["model_load_time_seconds"] = round(load_time, 2)
        
        # Add GPU info
        gpu_info = get_gpu_memory_info()
        if gpu_info.get("available"):
            status["gpu"] = {
                "device": gpu_info.get("device"),
                "allocated_gb": round(gpu_info.get("allocated_gb", 0), 2),
                "free_gb": round(gpu_info.get("free_gb", 0), 2),
            }
        
        return status
        
    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


# Entry point for local testing
if __name__ == "__main__":
    # Test handler with sample input
    test_event = {
        "input": {
            "prompt": "What is artificial intelligence?",
            "max_new_tokens": 100,
            "temperature": 0.7,
        }
    }
    
    print("Testing handler with sample input...")
    result = handler(test_event)
    print(f"Result: {result}")

