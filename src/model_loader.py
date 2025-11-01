"""
Model loader with singleton pattern for efficient model management.
Handles model initialization, authentication, and GPU optimization.
"""

import os
import time
import threading
from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.config import model_config, inference_config, server_config
from src.utils import (
    logger,
    ModelLoadError,
    get_gpu_memory_info,
    timing_decorator,
)


class ModelLoader:
    """
    Singleton class for loading and managing the Llama model.
    Ensures model is loaded only once per container lifecycle.
    Thread-safe model loading with double-checked locking pattern.
    """
    
    _instance: Optional['ModelLoader'] = None
    _model = None
    _tokenizer = None
    _model_loaded = False
    _load_start_time: Optional[float] = None
    _load_end_time: Optional[float] = None
    _load_lock: threading.Lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model loader (called once)."""
        if not ModelLoader._model_loaded:
            logger.info("Initializing ModelLoader singleton")
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is loaded."""
        return cls._model_loaded
    
    @classmethod
    def get_model(cls):
        """Get the loaded model instance."""
        if not cls._model_loaded:
            raise ModelLoadError("Model not loaded. Call load_model() first.")
        return cls._model
    
    @classmethod
    def get_tokenizer(cls):
        """Get the loaded tokenizer instance."""
        if not cls._model_loaded:
            raise ModelLoadError("Tokenizer not loaded. Call load_model() first.")
        return cls._tokenizer
    
    @classmethod
    def get_load_time(cls) -> Optional[float]:
        """Get model loading time in seconds."""
        if cls._load_start_time and cls._load_end_time:
            return cls._load_end_time - cls._load_start_time
        return None
    
    @classmethod
    @timing_decorator
    def load_model(cls) -> Tuple[any, any]:
        """
        Load the model and tokenizer with all optimizations.
        This should be called once at container startup.
        Thread-safe using double-checked locking pattern.
        
        Returns:
            tuple: (model, tokenizer)
            
        Raises:
            ModelLoadError: If model loading fails
        """
        # First check without lock (fast path)
        if cls._model_loaded:
            logger.info("Model already loaded, returning existing instance")
            return cls._model, cls._tokenizer
        
        # Acquire lock for thread-safe loading
        with cls._load_lock:
            # Double-check after acquiring lock
            if cls._model_loaded:
                logger.info("Model already loaded (post-lock check), returning existing instance")
                return cls._model, cls._tokenizer
            
            logger.info(f"Starting model loading: {model_config.model_id}")
            cls._load_start_time = time.time()
            
            try:
                # Validate HF token
                if not model_config.hf_token:
                    raise ModelLoadError("HF_TOKEN environment variable is required")
                
                # Log GPU info before loading
                gpu_info = get_gpu_memory_info()
                logger.info(f"GPU info before loading: {gpu_info}")
                
                # Check CUDA availability
                if not torch.cuda.is_available():
                    raise ModelLoadError("CUDA is not available. GPU is required for this model.")
                
                logger.info(f"Using device: CUDA - {torch.cuda.get_device_name(0)}")
                
                # Determine torch dtype
                torch_dtype = cls._get_torch_dtype()
                logger.info(f"Using dtype: {torch_dtype}")
                
                # Configure quantization if requested
                quantization_config = cls._get_quantization_config()
                
                # Load tokenizer
                logger.info("Loading tokenizer...")
                cls._tokenizer = AutoTokenizer.from_pretrained(
                    model_config.model_id,
                    token=model_config.hf_token,
                    cache_dir=model_config.cache_dir,
                    trust_remote_code=model_config.trust_remote_code,
                )
                
                # Set padding token if not set
                if cls._tokenizer.pad_token is None:
                    cls._tokenizer.pad_token = cls._tokenizer.eos_token
                
                logger.info("Tokenizer loaded successfully")
                
                # Prepare model loading kwargs
                model_kwargs = {
                    "pretrained_model_name_or_path": model_config.model_id,
                    "token": model_config.hf_token,
                    "cache_dir": model_config.cache_dir,
                    "device_map": model_config.device_map,
                    "trust_remote_code": model_config.trust_remote_code,
                    "low_cpu_mem_usage": True,
                }
                
                # Add dtype if not using quantization
                if quantization_config is None:
                    model_kwargs["torch_dtype"] = torch_dtype
                else:
                    model_kwargs["quantization_config"] = quantization_config
                
                # Add Flash Attention 2 if available and enabled
                if model_config.use_flash_attention and cls._check_flash_attention_available():
                    logger.info("Enabling Flash Attention 2")
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                
                # Load model
                logger.info("Loading model... This may take several minutes.")
                cls._model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                
                # Set to evaluation mode
                cls._model.eval()
                
                # Ensure pad_token_id is set on model config to avoid generation warnings
                if getattr(cls._model.config, "pad_token_id", None) is None:
                    if cls._tokenizer.pad_token_id is not None:
                        cls._model.config.pad_token_id = cls._tokenizer.pad_token_id
                        logger.info(f"Set model pad_token_id to {cls._tokenizer.pad_token_id}")
                
                cls._load_end_time = time.time()
                load_time = cls.get_load_time()
                
                # Log GPU info after loading
                gpu_info_after = get_gpu_memory_info()
                logger.info(f"GPU info after loading: {gpu_info_after}")
                
                cls._model_loaded = True
                
                logger.info(
                    f"Model loaded successfully in {load_time:.2f} seconds. "
                    f"GPU memory used: {gpu_info_after.get('allocated_gb', 0):.2f} GB"
                )
                
                return cls._model, cls._tokenizer
                
            except Exception as e:
                cls._load_end_time = time.time()
                logger.exception("Failed to load model")
                
                # Log GPU info on failure
                gpu_info = get_gpu_memory_info()
                logger.error(f"GPU info at failure: {gpu_info}")
                
                raise ModelLoadError(f"Model loading failed: {e!s}") from e
    
    @staticmethod
    def _get_torch_dtype():
        """
        Determine appropriate torch dtype based on configuration.
        
        Returns:
            torch.dtype: The dtype to use for model loading
        """
        if model_config.torch_dtype == "auto":
            return "auto"
        elif model_config.torch_dtype == "float16":
            return torch.float16
        elif model_config.torch_dtype == "bfloat16":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                logger.warning("bfloat16 not supported, falling back to float16")
                return torch.float16
        else:
            logger.warning(f"Unknown dtype {model_config.torch_dtype}, using auto")
            return "auto"
    
    @staticmethod
    def _get_quantization_config() -> Optional[BitsAndBytesConfig]:
        """
        Get quantization configuration if enabled.
        
        Returns:
            Optional[BitsAndBytesConfig]: Quantization config or None
        """
        if model_config.load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise ModelLoadError(
                    "4-bit quantization requested but bitsandbytes is not installed. "
                    "Install with: pip install bitsandbytes"
                ) from e
            logger.info("Using 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif model_config.load_in_8bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise ModelLoadError(
                    "8-bit quantization requested but bitsandbytes is not installed. "
                    "Install with: pip install bitsandbytes"
                ) from e
            logger.info("Using 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None
    
    @staticmethod
    def _check_flash_attention_available() -> bool:
        """
        Check if Flash Attention 2 is available.
        
        Returns:
            bool: True if Flash Attention 2 is available
        """
        try:
            import flash_attn
            logger.info("Flash Attention 2 is available")
            return True
        except ImportError:
            logger.info("Flash Attention 2 not available, using standard attention")
            return False
    
    @classmethod
    def warmup(cls, warmup_prompt: str = "Hello, how are you?") -> bool:
        """
        Perform a warmup inference to ensure model is ready.
        
        Args:
            warmup_prompt: Prompt to use for warmup
            
        Returns:
            bool: True if warmup successful
        """
        if not cls._model_loaded:
            logger.warning("Cannot warmup: model not loaded")
            return False
        
        try:
            logger.info("Performing model warmup...")
            start_time = time.time()
            
            inputs = cls._tokenizer(
                warmup_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(cls._model.device)
            
            with torch.no_grad():
                cls._model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=True,
                )
            
            warmup_time = time.time() - start_time
            logger.info(f"Model warmup completed in {warmup_time:.2f} seconds")
            
            # Clear cache after warmup
            torch.cuda.empty_cache()
            
            return True
            
        except Exception:
            logger.exception("Warmup failed")
            return False


# Global function to get or initialize the model loader
def get_model_loader() -> ModelLoader:
    """
    Get the singleton model loader instance.
    
    Returns:
        ModelLoader: The model loader instance
    """
    return ModelLoader()


# Auto-load model at module import (for container startup)
if os.getenv("AUTOLOAD_MODEL", "true").lower() == "true":
    logger.info("Auto-loading model at startup...")
    try:
        loader = get_model_loader()
        loader.load_model()
        
        # Perform warmup if enabled
        if server_config.model_warmup:
            loader.warmup(server_config.warmup_prompt)
        
        logger.info("Model initialization complete and ready for inference")
    except Exception:
        logger.exception("Failed to auto-load model")
        # Don't raise here - let the handler deal with it

