"""
FastAPI health check server for container health monitoring.
Runs alongside the RunPod handler to provide health status.
"""

from typing import Dict, Any
from fastapi import FastAPI, Response, status
import uvicorn

from src.model_loader import ModelLoader
from src.config import server_config, log_config
from src.utils import logger, get_gpu_memory_info


# Create FastAPI app
app = FastAPI(
    title="Llama-4-Scout RunPod Health Check",
    description="Health monitoring endpoint for RunPod serverless deployment",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "Llama-4-Scout-17B-16E-Instruct RunPod Handler",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check(response: Response) -> Dict[str, Any]:
    """
    Health check endpoint for container monitoring.
    Returns 200 if healthy, 503 if not ready or unhealthy.
    
    Returns:
        dict: Health status information
    """
    try:
        # Check if model is loaded
        model_loaded = ModelLoader.is_loaded()
        
        if not model_loaded:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "initializing",
                "model_loaded": False,
                "message": "Model is still loading",
            }
        
        # Get GPU memory info
        gpu_info = get_gpu_memory_info()
        
        # Build health response
        health_data = {
            "status": "healthy",
            "model_loaded": True,
            "model_ready": True,
        }
        
        # Add load time if available
        load_time = ModelLoader.get_load_time()
        if load_time:
            health_data["model_load_time_seconds"] = round(load_time, 2)
        
        # Add GPU info
        if gpu_info.get("available"):
            health_data["gpu"] = {
                "device": gpu_info.get("device", "unknown"),
                "memory_allocated_gb": round(gpu_info.get("allocated_gb", 0), 2),
                "memory_free_gb": round(gpu_info.get("free_gb", 0), 2),
                "memory_total_gb": round(gpu_info.get("total_gb", 0), 2),
            }
            
            # Check if GPU memory is critically low
            if gpu_info.get("free_gb", 0) < 1.0:
                logger.warning("GPU memory critically low")
                health_data["warnings"] = ["GPU memory critically low"]
        
        response.status_code = status.HTTP_200_OK
        return health_data
        
    except Exception as e:
        logger.exception("Health check failed")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@app.get("/ready")
async def readiness_check(response: Response) -> Dict[str, Any]:
    """
    Readiness probe endpoint.
    Returns 200 only when model is fully loaded and ready for inference.
    
    Returns:
        dict: Readiness status
    """
    model_loaded = ModelLoader.is_loaded()
    
    if model_loaded:
        response.status_code = status.HTTP_200_OK
        return {
            "ready": True,
            "message": "Service is ready to accept requests",
        }
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "ready": False,
            "message": "Service is not ready, model still loading",
        }


@app.get("/liveness")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness probe endpoint.
    Returns 200 to indicate the service is alive.
    
    Returns:
        dict: Liveness status
    """
    return {
        "alive": True,
        "message": "Service is alive",
    }


@app.get("/metrics")
async def metrics_endpoint() -> Dict[str, Any]:
    """
    Metrics endpoint for monitoring.
    
    Returns:
        dict: Service metrics
    """
    try:
        gpu_info = get_gpu_memory_info()
        load_time = ModelLoader.get_load_time()
        
        metrics = {
            "model_loaded": ModelLoader.is_loaded(),
        }
        
        if load_time:
            metrics["model_load_time_seconds"] = round(load_time, 2)
        
        if gpu_info.get("available"):
            metrics["gpu_memory_allocated_gb"] = round(gpu_info.get("allocated_gb", 0), 2)
            metrics["gpu_memory_free_gb"] = round(gpu_info.get("free_gb", 0), 2)
            metrics["gpu_memory_total_gb"] = round(gpu_info.get("total_gb", 0), 2)
            metrics["gpu_memory_utilization_percent"] = round(
                (gpu_info.get("allocated_gb", 0) / gpu_info.get("total_gb", 1)) * 100, 2
            )
        
        return metrics
        
    except Exception as e:
        logger.exception("Metrics endpoint failed")
        return {"error": str(e)}


def convert_log_level(level):
    """
    Convert log level to a string suitable for Uvicorn.
    Accepts int, str (digit), or str (name).
    
    Args:
        level: Log level as int (10, 20, 30, 40, 50) or str ("debug", "info", etc.)
        
    Returns:
        str: Lowercase log level string suitable for Uvicorn
    """
    # Map known log levels to Uvicorn log level strings
    uvicorn_levels = {
        10: "debug",
        20: "info",
        30: "warning",
        40: "error",
        50: "critical",
        "debug": "debug",
        "info": "info",
        "warning": "warning",
        "error": "error",
        "critical": "critical",
    }
    
    if isinstance(level, str):
        if level.isdigit():
            level = int(level)
        else:
            level = level.lower()
    
    return uvicorn_levels.get(level, "info")


def start_health_server():
    """
    Start the health check server.
    Safe to call from a non-main thread by disabling signal handler installation.
    
    Note: This implementation uses uvicorn internal APIs (install_signal_handlers).
    Tested with uvicorn>=0.25.0. May need adjustment if uvicorn internals change.
    """
    import asyncio
    
    logger.info(
        f"Starting health check server on {server_config.host}:{server_config.port}"
    )
    
    log_level_str = convert_log_level(log_config.level)
    
    # Configure uvicorn without signal handlers to allow running in a thread
    # This prevents "ValueError: signal only works in main thread" error
    config = uvicorn.Config(
        app,
        host=server_config.host,
        port=server_config.port,
        log_level=log_level_str,
        access_log=False,  # Reduce noise
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    
    # Disable signal handler installation (only works in main thread)
    # Note: Disabling via method assignment since uvicorn doesn't provide
    # a documented API for thread-safe server execution without signals
    server.install_signal_handlers = lambda: None
    
    # Run the server - asyncio.run() creates its own event loop
    asyncio.run(server.serve())


# Entry point for running as standalone
if __name__ == "__main__":
    start_health_server()

