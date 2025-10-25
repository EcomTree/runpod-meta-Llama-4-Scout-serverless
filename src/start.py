"""
Startup script that runs both the health check server and RunPod handler.
"""

import sys
import threading
from src.utils import logger
from src.config import validate_config, get_config_summary


def start_health_server_thread():
    """Start health check server in a separate thread."""
    try:
        from src.health_server import start_health_server
        logger.info("Starting health check server in background thread...")
        start_health_server()
    except Exception as e:
        # Catch all exceptions since start_health_server() can raise many types
        # (ValueError, TypeError, ConfigError, RuntimeError, OSError, etc.)
        logger.error(
            "Health server failed to start. Health monitoring is DISABLED for this process. "
            "This is non-fatal and the main handler will continue running, but health checks will not be available. "
            f"Exception: {e!s}"
        )
        logger.debug("Exception details:", exc_info=True)
def main():
    """Main startup function."""
    logger.info("=" * 80)
    logger.info("Starting Llama-4-Scout-17B-16E-Instruct RunPod Serverless Handler")
    logger.info("=" * 80)
    
    # Validate configuration
    logger.info("Validating configuration...")
    is_valid, errors = validate_config()
    
    if not is_valid:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Log configuration summary
    config_summary = get_config_summary()
    logger.info("Configuration summary:")
    for key, value in config_summary.items():
        logger.info(f"  {key}: {value}")
    # thread exits, even if it is currently handling a health check request. This can
    # result in a health check request being interrupted mid-response. However, this is
    # acceptable because:
    # 1. Health checks are stateless and idempotent, so interrupted requests can be retried safely.
    # 2. Health check requests are quick and don't maintain long-lived state.
    # 3. RunPod handles graceful shutdowns at the platform level.
    # 4. The main handler is the critical path; health server is auxiliary.
    # 1. Health check requests are quick and don't maintain long-lived state
    # 2. RunPod handles graceful shutdowns at the platform level
    # 3. The main handler is the critical path; health server is auxiliary
    # The health server runs uvicorn.Server which blocks, but since it's in a daemon
    # thread, it won't prevent the main thread from starting the RunPod handler.
    health_thread = threading.Thread(target=start_health_server_thread, daemon=True)
    health_thread.start()
    logger.info("Health check server thread started")
    
    # Import and start RunPod handler
    try:
        import runpod
        from src.handler import handler
        
        logger.info("Starting RunPod serverless handler...")
        logger.info("Handler is now ready to accept requests")
        
        # Start RunPod serverless worker
        runpod.serverless.start({"handler": handler})
        
    except ImportError as e:
        logger.error("Failed to import runpod. Is runpod package installed?")
        logger.error(f"Error: {e!s}")
        sys.exit(1)
    except Exception as e:
        # Catch all exceptions since runpod.serverless.start() can raise many types
        # (ValueError, TypeError, AttributeError, KeyError, RuntimeError, OSError, etc.)
        logger.exception(f"Failed to start RunPod handler: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()

