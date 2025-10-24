"""
Startup script that runs both the health check server and RunPod handler.
"""

import os
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
        logger.error(f"Health server failed: {str(e)}", exc_info=True)


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
    
    # Start health check server in background thread
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
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start RunPod handler: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

