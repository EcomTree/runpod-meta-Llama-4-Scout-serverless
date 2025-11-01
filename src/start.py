"""
Startup script that runs both the health check server and RunPod handler.
"""

import sys
import threading
from src.utils import logger
from src.config import validate_config, get_config_summary

# Note on HTTP/1.1 enforcement:
# Standard urllib3 (v1.x and v2.x) does not support HTTP/2 - it only uses HTTP/1.1.
# HTTP/2 support only exists in the experimental urllib3.future package.
# Since this project uses standard urllib3 (via requests/runpod dependencies),
# no HTTP/2 configuration is needed for urllib3.
#
# The health check server (uvicorn) is configured to use HTTP/1.1 only via
# the http="h11" parameter in src/health_server.py, which forces the h11
# HTTP/1.1 implementation instead of httptools (which could support HTTP/2).
#
# See Dockerfile for the accompanying PYTHONHTTPSVERIFY=1 setting that keeps
# TLS certificate verification enabled even when third-party packages try to
# modify urllib3/requests defaults. Together with the HTTP/1.1 enforcement
# below, this ensures outbound requests remain secure and predictable.
#
# Additionally, we configure urllib3 to disable HTTP/2 if available (for future-proofing
# in case experimental urllib3.future is ever used):
try:
    import urllib3
    # Disable HTTP/2 support in urllib3 (for versions that support it)
    if hasattr(urllib3.util, "connection"):
        if hasattr(urllib3.util.connection, "HAS_HTTP2"):
            urllib3.util.connection.HAS_HTTP2 = False
except (ImportError, AttributeError):
    # urllib3 not available or doesn't support this configuration
    pass


def start_health_server_thread():
    """Start health check server in a separate thread."""
    try:
        from src.health_server import start_health_server
        logger.info("Starting health check server in background thread...")
        start_health_server()
    except (ValueError, TypeError, RuntimeError, OSError, ImportError) as e:
        # Catch specific exceptions that start_health_server() can raise
        logger.error(
            "Health server failed to start. Health monitoring is DISABLED for this process. "
            "This is non-fatal and the main handler will continue running, but health checks will not be available. "
            f"Exception: {e!s}"
        )
        logger.debug("Exception details:", exc_info=True)
    except Exception as e:
        # Catch any other unexpected exceptions
        logger.error(
            "Health server failed with unexpected error. Health monitoring is DISABLED for this process. "
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
    # The daemon=True flag means the health server thread will be terminated when the main
    # thread exits, even if it is currently handling a health check request. This can
    # result in a health check request being interrupted mid-response. However, this is
    # acceptable because:
    # 1. Health checks are stateless and idempotent, so interrupted requests can be retried safely.
    # 2. Health check requests are quick and don't maintain long-lived state.
    # 3. RunPod handles graceful shutdowns at the platform level.
    # 4. The main handler is the critical path; health server is auxiliary.
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
    except (ValueError, TypeError, AttributeError, KeyError, RuntimeError, OSError) as e:
        # Catch specific exceptions that runpod.serverless.start() can raise
        logger.exception(f"Failed to start RunPod handler: {e!s}")
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected exceptions with full traceback for debugging
        logger.exception(f"Failed to start RunPod handler with unexpected error: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()

