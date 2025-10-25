#!/usr/bin/env python3
"""
Simple health check script for Docker container health monitoring.
Checks if the health endpoint returns 200 OK.
"""

import sys
import urllib.request
import urllib.error


def check_health(host: str = "localhost", port: int = 8000) -> bool:
    """
    Check if the health endpoint is responding.
    
    Args:
        host: Host to check (default: localhost)
        port: Port to check (default: 8000)
        
    Returns:
        bool: True if healthy (200 OK), False otherwise
    """
    try:
        url = f"http://{host}:{port}/health"
        response = urllib.request.urlopen(url, timeout=5)
        return response.getcode() == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    is_healthy = check_health()
    sys.exit(0 if is_healthy else 1)
