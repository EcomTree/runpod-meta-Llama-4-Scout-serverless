"""
Local testing script for the RunPod handler.
Simulates RunPod request format and tests the handler locally.

Run this script as a module for local testing:
  python -m scripts.test_local
"""

import os
import sys
import json
import argparse
from typing import Dict, Any


def test_handler_simple():
    """Test handler with a simple prompt."""
    from src.handler import handler
    
    print("=" * 80)
    print("Testing handler with simple prompt...")
    print("=" * 80)
    
    test_event = {
        "input": {
            "prompt": "What is artificial intelligence? Explain in simple terms.",
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }
    
    print(f"\nInput: {json.dumps(test_event, indent=2)}")
    print("\nGenerating response...")
    
    result = handler(test_event)
    
    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    
    return result


def test_handler_custom(prompt: str, max_new_tokens: int = 100, temperature: float = 0.7):
    """Test handler with custom parameters."""
    from src.handler import handler
    
    print("=" * 80)
    print("Testing handler with custom prompt...")
    print("=" * 80)
    
    test_event = {
        "input": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
    }
    
    print(f"\nInput: {json.dumps(test_event, indent=2)}")
    print("\nGenerating response...")
    
    result = handler(test_event)
    
    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    
    return result


def test_handler_edge_cases():
    """Test handler with edge cases and error scenarios."""
    from src.handler import handler
    
    print("=" * 80)
    print("Testing edge cases...")
    print("=" * 80)
    
    # Test 1: Very short prompt
    print("\n1. Testing very short prompt...")
    result1 = handler({
        "input": {
            "prompt": "Hi",
            "max_new_tokens": 50,
        }
    })
    print(f"Result: {'Success' if 'output' in result1 else 'Error'}")
    
    # Test 2: Missing prompt (should fail)
    print("\n2. Testing missing prompt (should fail)...")
    result2 = handler({
        "input": {
            "max_new_tokens": 50,
        }
    })
    print(f"Result: {'Error (expected)' if 'error' in result2 else 'Unexpected success'}")
    
    # Test 3: Invalid temperature (should fail)
    print("\n3. Testing invalid temperature (should fail)...")
    result3 = handler({
        "input": {
            "prompt": "Test",
            "temperature": 5.0,
        }
    })
    print(f"Result: {'Error (expected)' if 'error' in result3 else 'Unexpected success'}")
    
    # Test 4: Long prompt
    print("\n4. Testing long prompt...")
    long_prompt = "Explain the concept of machine learning. " * 20
    result4 = handler({
        "input": {
            "prompt": long_prompt,
            "max_new_tokens": 100,
        }
    })
    print(f"Result: {'Success' if 'output' in result4 else 'Error'}")
    
    print("\n" + "=" * 80)
    print("Edge case testing complete")
    print("=" * 80)


def test_health_check():
    """Test the health check endpoint."""
    from src.handler import health_check
    
    print("=" * 80)
    print("Testing health check...")
    print("=" * 80)
    
    result = health_check()
    print(json.dumps(result, indent=2))
    
    return result


def benchmark_handler(num_requests: int = 5):
    """Benchmark handler performance with multiple requests."""
    from src.handler import handler
    import time
    
    print("=" * 80)
    print(f"Benchmarking handler with {num_requests} requests...")
    print("=" * 80)
    
    test_event = {
        "input": {
            "prompt": "Write a short story about a robot learning to paint.",
            "max_new_tokens": 200,
            "temperature": 0.8,
        }
    }
    
    timings = []
    
    for i in range(num_requests):
        print(f"\nRequest {i+1}/{num_requests}...")
        start_time = time.time()
        
        result = handler(test_event)
        
        elapsed = time.time() - start_time
        timings.append(elapsed)
        
        if "output" in result:
            tokens = result["output"].get("tokens_generated", 0)
            tokens_per_sec = result["output"].get("tokens_per_second", 0)
            print(f"  Time: {elapsed:.2f}s | Tokens: {tokens} | Speed: {tokens_per_sec:.2f} tok/s")
        else:
            print(f"  Error: {result.get('error', {}).get('message', 'Unknown')}")
    
    print("\n" + "=" * 80)
    print("Benchmark Results:")
    print("=" * 80)
    print(f"Total requests: {num_requests}")
    print(f"Average time: {sum(timings)/len(timings):.2f}s")
    print(f"Min time: {min(timings):.2f}s")
    print(f"Max time: {max(timings):.2f}s")


def main():
    """Main entry point for local testing."""
    parser = argparse.ArgumentParser(description="Local testing for RunPod handler")
    parser.add_argument(
        "--mode",
        choices=["simple", "custom", "edge", "health", "benchmark"],
        default="simple",
        help="Test mode to run"
    )
    parser.add_argument("--prompt", type=str, help="Custom prompt for testing")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--num-requests", type=int, default=5, help="Number of requests for benchmark")
    
    args = parser.parse_args()
    
    # Check for HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        print("ERROR: HF_TOKEN environment variable not set!")
        print("Please set your Hugging Face token securely:")
        print("  Option 1: Load from file (recommended)")
        print("    export HF_TOKEN=$(cat ~/.hf_token)")
        print("  Option 2: Use environment file")
        print("    export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)")
        print("  Option 3: Direct export (will be visible in shell history)")
        print("    export HF_TOKEN='your_token_here'")
        sys.exit(1)
    
    try:
        if args.mode == "simple":
            test_handler_simple()
        elif args.mode == "custom":
            if not args.prompt:
                print("ERROR: --prompt required for custom mode")
                sys.exit(1)
            test_handler_custom(args.prompt, args.max_tokens, args.temperature)
        elif args.mode == "edge":
            test_handler_edge_cases()
        elif args.mode == "health":
            test_health_check()
        elif args.mode == "benchmark":
            benchmark_handler(args.num_requests)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

