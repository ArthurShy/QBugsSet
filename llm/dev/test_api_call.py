#!/usr/bin/env python3
"""Real API-call test for verifying key-pool rotation."""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
# Enable key-pool debug logging.
logging.getLogger("llm.key_pool").setLevel(logging.DEBUG)

from api_clients import OpenRouterClient
from llm.key_pool import reset_key_pool

def test_api_calls(num_calls: int = 6):
    """Run multiple API calls to test key-pool rotation."""
    # Reset the global pool.
    reset_key_pool()
    
    print(f"\n=== Real API Call Test ===")
    print(f"Number of calls: {num_calls}")
    
    # Create the client using a free model.
    try:
        client = OpenRouterClient(model="qwen3-coder-free")
    except ValueError as e:
        print(f"Failed to create client: {e}")
        return
    
    print(f"Key-pool status: {client.key_pool.get_available_count()} available\n")
    
    results = []
    for i in range(num_calls):
        print(f"--- Request {i+1}/{num_calls} ---")
        start = time.time()
        
        response = client.chat([
            {"role": "user", "content": "Say 'OK' only."}
        ])
        
        elapsed = time.time() - start
        
        if response.success:
            content = response.content.strip()[:50]
            print(f"Success ({elapsed:.1f}s): {content}")
            results.append(True)
        else:
            print(f"Failed: {response.error}")
            results.append(False)
        
        # Add a short pause to avoid sending requests too quickly.
        if i < num_calls - 1:
            time.sleep(0.5)
    
    # Summarize results.
    print(f"\n=== Test Results ===")
    success_count = sum(results)
    print(f"Successes: {success_count}/{num_calls}")
    
    if client.key_pool:
        print(f"\n--- Key usage stats ---")
        for ks in client.key_pool.keys:
            print(f"{ks.key_name}: daily {ks.daily_count}, minute {ks.get_minute_count()}")
    
    print(f"\n{'Test completed successfully' if success_count == num_calls else 'Some requests failed'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=6, help="Number of test calls")
    args = parser.parse_args()
    
    test_api_calls(args.num)
