#!/usr/bin/env python3
"""Test round-robin rotation in the key pool."""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

from llm.key_pool import OpenRouterKeyPool, reset_key_pool

def test_key_pool():
    # Reset the global pool.
    reset_key_pool()
    
    # Create a test pool.
    pool = OpenRouterKeyPool(
        key_prefix='OPENROUTER_API_KEY',
        daily_limit=1000,
        minute_limit=20
    )
    
    print(f'\n=== Key Pool Test ===')
    print(f'Loaded keys: {len(pool.keys)}')
    
    if not pool.keys:
        print('No keys found. Please set OPENROUTER_API_KEY_1/2/3...')
        return
    
    # Test round-robin rotation.
    print(f'\n--- Testing round-robin rotation ---')
    acquired_keys = []
    key_names = []
    
    for i in range(min(6, len(pool.keys) * 2)):
        key = pool.acquire_key()
        if key:
            # Find the matching key_name.
            for ks in pool.keys:
                if ks.key == key:
                    key_names.append(ks.key_name)
                    break
            acquired_keys.append(key[:20] + '...')
            print(f'Request {i+1}: acquired {key_names[-1]}')
    
    # Check whether rotation happened.
    print(f'\nRotation order: {key_names}')
    if len(pool.keys) > 1 and len(set(key_names[:len(pool.keys)])) == len(pool.keys):
        print('Round-robin rotation works as expected')
    elif len(pool.keys) == 1:
        print('Only one key is available, so rotation cannot be tested')
    else:
        print('Rotation may not be behaving as expected')
    
    # Show key status.
    print(f'\n--- Key status ---')
    for ks in pool.keys:
        print(f'{ks.key_name}: daily {ks.daily_count}/{ks.daily_limit}, minute {ks.get_minute_count()}/{ks.minute_limit}')
    
    print(f'\nTest completed')


if __name__ == "__main__":
    test_key_pool()
