#!/usr/bin/env python3
"""Query remaining credits for the OpenRouter key pool and optionally exclude exhausted keys."""
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm.key_pool import (
    print_credits_summary, 
    reset_key_pool, 
    get_openrouter_key_pool
)

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    parser = argparse.ArgumentParser(description="Query OpenRouter key-pool credits")
    parser.add_argument(
        "--exclude-exhausted", "-e",
        action="store_true",
        help="Automatically exclude keys with insufficient remaining balance"
    )
    parser.add_argument(
        "--min-balance", "-m",
        type=float,
        default=0.0,
        help="Minimum remaining-balance threshold. Keys below this value will be excluded (default: 0)"
    )
    args = parser.parse_args()
    
    reset_key_pool()
    pool = get_openrouter_key_pool()
    
    # Show the credit summary.
    print_credits_summary(pool)
    
    # Exclude exhausted keys.
    if args.exclude_exhausted:
        print("\nChecking for keys with insufficient remaining balance...")
        excluded = pool.check_and_exclude_exhausted_keys(min_remaining=args.min_balance)
        print(f"\nExcluded {excluded} keys with insufficient remaining balance")
        print(f"Remaining available keys: {pool.get_available_count()}")


if __name__ == "__main__":
    main()
