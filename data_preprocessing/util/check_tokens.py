# -*- coding: utf-8 -*-
import os
import sys
import requests
import datetime
import argparse
from pathlib import Path

try:
    import config
except ImportError:
    file_path = Path(__file__).resolve()
    project_root = file_path.parent.parent.parent
    sys.path.append(str(project_root))
    import config

TOKEN_ENV_VARS = config.GITHUB_TOKEN_ENV_VARS
GITHUB_API_URL = "https://api.github.com/rate_limit"

def validate_tokens(min_required=1):
    """Ensure that at least ``min_required`` tokens are configured."""
    valid_tokens = 0
    for token_name in TOKEN_ENV_VARS:
        if os.getenv(token_name):
            valid_tokens += 1
    
    if valid_tokens < min_required:
        print(f"❌ Found {valid_tokens} valid GitHub token(s), but at least {min_required} are required.", file=sys.stderr)
        print(f"   Set one or more of these environment variables: {', '.join(TOKEN_ENV_VARS)}", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ Found {valid_tokens} valid GitHub token(s).")

def check_rate_limit(token_name, token_value):
    """Query and print GitHub API rate-limit buckets for one token."""
    headers = {"Authorization": f"token {token_value}"}
    
    print(f"--- Checking: {token_name} ---")
    
    try:
        response = requests.get(GITHUB_API_URL, headers=headers)
        
        if response.status_code == 200:
            resources = response.json().get("resources", {})

            def format_reset(reset_timestamp):
                if isinstance(reset_timestamp, int):
                    reset_time = datetime.datetime.fromtimestamp(reset_timestamp)
                    now = datetime.datetime.now()
                    delta = reset_time - now
                    minutes_to_reset = delta.total_seconds() / 60
                    return f"{reset_time.strftime('%Y-%m-%d %H:%M:%S')} (about {minutes_to_reset:.1f} minutes from now)"
                return "N/A"

            for bucket_name in ("core", "search"):
                bucket = resources.get(bucket_name, {})
                limit = bucket.get("limit", "N/A")
                remaining = bucket.get("remaining", "N/A")
                reset_str = format_reset(bucket.get("reset", "N/A"))

                print(f"  [{bucket_name}]")
                print(f"    Remaining: {remaining}")
                print(f"    Limit:     {limit}")
                print(f"    Reset:     {reset_str}")

        elif response.status_code == 401:
            print("  [Error] Invalid token. Check whether it is correct or expired.")
        else:
            print(f"  [Error] Request failed with status {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"  [Error] Request exception: {e}")
    
    print("-" * (len(token_name) + 14) + "\n")


def main():
    """Run validation or detailed rate-limit checks."""
    parser = argparse.ArgumentParser(description="Check GitHub token availability and rate limits.")
    parser.add_argument(
        '--check',
        type=int,
        nargs='?',
        const=1,
        metavar='MIN_REQUIRED',
        help='Validate that at least this many tokens are configured (default: 1).'
    )
    args = parser.parse_args()

    if args.check is not None:
        validate_tokens(min_required=args.check)
    else:
        print("Checking GitHub token rate limits...\n")
        found_any = False
        
        for token_name in TOKEN_ENV_VARS:
            token_value = os.getenv(token_name)
            
            if token_value:
                found_any = True
                check_rate_limit(token_name, token_value)
            else:
                print(f"--- Environment variable not set: {token_name} ---\n")
                
        if not found_any:
            print("No GitHub tokens were found in the environment.")
            print(f"Set one or more of these environment variables: {', '.join(TOKEN_ENV_VARS)}")

if __name__ == "__main__":
    main() 
