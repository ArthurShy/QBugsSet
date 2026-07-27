#!/usr/bin/env python3
"""
OpenRouter API key-pool manager.

Features:
1. Supports rotating across multiple API keys (OPENROUTER_API_KEY_1, 2, 3...)
2. Enforces a 1000-request daily limit per key
3. Enforces a rolling 20-request-per-minute limit
4. Detects rate limits automatically and switches keys
"""

import os
import time
import threading
import logging
import requests
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class KeyStats:
    """Usage statistics for a single key."""
    key: str
    key_name: str  # Environment variable name, for example OPENROUTER_API_KEY_1.
    
    # Daily limit.
    daily_limit: int = 1000
    daily_count: int = 0
    daily_reset_time: datetime = field(default_factory=datetime.now)
    
    # Per-minute rolling-window limit.
    minute_limit: int = 20
    minute_window: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # State.
    is_exhausted: bool = False  # Daily quota exhausted.
    is_rate_limited: bool = False  # Currently rate-limited.
    is_credit_exhausted: bool = False  # Account balance exhausted or overdrawn.
    rate_limit_until: Optional[datetime] = None
    last_error: Optional[str] = None
    
    def __post_init__(self):
        self.minute_window = deque(maxlen=100)
    
    def reset_daily_if_needed(self):
        """Check whether the daily counter should be reset and reset it if needed."""
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self.daily_count = 0
            self.daily_reset_time = now
            self.is_exhausted = False
            logger.info(f"[KeyPool] {self.key_name} daily counter reset")
    
    def get_minute_count(self) -> int:
        """Return the number of requests made in the last rolling minute."""
        now = time.time()
        cutoff = now - 60
        # Remove expired records.
        while self.minute_window and self.minute_window[0] < cutoff:
            self.minute_window.popleft()
        return len(self.minute_window)
    
    def can_use(self) -> bool:
        """Check whether this key is currently available."""
        self.reset_daily_if_needed()
        
        # Check whether the account balance is exhausted.
        if self.is_credit_exhausted:
            return False
        
        # Check the daily limit.
        if self.daily_count >= self.daily_limit:
            self.is_exhausted = True
            return False
        
        # Check rate-limit state.
        if self.is_rate_limited:
            if self.rate_limit_until and datetime.now() >= self.rate_limit_until:
                self.is_rate_limited = False
                self.rate_limit_until = None
            else:
                return False
        
        # Check the per-minute limit.
        if self.get_minute_count() >= self.minute_limit:
            return False
        
        return True
    
    def record_request(self):
        """Record a completed request."""
        self.daily_count += 1
        self.minute_window.append(time.time())
        
        if self.daily_count >= self.daily_limit:
            self.is_exhausted = True
            logger.warning(f"[KeyPool] {self.key_name} daily quota exhausted ({self.daily_count}/{self.daily_limit})")
    
    def pre_acquire(self) -> bool:
        """Reserve quota for one request before sending it and return whether the reservation succeeded."""
        self.reset_daily_if_needed()
        
        # Check whether the account balance is exhausted.
        if self.is_credit_exhausted:
            return False
        
        # Check the daily limit.
        if self.daily_count >= self.daily_limit:
            self.is_exhausted = True
            return False
        
        # Check rate-limit state.
        if self.is_rate_limited:
            if self.rate_limit_until and datetime.now() >= self.rate_limit_until:
                self.is_rate_limited = False
                self.rate_limit_until = None
            else:
                return False
        
        # Check the per-minute limit.
        if self.get_minute_count() >= self.minute_limit:
            return False
        
        # Quota reservation succeeded, so record it immediately.
        self.daily_count += 1
        self.minute_window.append(time.time())
        
        if self.daily_count >= self.daily_limit:
            self.is_exhausted = True
            logger.warning(f"[KeyPool] {self.key_name} daily quota exhausted ({self.daily_count}/{self.daily_limit})")
        
        return True
    
    def record_rate_limit(self, retry_after: int = 60):
        """Record a rate-limit event."""
        self.is_rate_limited = True
        self.rate_limit_until = datetime.now() + timedelta(seconds=retry_after)
        logger.warning(f"[KeyPool] {self.key_name} rate-limited, retry after {retry_after}s")
    
    def record_error(self, error: str):
        """Record an error."""
        self.last_error = error
        if "429" in error or "rate" in error.lower():
            self.record_rate_limit()
        elif "402" in error or "credit" in error.lower():
            self.is_exhausted = True
            logger.warning(f"[KeyPool] {self.key_name} insufficient balance")
    
    def get_status(self) -> Dict:
        """Return a status summary."""
        return {
            "key_name": self.key_name,
            "daily_used": f"{self.daily_count}/{self.daily_limit}",
            "minute_used": f"{self.get_minute_count()}/{self.minute_limit}",
            "is_exhausted": self.is_exhausted,
            "is_rate_limited": self.is_rate_limited,
            "available": self.can_use()
        }


class OpenRouterKeyPool:
    """OpenRouter API key-pool manager."""
    
    def __init__(
        self,
        key_prefix: str = "OPENROUTER_API_KEY",
        daily_limit: int = 1000,
        minute_limit: int = 20
    ):
        """
        Initialize the key pool.
        
        Args:
            key_prefix: Environment-variable prefix. The loader will automatically check {prefix}_1, {prefix}_2, and so on.
            daily_limit: Daily request limit per key.
            minute_limit: Per-minute request limit per key.
        """
        self.key_prefix = key_prefix
        self.daily_limit = daily_limit
        self.minute_limit = minute_limit
        
        self.keys: List[KeyStats] = []
        self.current_index: int = 0
        self.lock = threading.Lock()
        
        self._load_keys()
    
    def _load_keys(self):
        """Load all keys from environment variables."""
        # First try numbered keys.
        for i in range(1, 20):  # Support up to 20 keys.
            key_name = f"{self.key_prefix}_{i}"
            key_value = os.environ.get(key_name)
            if key_value:
                self.keys.append(KeyStats(
                    key=key_value,
                    key_name=key_name,
                    daily_limit=self.daily_limit,
                    minute_limit=self.minute_limit
                ))
                logger.info(f"[KeyPool] Loaded {key_name}")
        
        # If no numbered keys are found, try the unnumbered key.
        if not self.keys:
            key_name = self.key_prefix
            key_value = os.environ.get(key_name)
            if key_value:
                self.keys.append(KeyStats(
                    key=key_value,
                    key_name=key_name,
                    daily_limit=self.daily_limit,
                    minute_limit=self.minute_limit
                ))
                logger.info(f"[KeyPool] Loaded {key_name} (unnumbered)")
        
        if not self.keys:
            logger.warning(f"[KeyPool] No {self.key_prefix} environment variables found")
        else:
            logger.info(f"[KeyPool] Loaded {len(self.keys)} API keys in total")
    
    def get_key(self) -> Optional[str]:
        """
        Get an available API key without reserving quota.
        
        Returns:
            Available API key, or None if no key is currently available.
        """
        with self.lock:
            if not self.keys:
                return None
            
            # Starting from the current index, try to find an available key.
            tried = 0
            while tried < len(self.keys):
                key_stats = self.keys[self.current_index]
                
                if key_stats.can_use():
                    return key_stats.key
                
                # Move to the next key.
                self.current_index = (self.current_index + 1) % len(self.keys)
                tried += 1
            
            # No keys are available.
            return None
    
    def acquire_key(self) -> Optional[str]:
        """
        Get an API key and reserve quota for it with round-robin rotation.
        
        This is the recommended method: reserve quota when acquiring the key
        to avoid oversubscription under concurrency. Each call advances to the
        next key to balance load.
        
        Returns:
            Available API key, or None if no key is currently available.
        """
        with self.lock:
            if not self.keys:
                return None
            
            # Round-robin: try all keys.
            tried = 0
            while tried < len(self.keys):
                key_stats = self.keys[self.current_index]
                
                # Try reserving quota.
                if key_stats.pre_acquire():
                    acquired_key = key_stats.key
                    logger.debug(
                        f"[KeyPool] Allocated {key_stats.key_name} "
                        f"(day: {key_stats.daily_count}/{key_stats.daily_limit}, "
                        f"minute: {key_stats.get_minute_count()}/{key_stats.minute_limit})"
                    )
                    # Move to the next key for round-robin balancing.
                    self.current_index = (self.current_index + 1) % len(self.keys)
                    return acquired_key
                
                # Current key is unavailable, try the next one.
                self.current_index = (self.current_index + 1) % len(self.keys)
                tried += 1
            
            # No keys are available.
            return None
    
    def get_key_with_stats(self) -> Optional[tuple]:
        """
        Get an available API key together with its statistics object.
        
        Returns:
            (key, key_stats) or None.
        """
        with self.lock:
            if not self.keys:
                return None
            
            tried = 0
            while tried < len(self.keys):
                key_stats = self.keys[self.current_index]
                
                if key_stats.can_use():
                    return key_stats.key, key_stats
                
                self.current_index = (self.current_index + 1) % len(self.keys)
                tried += 1
            
            return None
    
    def record_success(self, key: str):
        """
        Record a successful request when get_key() was used.

        If acquire_key() was used, quota has already been reserved, so calling
        this method would double count. Only call it after get_key().
        """
        # acquire_key() already reserves quota, so nothing is needed here.
        pass
    
    def record_success_legacy(self, key: str):
        """Record a successful request for backward compatibility."""
        with self.lock:
            for ks in self.keys:
                if ks.key == key:
                    ks.record_request()
                    break
    
    def record_error(self, key: str, error: str):
        """Record a request error."""
        with self.lock:
            for ks in self.keys:
                if ks.key == key:
                    ks.record_error(error)
                    # If the current key fails, try switching away from it.
                    if not ks.can_use():
                        self.current_index = (self.current_index + 1) % len(self.keys)
                    break
    
    def record_rate_limit(self, key: str, retry_after: int = 60):
        """Record a rate-limit event."""
        with self.lock:
            for ks in self.keys:
                if ks.key == key:
                    ks.record_rate_limit(retry_after)
                    # Switch to the next key.
                    self.current_index = (self.current_index + 1) % len(self.keys)
                    break
    
    def get_status(self) -> Dict:
        """Return the status of all keys."""
        with self.lock:
            return {
                "total_keys": len(self.keys),
                "current_index": self.current_index,
                "keys": [ks.get_status() for ks in self.keys]
            }
    
    def get_available_count(self) -> int:
        """Return the number of keys currently available."""
        with self.lock:
            return sum(1 for ks in self.keys if ks.can_use())
    
    def wait_for_available(self, timeout: float = 60, acquire: bool = True) -> Optional[str]:
        """
        Wait until a key becomes available.
        
        Args:
            timeout: Maximum wait time in seconds.
            acquire: Whether to reserve quota when acquiring the key. Defaults to True.
        
        Returns:
            Available API key, or None on timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            if acquire:
                key = self.acquire_key()
            else:
                key = self.get_key()
            if key:
                return key
            
            # Compute the shortest required wait time.
            min_wait = self._get_min_wait_time()
            if min_wait is None or min_wait > (timeout - (time.time() - start)):
                return None
            
            time.sleep(min(min_wait, 1.0))
        
        return None
    
    def _get_min_wait_time(self) -> Optional[float]:
        """Return the shortest time that must be waited before any key becomes available."""
        min_wait = None
        now = datetime.now()
        
        for ks in self.keys:
            if ks.is_exhausted or ks.is_credit_exhausted:
                continue
            
            if ks.is_rate_limited and ks.rate_limit_until:
                wait = (ks.rate_limit_until - now).total_seconds()
                if wait > 0 and (min_wait is None or wait < min_wait):
                    min_wait = wait
            
            # Check the minute window.
            minute_count = ks.get_minute_count()
            if minute_count >= ks.minute_limit and ks.minute_window:
                oldest = ks.minute_window[0]
                wait = 60 - (time.time() - oldest)
                if wait > 0 and (min_wait is None or wait < min_wait):
                    min_wait = wait
        
        return min_wait
    
    def check_and_exclude_exhausted_keys(self, min_remaining: float = 0.0) -> int:
        """
        Check keys and exclude those with insufficient balance.
        
        Args:
            min_remaining: Minimum remaining-credit threshold. Keys below this value will be excluded.
        
        Returns:
            Number of excluded keys.
        """
        excluded_count = 0
        
        for ks in self.keys:
            try:
                response = requests.get(
                    "https://openrouter.ai/api/v1/credits",
                    headers={"Authorization": f"Bearer {ks.key}"},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                
                if "data" in data:
                    total = data["data"].get("total_credits", 0)
                    usage = data["data"].get("total_usage", 0)
                    remaining = total - usage
                    
                    if remaining < min_remaining:
                        ks.is_credit_exhausted = True
                        excluded_count += 1
                        logger.warning(
                            f"[KeyPool] {ks.key_name} insufficient balance (${remaining:.4f}), excluded"
                        )
                    else:
                        ks.is_credit_exhausted = False
                        logger.info(
                            f"[KeyPool] {ks.key_name} balance OK (${remaining:.4f})"
                        )
            except Exception as e:
                logger.warning(f"[KeyPool] Failed to query balance for {ks.key_name}: {e}")
        
        return excluded_count
    
    def exclude_key(self, key_name: str):
        """Manually exclude a specific key."""
        with self.lock:
            for ks in self.keys:
                if ks.key_name == key_name:
                    ks.is_credit_exhausted = True
                    logger.info(f"[KeyPool] Excluded {key_name}")
                    break
    
    def include_key(self, key_name: str):
        """Re-enable a previously excluded key."""
        with self.lock:
            for ks in self.keys:
                if ks.key_name == key_name:
                    ks.is_credit_exhausted = False
                    logger.info(f"[KeyPool] Re-enabled {key_name}")
                    break


# Global singleton.
_global_pool: Optional[OpenRouterKeyPool] = None
_pool_lock = threading.Lock()


def get_openrouter_key_pool(
    key_prefix: str = "OPENROUTER_API_KEY",
    daily_limit: int = 1000,
    minute_limit: int = 20
) -> OpenRouterKeyPool:
    """
    Get the global key-pool singleton.
    
    Args:
        key_prefix: Environment-variable prefix.
        daily_limit: Daily limit.
        minute_limit: Per-minute limit.
    
    Returns:
        OpenRouterKeyPool instance.
    """
    global _global_pool
    
    with _pool_lock:
        if _global_pool is None:
            _global_pool = OpenRouterKeyPool(
                key_prefix=key_prefix,
                daily_limit=daily_limit,
                minute_limit=minute_limit
            )
        return _global_pool


def reset_key_pool():
    """Reset the global key pool, mainly for tests."""
    global _global_pool
    with _pool_lock:
        _global_pool = None


def query_openrouter_credits(api_key: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Query the remaining credits for an OpenRouter account.
    
    Args:
        api_key: OpenRouter API Key
        timeout: Request timeout.
    
    Returns:
        {
            "success": True/False,
            "total_credits": total purchased credits,
            "total_usage": credits already used,
            "remaining": remaining credits,
            "error": error message if the query fails
        }
    """
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/credits",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        if "data" in data:
            total = data["data"].get("total_credits", 0)
            usage = data["data"].get("total_usage", 0)
            return {
                "success": True,
                "total_credits": total,
                "total_usage": usage,
                "remaining": total - usage
            }
        else:
            return {"success": False, "error": "Unexpected response format"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def query_all_keys_credits(pool: Optional[OpenRouterKeyPool] = None) -> List[Dict[str, Any]]:
    """
    Query the remaining credits for all keys in the pool.
    
    Args:
        pool: Key-pool instance. If None, the global pool is used.
    
    Returns:
        List of credit info entries, one per key.
    """
    if pool is None:
        pool = get_openrouter_key_pool()
    
    results = []
    for ks in pool.keys:
        credits_info = query_openrouter_credits(ks.key)
        results.append({
            "key_name": ks.key_name,
            "local_daily_used": ks.daily_count,
            "local_minute_used": ks.get_minute_count(),
            **credits_info
        })
    
    return results


def print_credits_summary(pool: Optional[OpenRouterKeyPool] = None):
    """Print a credit summary for all keys."""
    results = query_all_keys_credits(pool)
    
    print("\n" + "=" * 60)
    print("OpenRouter key-pool credit summary")
    print("=" * 60)
    
    total_remaining = 0
    for r in results:
        print(f"\n{r['key_name']}:")
        if r.get("success"):
            remaining = r.get("remaining", 0)
            total_remaining += remaining
            print(f"  Remaining credits: ${remaining:.4f}")
            print(f"  Used: ${r.get('total_usage', 0):.4f} / ${r.get('total_credits', 0):.4f}")
        else:
            print(f"  Query failed: {r.get('error', 'Unknown error')}")
        print(f"  Local stats: daily {r['local_daily_used']}/1000, minute {r['local_minute_used']}/20")
    
    print(f"\n{'=' * 60}")
    print(f"Total remaining credits: ${total_remaining:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    # Test code.
    logging.basicConfig(level=logging.INFO)
    
    pool = get_openrouter_key_pool()
    
    print("\n=== Key Pool Status ===")
    status = pool.get_status()
    print(f"Total keys: {status['total_keys']}")
    print(f"Available: {pool.get_available_count()}")
    
    for i, ks in enumerate(status['keys']):
        print(f"\n[{i}] {ks['key_name']}:")
        print(f"    Daily: {ks['daily_used']}")
        print(f"    Minute: {ks['minute_used']}")
        print(f"    Available: {ks['available']}")
    
    # Test key acquisition.
    print("\n=== Testing get_key ===")
    for i in range(5):
        key = pool.get_key()
        if key:
            print(f"Got key: {key[:20]}...")
            pool.record_success(key)
        else:
            print("No available key!")
