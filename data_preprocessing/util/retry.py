"""Retry helpers with exponential backoff for network-heavy scripts."""

import time
import logging
import functools
from typing import Callable, TypeVar, Any, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 2
DEFAULT_MAX_DELAY = 30
DEFAULT_TIMEOUT = 30

T = TypeVar('T')


def exponential_backoff(attempt: int, base_delay: float = DEFAULT_BASE_DELAY, max_delay: float = DEFAULT_MAX_DELAY) -> float:
    """Return the sleep duration for one retry attempt."""
    delay = min(base_delay ** attempt, max_delay)
    return delay


def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exceptions: Tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    on_failure: Optional[Callable[[Exception], None]] = None,
) -> Callable[..., Optional[T]]:
    """Wrap a callable with retry and backoff behavior."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Optional[T]:
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = exponential_backoff(attempt, base_delay, max_delay)
                    if on_retry:
                        on_retry(attempt, e)
                    else:
                        logger.warning(f"Retry {attempt}/{max_retries}: {e} (sleep {wait_time}s)")
                    time.sleep(wait_time)
                continue
        
        if on_failure and last_error is not None:
            on_failure(last_error)
        elif last_error is not None:
            logger.error(f"Operation failed after {max_retries} retries: {last_error}")
        return None
    
    return wrapper


def retry_decorator(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exceptions: Tuple = (Exception,),
    return_on_failure: Any = None,
):
    """Decorator form of ``retry_with_backoff``."""
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            last_error = None
            func_name = func.__name__
            
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        wait_time = exponential_backoff(attempt, base_delay, max_delay)
                        logger.warning(f"{func_name} retry {attempt}/{max_retries}: {e} (sleep {wait_time}s)")
                        time.sleep(wait_time)
                    continue
            
            logger.error(f"{func_name} failed after {max_retries} retries: {last_error}")
            return return_on_failure
        
        return wrapper
    return decorator


class RetryableRequest:
    """Small HTTP client wrapper with retry support."""
    
    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        rate_limit_handler: Optional[Callable[[dict], None]] = None,
        headers_getter: Optional[Callable[[], dict]] = None,
    ):
        """Initialize retry settings and optional rate-limit hooks."""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.rate_limit_handler = rate_limit_handler
        self.headers_getter = headers_getter
    
    def request(
        self,
        method: str,
        url: str,
        headers: Optional[dict] = None,
        params: Optional[dict] = None,
        **kwargs
    ):
        """Send an HTTP request with retries."""
        import requests
        
        last_error = None
        kwargs.setdefault('timeout', self.timeout)
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.request(method, url, headers=headers, params=params, **kwargs)
                
                if response.status_code == 403 and self.rate_limit_handler:
                    self.rate_limit_handler(dict(response.headers))
                    if self.headers_getter:
                        headers = self.headers_getter()
                    response = requests.request(method, url, headers=headers, params=params, **kwargs)
                
                if response.status_code in (404, 409):
                    return response
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = exponential_backoff(attempt, self.base_delay, self.max_delay)
                    logger.warning(f"Request retry {attempt}/{self.max_retries} [{method} {url}]: {e} (sleep {wait_time}s)")
                    time.sleep(wait_time)
                continue
        
        logger.error(f"Request failed [{method} {url}] after {self.max_retries} retries: {last_error}")
        return None
    
    def get(self, url: str, **kwargs):
        """Send a GET request."""
        return self.request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs):
        """Send a POST request."""
        return self.request('POST', url, **kwargs)
