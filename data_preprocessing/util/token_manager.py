"""Helpers for discovering and rotating GitHub tokens."""

import os
import time
import logging
import threading
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


def detect_github_tokens() -> Tuple[List[str], List[str]]:
    """Load configured GitHub tokens from the environment."""
    tokens = []
    token_names = []
    
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        import config
        
        if hasattr(config, 'GITHUB_TOKEN_ENV_VARS'):
            token_vars = config.GITHUB_TOKEN_ENV_VARS
        else:
            logger.warning("⚠️ config.py does not define 'GITHUB_TOKEN_ENV_VARS'")
            return [], []
        
        for token_name in token_vars:
            token = os.getenv(token_name)
            if token and token not in tokens:
                tokens.append(token)
                token_names.append(token_name)
        
        return tokens, token_names
    except ImportError:
        logger.warning("⚠️ Failed to import config")
        return [], []


def detect_github_token_vars() -> List[str]:
    """Return the names of available GitHub token environment variables."""
    _, token_names = detect_github_tokens()
    return token_names


class TokenRotator:
    """Thread-safe GitHub token rotator."""
    
    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        token_names: Optional[List[str]] = None,
        requests_per_token: int = 100,
        auto_detect: bool = True,
        use_pygithub: bool = False
    ):
        """Initialize tokens, stats, and optional PyGithub clients."""
        if tokens is None and auto_detect:
            tokens, token_names = detect_github_tokens()
        
        if not tokens:
            raise ValueError("No GitHub tokens available")
        
        self.tokens = tokens
        self.token_names = token_names or [f"TOKEN_{i}" for i in range(len(tokens))]
        self.current_index = 0
        self.lock = threading.Lock()
        
        self.token_error_count: Dict[int, int] = {i: 0 for i in range(len(tokens))}
        self.request_count: Dict[int, int] = {i: 0 for i in range(len(tokens))}
        self.current_token_requests = 0
        self.requests_per_token = requests_per_token
        self.total_rotations = 0
        
        self.clients = None
        if use_pygithub:
            try:
                from github import Github
                self.clients = [Github(token, retry=None, timeout=15) for token in tokens]
            except ImportError:
                logger.warning("⚠️ PyGithub is not installed; client creation is skipped")
        
        logger.info(f"🔄 Token rotator initialized with {len(tokens)} token(s)")
        logger.info(f"⚙️ Auto-rotate after {requests_per_token} request(s) per token")
    
    def get_token(self) -> Tuple[str, int]:
        """Return the current token and its index."""
        with self.lock:
            if self.current_token_requests >= self.requests_per_token and len(self.tokens) > 1:
                self.current_index = (self.current_index + 1) % len(self.tokens)
                self.current_token_requests = 0
                self.total_rotations += 1
            
            index = self.current_index
            self.request_count[index] = self.request_count.get(index, 0) + 1
            self.current_token_requests += 1
            return self.tokens[index], index
    
    def get_token_name(self) -> Tuple[str, int]:
        """Return the current token variable name and its index."""
        with self.lock:
            if self.current_token_requests >= self.requests_per_token and len(self.tokens) > 1:
                self.current_index = (self.current_index + 1) % len(self.tokens)
                self.current_token_requests = 0
                self.total_rotations += 1
            
            index = self.current_index
            self.request_count[index] = self.request_count.get(index, 0) + 1
            self.current_token_requests += 1
            return self.token_names[index], index
    
    def get_client(self):
        """Return the current PyGithub client and its index."""
        if self.clients is None:
            raise RuntimeError("PyGithub clients are not initialized; use use_pygithub=True")
        
        with self.lock:
            if self.current_token_requests >= self.requests_per_token and len(self.tokens) > 1:
                self.current_index = (self.current_index + 1) % len(self.tokens)
                self.current_token_requests = 0
                self.total_rotations += 1
            
            index = self.current_index
            self.request_count[index] = self.request_count.get(index, 0) + 1
            self.current_token_requests += 1
            return self.clients[index], index
    
    def get_headers(self) -> Dict[str, str]:
        """Build request headers for the current token."""
        token, _ = self.get_token()
        return {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {token}"
        }
    
    def rotate(self, reason: str = "manual"):
        """Rotate to the next token and return it."""
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.tokens)
            self.current_token_requests = 0
            self.total_rotations += 1
            return self.tokens[self.current_index], self.current_index
    
    def get_current_token_name(self) -> str:
        """Return the current token variable name."""
        with self.lock:
            return self.token_names[self.current_index]
    
    def report_error(self, token_index: int):
        """Record an error against one token."""
        with self.lock:
            self.token_error_count[token_index] = self.token_error_count.get(token_index, 0) + 1
            logger.warning(f"⚠️ Token {self.token_names[token_index]} error #{self.token_error_count[token_index]}")
    
    def get_stats(self) -> List[Dict[str, Any]]:
        """Return request and error stats for all tokens."""
        with self.lock:
            stats = []
            for i in range(len(self.tokens)):
                stats.append({
                    'name': self.token_names[i],
                    'requests': self.request_count.get(i, 0),
                    'errors': self.token_error_count.get(i, 0)
                })
            return stats
    
    def log_stats(self):
        """Log token usage statistics."""
        logger.info("=" * 60)
        logger.info("📈 Token usage stats:")
        for stat in self.get_stats():
            logger.info(f"   {stat['name']}: {stat['requests']} requests, {stat['errors']} errors")
        logger.info(f"   Total rotations: {self.total_rotations}")
        logger.info("=" * 60)
    
    def __len__(self):
        return len(self.tokens)


def calculate_sleep_time(num_tokens: int) -> float:
    """Estimate a safe per-request delay from token count."""
    requests_per_hour_per_token = 4000
    total_requests_per_hour = requests_per_hour_per_token * num_tokens
    sleep_time = 3600 / total_requests_per_hour
    return max(0.1, sleep_time)
