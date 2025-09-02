# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import time
import threading
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
import logging

# Set up logger
logger = logging.getLogger("rate_limiter")

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    # Requests per minute
    rpm: int = 60
    # Requests per day
    rpd: Optional[int] = None
    # Cooldown period in seconds after hitting rate limit
    cooldown_period: int = 60
    # Whether to enable rate limiting
    enabled: bool = True


class RateLimiter:
    """
    Rate limiter for API calls.
    
    This class implements a token bucket algorithm for rate limiting.
    It supports both per-minute and per-day rate limits.
    """
    
    def __init__(self, name: str, config: RateLimitConfig):
        """
        Initialize the rate limiter.
        
        Args:
            name: Name of the rate limiter (for logging)
            config: Rate limit configuration
        """
        self.name = name
        self.config = config
        
        # Initialize token buckets
        self.minute_tokens = config.rpm
        self.day_tokens = config.rpd if config.rpd is not None else float('inf')
        
        # Initialize timestamps
        self.last_minute_refill = time.time()
        self.last_day_refill = time.time()
        
        # Initialize lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize cooldown state
        self.in_cooldown = False
        self.cooldown_until = 0
        
        # Initialize counters
        self.total_requests = 0
        self.rate_limited_requests = 0
        
        logger.info(f"Rate limiter '{name}' initialized with {config.rpm} RPM and {config.rpd or 'unlimited'} RPD")
    
    def _refill_tokens(self) -> None:
        """Refill token buckets based on elapsed time."""
        current_time = time.time()
        
        # Refill minute tokens
        minutes_elapsed = (current_time - self.last_minute_refill) / 60
        if minutes_elapsed >= 1:
            self.minute_tokens = min(self.config.rpm, self.minute_tokens + int(minutes_elapsed * self.config.rpm))
            self.last_minute_refill = current_time
        
        # Refill day tokens
        if self.config.rpd is not None:
            days_elapsed = (current_time - self.last_day_refill) / (24 * 60 * 60)
            if days_elapsed >= 1:
                self.day_tokens = self.config.rpd
                self.last_day_refill = current_time
    
    def _check_cooldown(self) -> bool:
        """Check if the rate limiter is in cooldown mode."""
        if not self.in_cooldown:
            return False
        
        if time.time() >= self.cooldown_until:
            self.in_cooldown = False
            logger.info(f"Rate limiter '{self.name}' cooldown period ended")
            return False
        
        return True
    
    def _enter_cooldown(self) -> None:
        """Enter cooldown mode after hitting rate limit."""
        self.in_cooldown = True
        self.cooldown_until = time.time() + self.config.cooldown_period
        logger.warning(
            f"Rate limiter '{self.name}' entering cooldown for {self.config.cooldown_period} seconds"
        )
    
    def acquire(self) -> bool:
        """
        Acquire a token from the rate limiter.
        
        Returns:
            bool: True if a token was acquired, False otherwise
        """
        if not self.config.enabled:
            return True
        
        with self.lock:
            self.total_requests += 1
            
            # Check if in cooldown
            if self._check_cooldown():
                self.rate_limited_requests += 1
                return False
            
            # Refill tokens
            self._refill_tokens()
            
            # Check if we have tokens available
            if self.minute_tokens <= 0 or self.day_tokens <= 0:
                self._enter_cooldown()
                self.rate_limited_requests += 1
                return False
            
            # Consume tokens
            self.minute_tokens -= 1
            if self.config.rpd is not None:
                self.day_tokens -= 1
            
            return True
    
    def wait_until_ready(self, max_wait: Optional[float] = None) -> bool:
        """
        Wait until a token is available or max_wait is reached.
        
        Args:
            max_wait: Maximum time to wait in seconds
        
        Returns:
            bool: True if a token was acquired, False if max_wait was reached
        """
        if not self.config.enabled:
            return True
        
        start_time = time.time()
        
        while True:
            if self.acquire():
                return True
            
            if max_wait is not None and time.time() - start_time >= max_wait:
                return False
            
            # Wait for a short time before trying again
            time.sleep(60)
    
    def execute_with_rate_limit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with rate limiting.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        
        Returns:
            Any: Result of the function
        
        Raises:
            Exception: If the function raises an exception
        """
        if not self.config.enabled:
            return func(*args, **kwargs)
        
        # Wait until we can acquire a token
        if not self.wait_until_ready():
            raise Exception(f"Rate limit exceeded for '{self.name}'")
        
        # Execute the function
        return func(*args, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the rate limiter."""
        with self.lock:
            return {
                "name": self.name,
                "total_requests": self.total_requests,
                "rate_limited_requests": self.rate_limited_requests,
                "minute_tokens": self.minute_tokens,
                "day_tokens": self.day_tokens,
                "in_cooldown": self.in_cooldown,
                "cooldown_until": self.cooldown_until if self.in_cooldown else 0,
            }


# Global rate limiters
openai_rate_limiter = None
serpapi_rate_limiter = None
duck_duck_go_rate_limiter = None
unsplash_rate_limiter = None
youtube_rate_limiter = None
openverse_rate_limiter = None
pexels_rate_limiter = None
pixabay_rate_limiter = None
hugging_face_rate_limiter = None


def initialize_rate_limiters(
    openai_config: Optional[RateLimitConfig] = None,
    serpapi_config: Optional[RateLimitConfig] = None,
    duckduckgo_config: Optional[RateLimitConfig] = None,
    unsplash_config: Optional[RateLimitConfig] = None,
    youtube_config: Optional[RateLimitConfig] = None,
    openverse_config: Optional[RateLimitConfig] = None,
    pexels_config: Optional[RateLimitConfig] = None,
    pixabay_config: Optional[RateLimitConfig] = None,
    huggingface_config: Optional[RateLimitConfig] = None
) -> None:
    """
    Initialize global rate limiters.
    
    Args:
        openai_config: OpenAI rate limit configuration
        serpapi_config: SerpAPI rate limit configuration
        duckduckgo_config: Duckduckgo rate limit configuration
        unsplash_config: Unsplash rate limit configuration
        youtube_config: YouTube rate limit configuration
        openverse_config: Openverse rate limit configuration
        pexels_config: Pexels rate limit configuration
        pixabay_config: Pixabay rate limit configuration
        huggingfacee_config: Pixabay rate limit configuration
    """
    global openai_rate_limiter, serpapi_rate_limiter,duck_duck_go_rate_limiter,unsplash_rate_limiter, youtube_rate_limiter,openverse_rate_limiter,pexels_rate_limiter,pixabay_rate_limiter,hugging_face_rate_limiter
    
    # Initialize OpenAI/OpenRouter rate limiter
    if openai_config is None:
        openai_config = RateLimitConfig(
            rpm=20,  # OpenRouter free tier limit of 20 requests per minute
            rpd=10000,  # Conservative daily limit
            cooldown_period=65  # Wait period for free tier (slightly over 1 minute)
        )
            
    # Initialize SerpAPI rate limiter
    if serpapi_config is None:
        serpapi_config = RateLimitConfig(rpm=5, rpd=100)  # Default SerpAPI limits
    serpapi_rate_limiter = RateLimiter("serpapi", serpapi_config)

    # Initialize duckduckgo rate limiter
    if duckduckgo_config is None:
        duckduckgo_config = RateLimitConfig(rpm=30, rpd=10000)  # Default duckduckgo limits
    duck_duck_go_rate_limiter = RateLimiter("duckduckgo", duckduckgo_config)
    
    # Initialize Unsplash rate limiter
    if unsplash_config is None:
        unsplash_config = RateLimitConfig(rpm=50, rpd=5000)  # Default Unsplash limits
    unsplash_rate_limiter = RateLimiter("unsplash", unsplash_config)
    
    # Initialize YouTube rate limiter
    if youtube_config is None:
        youtube_config = RateLimitConfig(rpm=100, rpd=10000)  # Default YouTube limits
    youtube_rate_limiter = RateLimiter("youtube", youtube_config) 
    
    # Initialize Openverse rate limiter
    if openverse_config is None:
        openverse_config = RateLimitConfig(rpm=60,rpd=5000)
    openverse_rate_limiter = RateLimiter("openverse",openverse_config)
    
    # Initialize Pexels rate limiter
    if pexels_config is None:
        pexels_config = RateLimitConfig(rpm=200,rpd=20000)
    pexels_rate_limiter = RateLimiter("pexels",pexels_config)
    
    # Initialize Pixabay rate limiter
    if pixabay_config is None:
        pixabay_config = RateLimitConfig(rpm=60,rpd=5000)
    pixabay_rate_limiter = RateLimiter("pixabay",pixabay_config)
    
    # Initialize huggingface rate limiter
    if huggingface_config is None:
        huggingface_config = RateLimitConfig(rpm=60,rpd=5000)
    hugging_face_rate_limiter = RateLimiter("huggingface",huggingface_config)