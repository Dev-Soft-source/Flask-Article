# Custom exception for rate limit errors
class RateLimitError(Exception):
    """Exception raised when API rate limits are hit."""
    
    def __init__(self, message, is_minute_limit=False, retry_after=None):
        """Initialize the exception.
        
        Args:
            message: Error message
            is_minute_limit: Whether this is a minute-based rate limit (e.g., free-models-per-min)
            retry_after: Suggested time to wait before retrying (in seconds)
        """
        super().__init__(message)
        self.is_minute_limit = is_minute_limit
        self.retry_after = retry_after or 60  # Default to 60 seconds for free tier
