from typing import Callable, Any
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""

    pass


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, reset_timeout: int = 60):
        """
        Initialize the circuit breaker.

        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Time in seconds to wait before trying again
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_failure_time = 0

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker pattern.

        Args:
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function call fails
        """
        current_time = time.time()

        # Check if circuit is OPEN
        if self.state == "OPEN":
            # Check if reset timeout has elapsed
            if current_time - self.last_failure_time > self.reset_timeout:
                self.state = "HALF-OPEN"
                logger.info(f"Circuit {self.name} changed from OPEN to HALF-OPEN")
            else:
                logger.warning(f"Circuit {self.name} is OPEN. Skipping call.")
                raise CircuitBreakerError(f"Circuit {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)

            # If successful and in HALF-OPEN, close the circuit
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failures = 0
                logger.info(f"Circuit {self.name} changed from HALF-OPEN to CLOSED")

            return result

        except Exception as e:
            # Increment failure count
            self.failures += 1
            self.last_failure_time = current_time

            # Check if failure threshold reached
            if self.failures >= self.failure_threshold and self.state == "CLOSED":
                self.state = "OPEN"
                logger.warning(
                    f"Circuit {self.name} changed from CLOSED to OPEN after {self.failures} failures"
                )

            # Re-raise the exception
            raise e


def circuit_breaker(name: str, failure_threshold: int = 5, reset_timeout: int = 60):
    """
    Decorator version of the circuit breaker.

    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of failures before opening the circuit
        reset_timeout: Time in seconds to wait before trying again

    Returns:
        Decorator function
    """
    cb = CircuitBreaker(name, failure_threshold, reset_timeout)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.execute(func, *args, **kwargs)

        return wrapper

    return decorator
