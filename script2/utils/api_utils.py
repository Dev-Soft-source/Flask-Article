# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import time
import random
import requests
from typing import Optional, Dict, Any, List
from openai import OpenAI
from serpapi import GoogleSearch
from config import Config
from utils.rate_limiter import serpapi_rate_limiter
from utils.error_utils import ErrorHandler, format_error_message
from article_generator.logger import logger
from ddgs import DDGS


# Global error handler
error_handler = ErrorHandler(show_traceback=True)

class APIValidator:
    """Validates various API keys and their functionality."""
    
    # @staticmethod
    # def validate_duckduckgo_access() -> tuple:
    #     """
    #     Validates that DuckDuckGo search is reachable by performing a simple search.

    #     Returns:
    #         Tuple[bool, dict]: (True, info) if successful, (False, {}) otherwise.
    #     """
    #     try:
    #         with DDGS() as ddgs:
    #             # Perform a lightweight query
    #             time.sleep(10)
    #             results = ddgs.text("OpenAI", region="wt-wt", max_results=1)

    #             # Check if we received any result
    #             if results:
    #                 info = {
    #                     "valid": True,
    #                     "status": "reachable",
    #                     "example_result": next(iter(results), {})
    #                 }
    #                 error_handler.handle_error(
    #                     Exception("DuckDuckGo is reachable and responded to query."),
    #                     severity="info"
    #                 )
    #                 return True, info
    #             else:
    #                 error_handler.handle_error(
    #                     Exception("DuckDuckGo returned no results for a basic query."),
    #                     severity="warning"
    #                 )
    #                 return False, {}

    #     except Exception as e:
    #         error_handler.handle_error(e, severity="error")
    #         return False, {}
        
        
    @staticmethod   
    def validate_duckduckgo_access(query="OpenAI", region="wt-wt", max_results=1, max_retries=5, initial_delay=10) -> tuple:
        """
        Validates access to DuckDuckGo search with retry mechanism for rate limits.
        
        Args:
            query (str): Search query
            region (str): Region for search
            max_results (int): Maximum number of results
            max_retries (int): Maximum retry attempts for rate limit
            initial_delay (int): Initial delay in seconds for retries
        
        Returns:
            list: Search results or empty list if failed
        """
        ddgs = DDGS()
        attempt = 0
        
        while attempt < max_retries:
            try:
                results = ddgs.text(query, region=region, max_results=max_results)
                # Check if we received any result
                error_handler.handle_error(
                        Exception(results),
                        severity="info"
                )
                if results:
                    info = {
                        "valid": True,
                        "status": "reachable",
                        "example_result": next(iter(results), {})
                    }
                    error_handler.handle_error(
                        Exception("DuckDuckGo is reachable and responded to query."),
                        severity="info"
                    )
                    return True, info
                else:
                    error_handler.handle_error(
                        Exception("DuckDuckGo returned no results for a basic query."),
                        severity="warning"
                    )
                    return False, {}
            
            except Exception as e:
                if "Ratelimit" in str(e):
                    attempt += 1
                    if attempt == max_retries:
                        error_handler.handle_error(
                            Exception(f"Max retries ({max_retries}) reached for rate limit: {e}"),
                            severity="warning"
                        )
                        return False, {}
                    
                    # Exponential backoff: delay = initial_delay * 2^(attempt-1)
                    delay = initial_delay * (2 ** (attempt - 1))
                    # error_handler.handle_error(
                    #     Exception(f"Rate limit hit: {e}. Retrying after {delay} seconds (Attempt {attempt}/{max_retries})"),
                    #     severity="warning"
                    # )                
                    time.sleep(delay)
                
                else:
                    error_handler.handle_error(
                        Exception(f"DuckDuckGo search error: {e}"),
                        severity="error"
                    )
                    return False, {}
            
            except Exception as e:
                error_handler.handle_error(
                        Exception(f"Unexpected error in DuckDuckGo search: {e}"),
                        severity="error"
                )
                return False, {}
            
            finally:
                # Add a small delay between requests to prevent rapid calls
                time.sleep(0.5)

        return False, {}


    @staticmethod
    def validate_openai_key(key: str) -> bool:
        """Validate OpenAI API key by checking models endpoint."""
        if not key:
            error_handler.handle_error(
                Exception("OpenAI key is missing"),
                context={"component": "APIValidator", "validation": "OpenAI API"},
                severity="warning"
            )
            return False

        try:
            client = OpenAI(api_key=key)
            client.models.list()
            logger.success("OpenAI API Key is valid!")
            return True
        except Exception as e:
            error_handler.handle_error(
                e,
                context={"component": "APIValidator", "validation": "OpenAI API"},
                severity="error"
            )
            return False

    @staticmethod
    def validate_youtube_api_key(key: str) -> bool:
        """Validate YouTube API key by checking API response."""
        if not key:
            error_handler.handle_error(
                Exception("YouTube API key is missing"),
                context={"component": "APIValidator", "validation": "YouTube API"},
                severity="warning"
            )
            return False

        try:
            response = requests.get(
                f"https://www.googleapis.com/youtube/v3/search?part=snippet&q=test&key={key}&maxResults=1"
            )

            if response.status_code == 200:
                logger.success("YouTube API Key is valid!")
                return True
            else:
                error_handler.handle_error(
                    Exception(f"YouTube API Key validation failed: HTTP {response.status_code}"),
                    context={"component": "APIValidator", "validation": "YouTube API", "status_code": response.status_code},
                    severity="warning"
                )
                return False
        except Exception as e:
            error_handler.handle_error(
                e,
                context={"component": "APIValidator", "validation": "YouTube API"},
                severity="error"
            )
            return False

    @staticmethod
    def validate_serp_api_key(key: str) -> bool:
        """Validate SerpAPI key by checking account info."""
        if not key:
            error_handler.handle_error(
                Exception("SerpAPI key is missing"),
                context={"component": "APIValidator", "validation": "SerpAPI"},
                severity="warning"
            )
            return False

        try:
            params = {
                "api_key": key,
                "engine": "google",
                "q": "test query",
                "num": 1
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            if "error" in results:
                error_message = results["error"]

                # Check if the error is specifically about quota being exhausted
                if "account has run out of searches" in error_message:
                    error_handler.handle_error(
                        Exception("Your account has run out of searches. The system will continue with RAG and PAA features disabled."),
                        context={"component": "APIValidator", "validation": "SerpAPI", "quota_exhausted": True},
                        severity="warning"
                    )
                else:
                    error_handler.handle_error(
                        Exception(f"SerpAPI Error: {error_message}"),
                        context={"component": "APIValidator", "validation": "SerpAPI"},
                        severity="warning"
                    )

                return False

            # Key is valid and has available quota
            logger.success("SerpAPI Key is valid!")
            return True
        except Exception as e:
            error_handler.handle_error(
                e,
                context={"component": "APIValidator", "validation": "SerpAPI"},
                severity="error"
            )
            return False

class RetryHandler:
    """Handles retrying failed API calls with exponential backoff."""

    def __init__(self, config: Config):
        """Initialize with retry configuration."""
        self.config = config

    def execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic."""
        delay = self.config.initial_delay

        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise e

                # Calculate next delay with exponential backoff
                jitter = random.uniform(0.8, 1.2) if self.config.jitter else 1
                delay *= self.config.exponential_base * jitter

                logger.warning(f"Attempt {attempt + 1} failed. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

    def retry_openai_completion(self, client, messages, max_tokens=2048, temperature=1.0,
                               top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, seed=None):
        """Execute OpenAI completion with retry logic."""
        # Set up the parameters for the API call
        params = {
            "model": self.config.openai_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        # Add seed if provided and seed control is enabled
        if seed is not None and self.config.enable_seed_control:
            params["seed"] = seed

        # Define the function to be retried
        def _make_api_call():
            return client.chat.completions.create(**params)

        # Execute with retry
        return self.execute_with_retry(_make_api_call)

class SerpAPI:
    """Handles SerpAPI interactions."""

    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.retry_handler = RetryHandler(config)

    def perform_search(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Perform a search using SerpAPI.

        Args:
            query (str): Search query
        Returns:
            Optional[Dict[str, Any]]: Search results or None if failed
        """
        if not self.config.serp_api_key:
            error_handler.handle_error(
                Exception("No SerpAPI key available"),
                context={"component": "SerpAPI", "method": "perform_search", "query": query},
                severity="warning"
            )
            return None

        try:
            # Set up parameters
            params = {
                "api_key": self.config.serp_api_key,
                "engine": "google",
                "q": query,
                "gl": "us",
                "hl": "en",
                "num": 10
            }

            # Add safe search if enabled
            # if self.config.safe_search:
            #     params["safe"] = "active"

            # Make the API request
            search = GoogleSearch(params)
            results = search.get_dict()

            # Check for errors
            if "error" in results:
                error_handler.handle_error(
                    Exception(f"SerpAPI error: {results['error']}"),
                    context={"component": "SerpAPI", "method": "perform_search", "query": query},
                    severity="error"
                )
                return None

            return results
        except Exception as e:
            error_handler.handle_error(
                e,
                context={"component": "SerpAPI", "method": "perform_search", "query": query},
                severity="error"
            )
            return None