# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import requests
from serpapi import GoogleSearch
import logging
from typing import Tuple, Dict, Optional
from utils.error_utils import ErrorHandler
import time
from ddgs import DDGS



# Initialize error handler
error_handler = ErrorHandler()

def validate_openai_api_key(api_key: str) -> bool:
    """
    Validates the OpenAI API key by making a request to the models endpoint.
    
    Args:
        api_key (str): OpenAI API key to validate
    Returns:
        bool: True if valid, False otherwise
    """
    TEST_URL = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(TEST_URL, headers=headers)
        if response.status_code == 200:
            error_handler.handle_error(Exception("OpenAI API Key is VALID!"), severity="info")
            return True
        elif response.status_code == 401:
            error_handler.handle_error(Exception("Invalid OpenAI API Key!"), severity="error")
        else:
            error_handler.handle_error(Exception(f"Unexpected error: {response.status_code} - {response.text}"), severity="error")
    except requests.exceptions.RequestException as e:
        error_handler.handle_error(e, severity="error")
    return False

def validate_youtube_api_key(api_key: str) -> bool:
    """
    Validates the YouTube API key by making a test request.
    
    Args:
        api_key (str): YouTube API key to validate
    Returns:
        bool: True if valid, False otherwise
    """
    url = f"https://www.googleapis.com/youtube/v3/videos?id=Ks-_Mh1QhMc&key={api_key}&part=id"
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            if "error" in data:
                error_handler.handle_error(Exception("API Key is valid, but quota exceeded!"), severity="warning")
                return False
            error_handler.handle_error(Exception("Youtube API Key is valid!"), severity="info")
            return True
        elif response.status_code == 403:
            error_handler.handle_error(Exception("Youtube Invalid API Key or quota exceeded!"), severity="error")
        elif response.status_code == 400:
            error_handler.handle_error(Exception("Youtube Bad request. Check if API key is correct!"), severity="error")
        else:
            error_handler.handle_error(Exception(f"Youtube Unexpected error: {response.status_code}"), severity="error")
    except requests.exceptions.RequestException as e:
        error_handler.handle_error(e, severity="error")
    return False



def validate_duckduckgo_access(query="OpenAI", region="wt-wt", max_results=1, max_retries=5, initial_delay=10):
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

def validate_serpapi_key(serp_api_key: str) -> tuple:
    """Validates the SerpAPI key by making a test request."""
    if not serp_api_key:
        error_handler.handle_error(Exception("SerpAPI key is missing!"), severity="error")
        return False, {}
    
    try:
        # Define the parameters for a simple search
        params = {
            "api_key": serp_api_key,
            "engine": "google",
            "q": "OpenAI",  # Simple search query
            "num": 1,       # Only need one result for validation
            "safe": "active"
        }
        
        # Use the SerpAPI Python package to perform the search
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Check if search was successful and has key info
        if "error" in results:
            error_handler.handle_error(Exception(f"SerpAPI Error: {results['error']}"), severity="error")
            return False, {}
        
        # Get remaining searches if available
        if isinstance(results, dict) and "search_metadata" in results:
            metadata = results["search_metadata"]
            
            # Create a dictionary with key info
            key_info = {
                "valid": True,
                "type": metadata.get("plan", "unknown"),
                "remaining": metadata.get("remaining_searches", "unknown"),
                "total_quota": metadata.get("total_searches", "unknown"),
                "status": metadata.get("status", "unknown")
            }
            
            if key_info["remaining"] == 0:
                # Removed emojis from message
                error_handler.handle_error(Exception("SerpAPI Key is valid but quota exhausted!"), severity="warning")
                return True, key_info
            else:
                # Removed emojis from message
                error_handler.handle_error(Exception(f"SerpAPI Key is valid! ({key_info['remaining']} searches remaining)"), severity="info")
                return True, key_info
        
        # If we couldn't verify but no error, assume valid
        return True, {"valid": True}
    
    except Exception as e:
        error_handler.handle_error(e, severity="error")
        return False, {}

def validate_wordpress_api(json_url: str, headers: dict, status: str, categories: str, author: str) -> bool:
    """
    Validates WordPress API connection and credentials.
    
    Args:
        json_url (str): WordPress API URL
        headers (dict): Request headers with authentication
        status (str): Post status (draft/publish)
        categories (str): Category ID
        author (str): Author ID
    Returns:
        bool: True if valid, False otherwise
    """
    error_handler.handle_error(Exception("===== WordPress Connection Start ====="), severity="info")
    logging.info("Checking Your WordPress API Connection And Credentials")
    
    # Validate user existence
    logging.info("Validating user...")
    user_response = requests.get(f"{json_url}/users/{author}", headers=headers)
    if user_response.status_code == 404:
        error_handler.handle_error(Exception("User does not exist. Please check the author ID."), severity="error")
        return False
    logging.info("User exists!")

    # Create test post
    post = {
        'title': "API Connection Test",
        'slug': "test",
        'status': status,
        'content': "Test post body",
        'categories': categories,
        'author': author,
        'format': 'standard',
    }

    post_response = requests.post(f"{json_url}/posts", headers=headers, json=post)
    if post_response.status_code in [200, 201]:
        logging.info("API Connection Successful!")
    else:
        error_handler.handle_error(Exception(f"API Connection Failed: {post_response.text}"), severity="error")
        return False

    # Validate category
    category_response = requests.get(f"{json_url}/categories", headers=headers)
    try:
        categories_data = category_response.json()
        category_exists = any(cat['id'] == int(categories) for cat in categories_data)
        if not category_exists:
            error_handler.handle_error(Exception("Category does not exist. Please check the category ID."), severity="error")
            return False
        logging.info("Category exists!")
    except Exception as e:
        error_handler.handle_error(e, severity="error")
        return False

    error_handler.handle_error(Exception("===== WordPress Connection Done ====="), severity="info")
    return True 