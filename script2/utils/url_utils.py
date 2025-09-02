"""Utilities for URL handling and generation."""

# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import re
import uuid
from typing import Optional, Set
from pathlib import Path
from article_generator.logger import logger

# Keep track of used URLs to handle duplicates
_used_urls: Set[str] = set()

def extract_keyword_for_url(title: str, keyword: str) -> str:
    """
    Extract the long-tail keyword from the title or use the keyword itself.
    Returns a URL-friendly version.
    """
    # Use the keyword if it's shorter than the title
    text_to_use = keyword if len(keyword) < len(title) else title
    
    # Convert to lowercase and remove special characters
    url_text = text_to_use.lower()
    url_text = re.sub(r'[^a-z0-9\s-]', '', url_text)
    
    # Replace spaces with hyphens and remove multiple hyphens
    url_text = re.sub(r'\s+', '-', url_text)
    url_text = re.sub(r'-+', '-', url_text)
    
    # Remove leading/trailing hyphens
    url_text = url_text.strip('-')
    
    return url_text

def handle_duplicate_url(base_url: str, handling_method: str = "increment") -> str:
    """
    Handle duplicate URLs based on the specified method.
    
    Args:
        base_url: The original URL to check for duplicates
        handling_method: Either "increment" (default) or "uuid"
    
    Returns:
        A unique URL that hasn't been used before
    """
    if base_url not in _used_urls:
        _used_urls.add(base_url)
        return base_url
        
    if handling_method == "uuid":
        # Add a short UUID suffix
        unique_url = f"{base_url}-{str(uuid.uuid4())[:8]}"
        _used_urls.add(unique_url)
        return unique_url
    else:  # increment method
        counter = 2
        while True:
            new_url = f"{base_url}-{counter}"
            if new_url not in _used_urls:
                _used_urls.add(new_url)
                return new_url
            counter += 1

def generate_post_url(title: str, keyword: str, use_keyword: bool = True, handling_method: str = "increment") -> str:
    """
    Generate a unique URL for a blog post.
    
    Args:
        title: The full article title
        keyword: The main keyword for the article
        use_keyword: Whether to use the keyword instead of full title (default: True)
        handling_method: How to handle duplicates - "increment" or "uuid" (default: "increment")
    
    Returns:
        A unique, URL-friendly string suitable for use as a WordPress post URL
    """
    try:
        # Extract the base URL text
        base_url = extract_keyword_for_url(title, keyword) if use_keyword else extract_keyword_for_url(title, title)
        
        # Handle any duplicates
        final_url = handle_duplicate_url(base_url, handling_method)
        
        logger.debug(f"Generated URL '{final_url}' from title '{title}' using keyword '{keyword}'")
        return final_url
        
    except Exception as e:
        logger.error(f"Error generating URL from title '{title}': {str(e)}")
        # Fallback to a safe but unique URL
        fallback_url = f"post-{str(uuid.uuid4())[:8]}"
        _used_urls.add(fallback_url)
        return fallback_url

def clear_url_cache() -> None:
    """Clear the cache of used URLs. Useful for testing or resetting state."""
    _used_urls.clear()