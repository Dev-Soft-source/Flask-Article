#!/usr/bin/env python3
# Test script for URL exclusion in RAG retriever

from article_generator.rag_retriever import WebContentRetriever
from article_generator.logger import logger

def test_url_exclusion():
    """Test the URL exclusion logic in the WebContentRetriever."""
    print("Starting URL exclusion test...")
    
    # Initialize retriever with a dummy API key
    retriever = WebContentRetriever("dummy_key")
    
    # Test URLs
    test_urls = [
        # Valid URLs
        ("https://example.com", True),
        ("https://www.nytimes.com/article", True),
        ("https://www.sciencedirect.com/article/123", True),
        
        # Wikipedia and related
        ("https://en.wikipedia.org/wiki/Python", False),
        ("https://wikipedia.org/wiki/Test", False),
        ("https://en.m.wikipedia.org/wiki/Coding", False),
        ("https://wikimedia.org/something", False),
        
        # Social media
        ("https://www.youtube.com/watch?v=12345", False),
        ("https://www.facebook.com/page", False),
        ("https://twitter.com/user", False),
        ("https://www.instagram.com/user", False),
        ("https://www.tiktok.com/@user", False),
        ("https://www.pinterest.com/pin/123", False),
        ("https://www.reddit.com/r/python", False),
        
        # Google services
        ("https://maps.google.com/location", False),
        ("https://www.google.com/shopping/product", False),
        
        # Other excluded domains
        ("https://example.com/login", False),
        ("https://example.com/signin", False),
        ("https://example.com/download", False),
        ("https://example.com/file.pdf", False),
        
        # Invalid URL format
        ("example.com", False),
        ("not-a-url", False)
    ]
    
    results = []
    for url, expected in test_urls:
        is_valid = retriever._is_valid_url(url)
        results.append((url, is_valid, expected, is_valid == expected))
    
    # Print results
    print("\nURL Exclusion Test Results:")
    print("-" * 90)
    print(f"{'URL':<50} | {'Result':<7} | {'Expected':<8} | {'Pass/Fail':<9}")
    print("-" * 90)
    
    failed_tests = 0
    for url, result, expected, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            failed_tests += 1
        print(f"{url:<50} | {str(result):<7} | {str(expected):<8} | {status:<9}")
    
    print("-" * 90)
    if failed_tests == 0:
        print("\n✅ All URL exclusion tests passed successfully!")
    else:
        print(f"\n❌ {failed_tests} URL exclusion tests failed!")
    
    # Test fallback URLs don't include Wikipedia
    fallback_urls = retriever._get_fallback_urls("python programming")
    
    print("\nFallback URLs Test:")
    print("-" * 90)
    for url in fallback_urls:
        has_wikipedia = any(domain in url.lower() for domain in ["wikipedia", "wiki/"])
        status = "FAIL" if has_wikipedia else "PASS"
        print(f"{url:<80} | {status:<9}")
    
    # Verify no fallback URLs contain Wikipedia
    has_wikipedia_in_fallbacks = any(
        any(domain in url.lower() for domain in ["wikipedia", "wiki/"]) 
        for url in fallback_urls
    )
    
    if not has_wikipedia_in_fallbacks:
        print("\n✅ No Wikipedia URLs found in fallback URLs!")
    else:
        print("\n❌ Found Wikipedia URLs in fallback URLs!")

if __name__ == "__main__":
    test_url_exclusion() 