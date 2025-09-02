#!/usr/bin/env python3
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import time
from article_generator.logger import logger
from utils.error_utils import ErrorHandler, format_error_message

def test_logger():
    """Test the RAG logger functionality."""
    # Create error handler
    error_handler = ErrorHandler(show_traceback=True)
    
    print("Starting logger test...")
    
    # Test info logging
    logger.info("Testing info message")
    
    # Test debug logging
    logger.debug("Testing debug message with some details")
    
    # Test success logging
    logger.success("Testing success message")
    
    # Test warning logging
    logger.warning("Testing warning message")
    
    # Test error logging with string
    logger.error("Testing error message")
    
    # Test error logging with exception
    try:
        1/0
    except Exception as e:
        # Test the centralized error handler
        error_handler.handle_error(e, {"test": "division by zero", "module": "test_logger"}, "error")
        # Also test the logger's error method
        logger.error(e)
    
    # Test specialized logging
    logger.log_url_access("https://example.com", 200, True)
    logger.log_url_access("https://example.com/404", 404, False)
    
    # Test content scraping log
    logger.log_content_scraping("https://example.com", 5000, "Example Page")
    
    # Test embedding log
    logger.log_embedding_generation(10, 384, "test source")
    
    # Test RAG search log
    logger.log_rag_search("test query", 3, 0.85)
    
    # Test RAG context log
    logger.log_rag_context(2000, 5, ["source1", "source2"])
    
    # Display stats
    logger.display_rag_stats()
    
    # Verify log files were created
    assert os.path.exists(logger.log_dir)
    assert os.path.exists(logger.session_log_file)
    assert os.path.exists(os.path.join(logger.log_dir, "url_access"))
    assert os.path.exists(os.path.join(logger.log_dir, "content_scraping"))
    
    print(f"Log files have been created in: {os.path.abspath(logger.log_dir)}")
    print("Logger test completed successfully")

if __name__ == "__main__":
    test_logger() 