#!/usr/bin/env python3

# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import unittest
from utils.url_utils import (
    extract_keyword_for_url,
    handle_duplicate_url,
    generate_post_url,
    clear_url_cache
)

class TestURLGeneration(unittest.TestCase):
    def setUp(self):
        clear_url_cache()

    def test_extract_keyword_for_url(self):
        # Test basic conversion
        self.assertEqual(
            extract_keyword_for_url(
                "10 Expert Tips for Growing Perfect Tomatoes",
                "growing tomatoes"
            ),
            "growing-tomatoes"
        )
        
        # Test special character removal
        self.assertEqual(
            extract_keyword_for_url(
                "How to Grow Tomatoes? (A Complete Guide!)",
                "grow tomatoes"
            ),
            "grow-tomatoes"
        )
        
        # Test multiple spaces and hyphens
        self.assertEqual(
            extract_keyword_for_url(
                "Growing  Tomatoes  -  Expert  Guide",
                "growing   tomatoes"
            ),
            "growing-tomatoes"
        )

    def test_handle_duplicate_url(self):
        # Test increment method
        url1 = handle_duplicate_url("test-url")
        self.assertEqual(url1, "test-url")
        
        url2 = handle_duplicate_url("test-url")
        self.assertEqual(url2, "test-url-2")
        
        url3 = handle_duplicate_url("test-url")
        self.assertEqual(url3, "test-url-3")
        
        # Test UUID method
        url4 = handle_duplicate_url("another-url", "uuid")
        self.assertEqual(url4, "another-url")
        
        url5 = handle_duplicate_url("another-url", "uuid")
        self.assertTrue(url5.startswith("another-url-"))
        self.assertEqual(len(url5), len("another-url") + 9)  # -[8 chars]

    def test_generate_post_url(self):
        # Test using keyword
        url1 = generate_post_url(
            "10 Expert Tips for Growing Perfect Tomatoes",
            "growing tomatoes",
            use_keyword=True
        )
        self.assertEqual(url1, "growing-tomatoes")
        
        # Test using full title
        url2 = generate_post_url(
            "10 Expert Tips for Growing Perfect Tomatoes",
            "growing tomatoes",
            use_keyword=False
        )
        self.assertEqual(url2, "10-expert-tips-for-growing-perfect-tomatoes")
        
        # Test duplicate handling
        url3 = generate_post_url(
            "Another Guide to Growing Tomatoes",
            "growing tomatoes",
            use_keyword=True
        )
        self.assertEqual(url3, "growing-tomatoes-2")
        
        # Test UUID duplicate handling
        url4 = generate_post_url(
            "Yet Another Growing Tomatoes Guide",
            "growing tomatoes",
            use_keyword=True,
            handling_method="uuid"
        )
        self.assertTrue(url4.startswith("growing-tomatoes-"))
        self.assertEqual(len(url4), len("growing-tomatoes") + 9)

if __name__ == '__main__':
    unittest.main()
