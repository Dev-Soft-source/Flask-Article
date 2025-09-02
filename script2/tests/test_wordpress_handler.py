#!/usr/bin/env python3

# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from article_generator.wordpress_handler import post_to_wordpress
from utils.url_utils import clear_url_cache

class TestWordPressHandler(unittest.TestCase):
    def setUp(self):
        self.test_article = {
            'title': '10 Expert Tips for Growing Perfect Tomatoes',
            'content': '<p>Test content about growing tomatoes.</p>'
        }
        self.test_credentials = {
            'json_url': 'https://example.com/wp-json/wp/v2',
            'headers': {'Authorization': 'Basic dGVzdDp0ZXN0'}
        }
        clear_url_cache()

    @patch('article_generator.wordpress_handler.get_wordpress_credentials')
    @patch('article_generator.wordpress_handler.requests.post')
    def test_post_creation_with_keyword_url(self, mock_post, mock_get_creds):
        # Mock WordPress credentials
        mock_get_creds.return_value = self.test_credentials
        
        # Mock successful post creation response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            'id': 123,
            'link': 'https://example.com/growing-tomatoes'
        }
        mock_post.return_value = mock_response

        # Test with keyword-based URL
        result = post_to_wordpress(
            website_name='test.com',
            Username='test',
            App_pass='test123',
            categories='1',
            author='1',
            status='draft',
            article=self.test_article,
            keyword='growing tomatoes',
            use_keyword_for_url=True
        )

        # Verify the post request was made with the correct slug
        calls = mock_post.call_args_list
        self.assertEqual(len(calls), 1)
        
        # Get the post data from the request
        post_data = calls[0].kwargs['json']
        
        # Verify the slug was generated correctly
        self.assertEqual(post_data['slug'], 'growing-tomatoes')

    @patch('article_generator.wordpress_handler.get_wordpress_credentials')
    @patch('article_generator.wordpress_handler.requests.post')
    def test_post_creation_with_title_url(self, mock_post, mock_get_creds):
        # Mock WordPress credentials
        mock_get_creds.return_value = self.test_credentials
        
        # Mock successful post creation response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            'id': 123,
            'link': 'https://example.com/10-expert-tips-for-growing-perfect-tomatoes'
        }
        mock_post.return_value = mock_response

        # Test with title-based URL
        result = post_to_wordpress(
            website_name='test.com',
            Username='test',
            App_pass='test123',
            categories='1',
            author='1',
            status='draft',
            article=self.test_article,
            keyword='growing tomatoes',
            use_keyword_for_url=False
        )

        # Verify the post request was made with the correct slug
        calls = mock_post.call_args_list
        self.assertEqual(len(calls), 1)
        
        # Get the post data from the request
        post_data = calls[0].kwargs['json']
        
        # Verify the slug was generated correctly
        self.assertEqual(post_data['slug'], '10-expert-tips-for-growing-perfect-tomatoes')

    @patch('article_generator.wordpress_handler.get_wordpress_credentials')
    @patch('article_generator.wordpress_handler.requests.post')
    def test_post_creation_with_duplicate_url_handling(self, mock_post, mock_get_creds):
        # Mock WordPress credentials
        mock_get_creds.return_value = self.test_credentials
        
        # Mock successful post creation response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            'id': 123,
            'link': 'https://example.com/growing-tomatoes'
        }
        mock_post.return_value = mock_response

        # Create first post (should get growing-tomatoes)
        result1 = post_to_wordpress(
            website_name='test.com',
            Username='test',
            App_pass='test123',
            categories='1',
            author='1',
            status='draft',
            article=self.test_article,
            keyword='growing tomatoes',
            use_keyword_for_url=True
        )

        # Create second post (should get growing-tomatoes-2)
        result2 = post_to_wordpress(
            website_name='test.com',
            Username='test',
            App_pass='test123',
            categories='1',
            author='1',
            status='draft',
            article=self.test_article,
            keyword='growing tomatoes',
            use_keyword_for_url=True
        )

        # Verify both posts were created with different slugs
        calls = mock_post.call_args_list
        self.assertEqual(len(calls), 2)
        
        # Get the post data from both requests
        post_data1 = calls[0].kwargs['json']
        post_data2 = calls[1].kwargs['json']
        
        # Verify the slugs were generated correctly
        self.assertEqual(post_data1['slug'], 'growing-tomatoes')
        self.assertEqual(post_data2['slug'], 'growing-tomatoes-2')

if __name__ == '__main__':
    unittest.main()
