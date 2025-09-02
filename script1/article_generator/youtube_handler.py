"""YouTube video search and embedding functionality."""

import os
import sys
from typing import Dict, List, Optional, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_utils import format_text_for_wordpress
from .logger import logger
from utils.rate_limiter import youtube_rate_limiter

class YouTubeHandler:
    """Handler for YouTube video search and embedding."""
    
    def __init__(self, api_key: str, video_width: int = 560, video_height: int = 315):
        """
        Initialize YouTube handler.
        
        Args:
            api_key (str): YouTube API key
            video_width (int): Width of the video iframe
            video_height (int): Height of the video iframe
        """
        logger.debug("Initializing YouTube handler")
        self.api_key = api_key
        self.video_width = video_width
        self.video_height = video_height
        
        try:
            logger.debug("Building YouTube API client")
            self.youtube = build('youtube', 'v3', developerKey=api_key)
            logger.debug("YouTube API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API client: {str(e)}")
            raise
        
    def search_video(self, keyword: str, max_results: int = 1) -> List[Dict[str, str]]:
        """
        Search for YouTube videos related to a keyword.
        
        Args:
            keyword (str): Search keyword
            max_results (int): Maximum number of results to return
        Returns:
            List[Dict[str, str]]: List of video data dictionaries
        """
        try:
            logger.info(f"Searching for YouTube videos with keyword: {keyword}")
            
            # Use rate limiter if available
            if youtube_rate_limiter:
                logger.debug("Using rate limiter for YouTube API call")
                
                def make_api_call():
                    search_response = self.youtube.search().list(
                        q=keyword,
                        part='snippet',
                        maxResults=max_results,
                        type='video',
                        relevanceLanguage='en',
                        safeSearch='moderate'
                    ).execute()
                    return search_response
                    
                search_response = youtube_rate_limiter.execute_with_rate_limit(make_api_call)
            else:
                search_response = self.youtube.search().list(
                    q=keyword,
                    part='snippet',
                    maxResults=max_results,
                    type='video',
                    relevanceLanguage='en',
                    safeSearch='moderate'
                ).execute()
            
            # Extract video data
            videos = []
            total_items = len(search_response.get('items', []))
            logger.debug(f"Found {total_items} video results")
            
            for i, item in enumerate(search_response.get('items', []), 1):
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                description = item['snippet']['description']
                thumbnail = item['snippet']['thumbnails']['high']['url']
                
                video_data = {
                    'id': video_id,
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'embed_url': f"https://www.youtube.com/embed/{video_id}",
                    'title': title,
                    'description': description,
                    'thumbnail': thumbnail
                }
                videos.append(video_data)
                logger.debug(f"Processed video {i}/{total_items}: {title}")
                
            logger.success(f"Successfully retrieved {len(videos)} videos")
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API error: {str(e)}")
            return []
            
    def get_embed_html(self, video_id: str, width: int = 560, height: int = 315) -> str:
        """
        Get HTML iframe code for embedding a YouTube video.
        
        Args:
            video_id (str): YouTube video ID
            width (int): Iframe width
            height (int): Iframe height
        Returns:
            str: HTML iframe code
        """
        logger.debug(f"Generating HTML embed code for video ID: {video_id}")
        embed_code = f'''<iframe 
            width="{width}" 
            height="{height}" 
            src="https://www.youtube.com/embed/{video_id}"
            title="YouTube video player" 
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>'''
        logger.debug("HTML embed code generated successfully")
        return embed_code
        
    def get_markdown_embed(self, video_id: str) -> str:
        """
        Get markdown code for embedding a YouTube video using HTML iframe.
        
        Args:
            video_id (str): YouTube video ID
        Returns:
            str: HTML iframe embed code for markdown
        """
        logger.debug(f"Generating Markdown embed code for video ID: {video_id}")
        embed_code = f'''<iframe 
            width="{self.video_width}" 
            height="{self.video_height}" 
            src="https://www.youtube.com/embed/{video_id}"
            title="YouTube video player" 
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>'''
        logger.debug("Markdown embed code generated successfully")
        return embed_code
        
    def get_wordpress_embed(self, video_id: str) -> str:
        """
        Get WordPress block code for embedding a YouTube video.
        
        Args:
            video_id (str): YouTube video ID
        Returns:
            str: WordPress embed block code
        """
        logger.debug(f"Generating WordPress embed code for video ID: {video_id}")
        embed_code = f'''<!-- wp:embed {{"url":"https://www.youtube.com/watch?v={video_id}","type":"video","providerNameSlug":"youtube","responsive":true,"className":"wp-embed-aspect-16-9 wp-has-aspect-ratio"}} -->
<figure class="wp-block-embed is-type-video is-provider-youtube wp-block-embed-youtube wp-embed-aspect-16-9 wp-has-aspect-ratio">
<div class="wp-block-embed__wrapper">
https://www.youtube.com/watch?v={video_id}
</div>
</figure>
<!-- /wp:embed -->'''
        logger.debug("WordPress embed code generated successfully")
        return embed_code

def get_video_for_article(api_key: str, keyword: str, output_format: str = 'wordpress', video_width: int = 560, video_height: int = 315) -> Optional[str]:
    """
    Get a relevant video for an article.
    
    Args:
        api_key (str): YouTube API key
        keyword (str): Search keyword
        output_format (str): Output format ('wordpress', 'markdown', or 'html')
        video_width (int): Width of the video iframe
        video_height (int): Height of the video iframe
    Returns:
        Optional[str]: Video embed code in requested format
    """
    try:
        logger.info(f"Getting video for article with keyword: {keyword}")
        
        logger.debug("Initializing YouTube handler")
        handler = YouTubeHandler(api_key, video_width=video_width, video_height=video_height)
        
        logger.debug("Searching for videos")
        videos = handler.search_video(keyword)
        
        if not videos:
            logger.warning("No videos found")
            return None
            
        video = videos[0]  # Get first result
        video_id = video['id']
        logger.debug(f"Selected video: {video['title']} (ID: {video_id})")
        
        logger.debug(f"Generating embed code in {output_format} format")
        if output_format == 'wordpress':
            embed_code = handler.get_wordpress_embed(video_id)
        elif output_format in ['markdown', 'html']:
            embed_code = handler.get_embed_html(video_id)  # Use HTML iframe for both markdown and html
        else:
            error_msg = f"Unsupported output format: {output_format}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.success(f"Successfully generated {output_format} embed code for video")
        return embed_code
            
    except Exception as e:
        logger.error(f"Error getting video for article: {str(e)}")
        return None 