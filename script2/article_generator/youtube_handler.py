"""
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

YouTube video search and embedding functionality.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
import traceback
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from utils.rich_provider import provider
from utils.rate_limiter import youtube_rate_limiter

class YouTubeHandler:
    """Handler for YouTube video search and embedding."""
    
    def __init__(self, config):
        """
        Initialize YouTube handler.
        
        Args:
            config: Configuration object containing YouTube settings
        """
        try:
            provider.debug("Initializing YouTube handler")
            self.api_key = config.youtube_api_key
            self.video_width = config.youtube_video_width
            self.video_height = config.youtube_video_height
            self.config = config
            
            provider.debug("Building YouTube API client")
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            provider.debug("YouTube API client initialized successfully")
        except Exception as e:
            provider.error(f"Failed to initialize YouTube API client: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
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
            provider.info(f"Searching YouTube videos for keyword: {keyword}")
            
            # Define the API call function
            def _make_api_call():
                request = self.youtube.search().list(
                    q=keyword,
                    part="snippet",
                    maxResults=max_results,
                    type="video"
                )
                return request.execute()
            
            # Use rate limiter if enabled
            if hasattr(self.config, 'enable_rate_limiting') and self.config.enable_rate_limiting and youtube_rate_limiter:
                provider.debug("Using rate limiter for YouTube API call")
                response = youtube_rate_limiter.execute_with_rate_limit(_make_api_call)
            else:
                response = _make_api_call()
            
            videos = []
            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]
                description = item["snippet"]["description"]
                thumbnail = item["snippet"]["thumbnails"]["high"]["url"]
                
                videos.append({
                    "id": video_id,
                    "title": title,
                    "description": description,
                    "thumbnail": thumbnail,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                })
            
            provider.success(f"Found {len(videos)} videos for keyword: {keyword}")
            return videos
            
        except HttpError as e:
            provider.error(f"YouTube API error: {str(e)}")
            return []
        except Exception as e:
            provider.error(f"Error searching YouTube videos: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return []
            
    def get_embed_html(self, video_id: str) -> str:
        """
        Get HTML iframe code for embedding a YouTube video.
        
        Args:
            video_id (str): YouTube video ID
        Returns:
            str: HTML iframe code
        """
        try:
            provider.debug(f"Generating HTML embed code for video ID: {video_id}")
            embed_code = f'''<iframe 
                width="{self.video_width}" 
                height="{self.video_height}" 
                src="https://www.youtube.com/embed/{video_id}"
                title="YouTube video player" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>'''
            provider.debug("HTML embed code generated successfully")
            return embed_code
        except Exception as e:
            provider.error(f"Error generating HTML embed code: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""
        
    def get_markdown_embed(self, video_id: str) -> str:
        """
        Get markdown code for embedding a YouTube video using HTML iframe.
        
        Args:
            video_id (str): YouTube video ID
        Returns:
            str: HTML iframe embed code for markdown
        """
        try:
            provider.debug(f"Generating Markdown embed code for video ID: {video_id}")
            return self.get_embed_html(video_id)
        except Exception as e:
            provider.error(f"Error generating Markdown embed code: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""
        
    def get_wordpress_embed(self, video_id: str) -> str:
        """
        Get WordPress block code for embedding a YouTube video.
        
        Args:
            video_id (str): YouTube video ID
        Returns:
            str: WordPress embed block code
        """
        try:
            provider.debug(f"Generating WordPress embed code for video ID: {video_id}")
            embed_code = f'''<!-- wp:embed {{"url":"https://www.youtube.com/watch?v={video_id}","type":"video","providerNameSlug":"youtube","responsive":true,"className":"wp-embed-aspect-16-9 wp-has-aspect-ratio"}} -->
<figure class="wp-block-embed is-type-video is-provider-youtube wp-block-embed-youtube wp-embed-aspect-16-9 wp-has-aspect-ratio">
<div class="wp-block-embed__wrapper">
https://www.youtube.com/watch?v={video_id}
</div>
</figure>
<!-- /wp:embed -->'''
            provider.debug("WordPress embed code generated successfully")
            return embed_code
        except Exception as e:
            provider.error(f"Error generating WordPress embed code: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""

    def get_video_for_article(self, keyword: str, output_format: str = 'wordpress') -> Optional[str]:
        """
        Get a relevant video for an article.
        
        Args:
            keyword (str): Search keyword
            output_format (str): Output format ('wordpress', 'markdown', or 'html')
        Returns:
            Optional[str]: Video embed code in requested format
        """
        try:
            provider.info(f"Getting video for article with keyword: {keyword}")
            
            provider.debug("Searching for videos")
            videos = self.search_video(keyword)
            
            if not videos:
                provider.warning("No videos found")
                return None
                
            video = videos[0]  # Get first result
            video_id = video['id']
            provider.debug(f"Selected video: {video['title']} (ID: {video_id})")
            
            provider.debug(f"Generating embed code in {output_format} format")
            if output_format == 'wordpress':
                embed_code = self.get_wordpress_embed(video_id)
            elif output_format in ['markdown', 'html']:
                embed_code = self.get_embed_html(video_id)  # Use HTML iframe for both markdown and html
            else:
                error_msg = f"Unsupported output format: {output_format}"
                provider.error(error_msg)
                raise ValueError(error_msg)
                
            provider.success(f"Successfully generated {output_format} embed code for video")
            return embed_code
                
        except Exception as e:
            provider.error(f"Error getting video for article: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return None

# Standalone function for backward compatibility
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
        provider.info(f"Getting video for article with keyword: {keyword}")
        
        # Create a temporary config-like object
        class TempConfig:
            def __init__(self, api_key, width, height):
                self.youtube_api_key = api_key
                self.youtube_video_width = width
                self.youtube_video_height = height
                
        temp_config = TempConfig(api_key, video_width, video_height)
        
        provider.debug("Initializing YouTube handler")
        handler = YouTubeHandler(temp_config)
        
        return handler.get_video_for_article(keyword, output_format)
            
    except Exception as e:
        provider.error(f"Error getting video for article: {str(e)}")
        provider.error(f"Stack trace:\n{traceback.format_exc()}")
        return None 