# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import base64
import json
import requests
import os
import time
from typing import Dict, Any, Optional, Union, List
from config import Config
from utils.api_utils import RetryHandler
from utils.rich_provider import provider
from utils.url_utils import generate_post_url
import traceback
import uuid
import re
from article_generator.image_handler import generate_image_description,generate_filename,randomize_image_selection,generate_ai_filename
from transformers import CLIPProcessor, CLIPModel
from article_generator.image_handler import ImageConfig
import shutil

class WordPressHandler:
    """Handles WordPress integration for article publishing."""
    
    def __init__(self, config: Config):
        """Initialize the WordPress handler with configuration."""
        self.config = config
        
        # Format website URL
        WP_WEBSITE_NAME = config.WP_WEBSITE_NAME
        if not WP_WEBSITE_NAME.startswith(('http://', 'https://')):
            WP_WEBSITE_NAME = f"https://{WP_WEBSITE_NAME}"
            
        # Remove trailing slash if present
        if WP_WEBSITE_NAME.endswith('/'):
            WP_WEBSITE_NAME = WP_WEBSITE_NAME[:-1]
            
        self.base_url = f'{WP_WEBSITE_NAME}/wp-json/wp/v2'
        self.auth_token = base64.b64encode(
            f'{config.WP_USERNAME}:{config.wp_app_pass}'.encode()
        ).decode('utf-8')
        self.retry_handler = RetryHandler(config)
        
        provider.debug(f"WordPress handler initialized with base URL: {self.base_url}")
        
    def validate_connection(self) -> bool:
        """Validate WordPress connection and credentials."""
        provider.info("Validating WordPress connection...")
        
        try:
            def _test_connection():
                response = requests.get(
                    f"{self.base_url}/users/me",
                    headers={"Authorization": f"Basic {self.auth_token}"},
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
                
            user_data = self.retry_handler.execute_with_retry(_test_connection)
            
            if user_data and 'id' in user_data:
                provider.success(f"WordPress connection validated. User: {user_data.get('name', 'Unknown')}")
                return True
            else:
                provider.error("WordPress validation failed: Invalid response")
            return False
            
        except Exception as e:
            provider.error(f"WordPress validation failed: {str(e)}")
            return False
            
    def upload_media(self, file_path: str, title: str) -> Optional[Dict[str, Any]]:
        """Upload media to WordPress."""
        provider.info(f"Uploading media: {file_path}")
        
        try:
            def _upload_file():
                with open(file_path, 'rb') as f:
                    files = {'file': f}
                    headers = {
                        'Authorization': f'Basic {self.auth_token}',
                        'Content-Disposition': f'attachment; filename="{os.path.basename(file_path)}"'
                    }
                    
                    response = requests.post(
                                    f"{self.base_url}/media",
                        headers=headers,
                        files=files
                    )
                    response.raise_for_status()
                    return response.json()
                    
            media_data = self.retry_handler.execute_with_retry(_upload_file)
            
            if media_data and 'id' in media_data:
                provider.success(f"Media uploaded successfully. ID: {media_data['id']}")
                return media_data
            else:
                provider.error("Media upload failed: Invalid response")
            return None
            
        except Exception as e:
            provider.error(f"Media upload failed: {str(e)}")
            return None
            
    def create_post(self, article_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new WordPress post."""
        provider.info(f"Creating WordPress post: {article_data.get('title', 'Untitled')}")
        
        try:
            def _create_post():
                headers = {
                    'Authorization': f'Basic {self.auth_token}',
                    'Content-Type': 'application/json'
                }
                
                response = requests.post(
                    f"{self.base_url}/posts",
                    headers=headers,
                    json=article_data
                )
                response.raise_for_status()
                return response.json()
                
            post_data = self.retry_handler.execute_with_retry(_create_post)
            
            if post_data and 'id' in post_data:
                post_id = post_data['id']
                post_url = post_data.get('link', '')
                provider.success(f"Post created successfully. ID: {post_id}")
                if post_url:
                    provider.success(f"WordPress Post URL: {post_url}")
                return post_data
            else:
                provider.error("Post creation failed: Invalid response")
            return None
            
        except Exception as e:
            provider.error(f"Post creation failed: {str(e)}")
            return None
            
    def delete_post(self, post_id: Union[int, str]) -> bool:
        """Delete a WordPress post."""
        provider.info(f"Deleting WordPress post: {post_id}")
        
        try:
            def _delete_post():
                headers = {'Authorization': f'Basic {self.auth_token}'}
                response = requests.delete(
                                f"{self.base_url}/posts/{post_id}",
                                headers=headers,
                                params={'force': True}
                            )
                response.raise_for_status()
                return response.json()
                
            result = self.retry_handler.execute_with_retry(_delete_post)
            
            if result and result.get('deleted', False):
                provider.success(f"Post {post_id} deleted successfully")
                return True
            else:
                provider.error(f"Post deletion failed: {result}")
                return False
            
        except Exception as e:
            provider.error(f"Post deletion failed: {str(e)}")
            return False
            
    def update_post(self, post_id: Union[int, str], update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing WordPress post."""
        provider.info(f"Updating WordPress post: {post_id}")
        
        try:
            def _update_post():
                headers = {
                    'Authorization': f'Basic {self.auth_token}',
                    'Content-Type': 'application/json'
                }
                
                response = requests.post(
                                f"{self.base_url}/posts/{post_id}",
                                headers=headers,
                    json=update_data
                )
                response.raise_for_status()
                return response.json()
                
            post_data = self.retry_handler.execute_with_retry(_update_post)
            
            if post_data and 'id' in post_data:
                post_id = post_data['id']
                post_url = post_data.get('link', '')
                provider.success(f"Post updated successfully. ID: {post_id}")
                if post_url:
                    provider.success(f"WordPress Post URL: {post_url}")
                return post_data
            else:
                provider.error("Post update failed: Invalid response")
                return None
                
        except Exception as e:
            provider.error(f"Post update failed: {str(e)}")
            return None

    def generate_tags(self, keyword: str, title: str, content: str) -> List[str]:
        """
        Generate relevant tags for a WordPress post based on keyword, title and content.
        
        Args:
            keyword (str): Main keyword for the article
            title (str): Article title
            content (str): Article content
            
        Returns:
            List[str]: List of generated tags
        """
        provider.info(f"Generating tags for article: {title}")
        
        try:
            # Start with the main keyword as a tag
            tags = [keyword]
            
            # Add variations of the keyword
            keyword_parts = keyword.split()
            if len(keyword_parts) > 1:
                # Add individual words from multi-word keyword
                tags.extend(keyword_parts)
            
            # Extract potential tags from title
            title_words = title.replace(',', '').replace('.', '').replace('?', '').replace('!', '').split()
            for word in title_words:
                if len(word) > 3 and word.lower() not in [t.lower() for t in tags]:
                    tags.append(word)
            
            # Limit to 10 tags maximum
            tags = tags[:10]
            
            provider.success(f"Generated {len(tags)} tags: {', '.join(tags)}")
            return tags
            
        except Exception as e:
            provider.error(f"Error generating tags: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            # Return at least the keyword as a tag
            return [keyword]

    def create_or_get_tags(self, tag_names: List[str]) -> List[int]:
        """
        Create tags in WordPress or get IDs of existing tags.
        
        Args:
            tag_names (List[str]): List of tag names to create or get
            
        Returns:
            List[int]: List of tag IDs
        """
        provider.info(f"Creating or getting tags: {', '.join(tag_names)}")
        tag_ids = []
        
        try:
            for tag_name in tag_names:
                # First check if tag exists
                def _get_tag():
                    headers = {'Authorization': f'Basic {self.auth_token}'}
                    response = requests.get(
                        f"{self.base_url}/tags",
                        headers=headers,
                        params={'search': tag_name}
                    )
                    response.raise_for_status()
                    return response.json()
                
                existing_tags = self.retry_handler.execute_with_retry(_get_tag)
                
                # If tag exists, use its ID
                if existing_tags and len(existing_tags) > 0:
                    for tag in existing_tags:
                        if tag.get('name', '').lower() == tag_name.lower():
                            tag_ids.append(tag['id'])
                            provider.debug(f"Found existing tag: {tag_name} (ID: {tag['id']})")
                            break
                    else:
                        # Tag not found, create it
                        def _create_tag():
                            headers = {
                                'Authorization': f'Basic {self.auth_token}',
                                'Content-Type': 'application/json'
                            }
                            response = requests.post(
                                f"{self.base_url}/tags",
                                headers=headers,
                                json={'name': tag_name}
                            )
                            response.raise_for_status()
                            return response.json()
                        
                        new_tag = self.retry_handler.execute_with_retry(_create_tag)
                        if new_tag and 'id' in new_tag:
                            tag_ids.append(new_tag['id'])
                            provider.debug(f"Created new tag: {tag_name} (ID: {new_tag['id']})")
                else:
                    # No tags found, create new one
                    def _create_tag():
                        headers = {
                            'Authorization': f'Basic {self.auth_token}',
                            'Content-Type': 'application/json'
                        }
                        response = requests.post(
                            f"{self.base_url}/tags",
                            headers=headers,
                            json={'name': tag_name}
                        )
                        response.raise_for_status()
                        return response.json()
                    
                    new_tag = self.retry_handler.execute_with_retry(_create_tag)
                    if new_tag and 'id' in new_tag:
                        tag_ids.append(new_tag['id'])
                        provider.debug(f"Created new tag: {tag_name} (ID: {new_tag['id']})")
            
            provider.success(f"Processed {len(tag_ids)} tags")
            return tag_ids
            
        except Exception as e:
            provider.error(f"Error creating/getting tags: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return tag_ids

def get_wordpress_credentials(website_name: str, Username: str, App_pass: str) -> Dict[str, str]:
    """
    Generate WordPress API credentials.
    
    Args:
        website_name (str): WordPress site URL
        Username (str): WordPress username
        App_pass (str): WordPress application password
    Returns:
        Dict[str, str]: WordPress credentials
    """
    try:
        provider.debug(f"Generating WordPress credentials for {website_name}")
        
        # Format website URL
        if not website_name.startswith(('http://', 'https://')):
            website_name = f"https://{website_name}"
            
        # Remove trailing slash if present
        if website_name.endswith('/'):
            website_name = website_name[:-1]
            
        # Create authentication token
        token = base64.b64encode(f"{Username}:{App_pass}".encode()).decode()
        
        # Create credentials dictionary
        credentials = {
            'website': website_name,
            'json_url': f"{website_name}/wp-json/wp/v2",
            'headers': {
                'Authorization': f'Basic {token}'
            },
            'auth_token': token
        }
        
        provider.debug(f"WordPress credentials generated for {website_name}")
        provider.debug(f"WordPress API URL: {credentials['json_url']}")
        return credentials
        
    except Exception as e:
        provider.error(f"Error generating WordPress credentials: {str(e)}")
        provider.error(f"Detailed traceback:\n{traceback.format_exc()}")
        raise

def upload_media_to_wordpress(
    title: str,
    processor:CLIPProcessor,
    model:CLIPModel,
    keyword:str,
    file_path_or_url: str,
    alt_text: str,
    caption: str,
    credentials: Dict[str, str],
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[Dict]:
    """
    Upload media to WordPress.
    
    Args:
        title: Title of the post.
        file_path_or_url (str): Path to file or URL
        alt_text (str): Alt text for the media
        caption (str): Caption for the media
        credentials (Dict[str, str]): WordPress credentials
        max_retries (int, optional): Maximum number of retries
        retry_delay (int, optional): Delay between retries in seconds
    Returns:
        Optional[Dict]: Media data if successful, None otherwise
    """


    provider.info(f"Uploading media to WordPress: {file_path_or_url}")
   
    
    # Check if file_path_or_url is a URL
    if file_path_or_url.startswith(('http://', 'https://')):
        provider.debug(f"Downloading file from URL: {file_path_or_url}")
        try:
            # Create a temporary directory for downloaded images if it doesn't exist
            tmp_dir = 'tmp_images'
            os.makedirs(tmp_dir, exist_ok=True)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": "https://www.google.com/",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9"
            }
            # Download the file
            response = requests.get(file_path_or_url,headers=headers, stream=True,timeout=60)
            response.raise_for_status()
                
            # Create a unique filename
            filename = f"{'temp_image'}.jpg"
            file_path = os.path.join(tmp_dir, filename)
            
            # Save the file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            provider.debug(f"File downloaded to: {file_path}")
            
            file_path_or_url = file_path
            
            if file_path_or_url:
                provider.info("Generating meaningful alt text")
                
                alt_text = generate_image_description(processor, model, file_path_or_url, title)
                
                if alt_text:
                    provider.info("Generated meaningful alt text successfully")
                    file_path_or_url = generate_filename(alt_text)
                    provider.info(f"file named as {file_path_or_url}")
                    
        except Exception as e:
            provider.error(f"Error downloading file: {str(e)}")
            return None
    else:
        
        provider.debug(f"Image found: {file_path_or_url}")
        
        if file_path_or_url:
            provider.info("Generating meaningful alt text")
            
            alt_text = generate_image_description(processor, model, file_path_or_url, keyword)
            
            if alt_text:
                provider.info("Generated meaningful alt text successfully")
                file_path_or_url = generate_ai_filename(alt_text,file_path_or_url)
                provider.info(f"file named as {file_path_or_url}")

    # Upload the file
    for attempt in range(max_retries):
        try:
            provider.debug(f"Upload attempt {attempt + 1}/{max_retries}")
            
            # Check if file exists
            if not os.path.exists(file_path_or_url):
                provider.error(f"File not found: {file_path_or_url}")
                return None
            
            # Get file name and mime type
            file_name = os.path.basename(file_path_or_url)
            mime_type = 'image/jpg'  # Default
            if file_name.lower().endswith('.png'):
                mime_type = 'image/png'
            elif file_name.lower().endswith('.gif'):
                mime_type = 'image/gif'
            
            # Create upload headers - IMPORTANT: Remove Content-Type for multipart/form-data
            upload_headers = {
                'Authorization': credentials['headers']['Authorization']
            }
            
            # Read file content
            with open(file_path_or_url, 'rb') as img_file:
                file_content = img_file.read()
            
            # Upload the file using requests
            upload_response = requests.post(
                f"{credentials['json_url']}/media",
                headers=upload_headers,
                files={
                    'file': (file_name, file_content, mime_type)
                },
                data={
                    'alt_text': alt_text,
                    'caption': caption
                },
                timeout=60
            )
            
            # Check response
            if upload_response.status_code == 201:
                media_data = upload_response.json()
                provider.success(f"Media uploaded successfully. ID: {media_data['id']}")
                
                # Update media metadata if needed (sometimes the data parameter doesn't work)
                if alt_text or caption:
                    update_data = {}
                    if alt_text:
                        update_data['alt_text'] = alt_text
                    if caption:
                        update_data['caption'] = caption
                    
                    if update_data:
                        # Use regular headers with Content-Type for JSON requests
                        update_response = requests.post(
                            f"{credentials['json_url']}/media/{media_data['id']}",
                            headers=credentials['headers'],
                            json=update_data
                        )
                        if update_response.status_code == 200:
                            provider.debug("Media metadata updated successfully")
                
                # Clean up temporary file
                if file_path_or_url.startswith('tmp_images') or file_path.startswith("ai_images"):
                    try:
                        os.remove(file_path_or_url)
                        provider.debug(f"Temporary file removed: {file_path_or_url}")
                    except Exception as e:
                        provider.warning(f"Failed to remove temporary file: {str(e)}")
                
                return media_data
            
            elif attempt < max_retries - 1:
                provider.warning(f"Upload failed (attempt {attempt + 1}). Status code: {upload_response.status_code}")
                provider.warning(f"Response: {upload_response.text}")
                time.sleep(retry_delay)
            else:
                provider.error(f"Max retries reached. Upload failed. Status code: {upload_response.status_code}")
                provider.error(f"Response: {upload_response.text}")
                return None
        
        except Exception as e:
            provider.error(f"Error during upload: {str(e)}")
            provider.error(f"Traceback: {traceback.format_exc()}")
            if attempt < max_retries - 1:
                provider.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return None
    
    return None

def create_wordpress_post(
    title: str,
    content: str,
    categories: str,
    author: str,
    status: str,
    website_name: str,
    Username: str,
    App_pass: str,
    featured_media_id: Optional[int] = None,
    tags: Optional[List[int]] = None,
    credentials: Optional[Dict[str, str]] = None
) -> Optional[Dict]:
    """
    Create a WordPress post.
    
    Args:
        title (str): Post title
        content (str): Post content
        categories (str): Category IDs
        author (str): Author ID
        status (str): Post status
        website_name (str): WordPress site URL
        Username (str): WordPress username
        App_pass (str): WordPress application password
        featured_media_id (Optional[int]): Featured media ID
        tags (Optional[List[int]]): List of tag IDs
        credentials (Optional[Dict[str, str]]): WordPress credentials
    Returns:
        Optional[Dict]: Post data if successful, None otherwise
    """
    try:
        provider.info("Creating WordPress post...")
        
        # Get credentials if not provided
        if not credentials:
            credentials = get_wordpress_credentials(website_name, Username, App_pass)
            
        # Prepare post data
        post_data = {
            'title': title,
            'content': content,
            'status': status,
            'author': author,
            'categories': categories.split(',') if isinstance(categories, str) else categories
        }
        
        # Add featured media if provided
        if featured_media_id:
            post_data['featured_media'] = featured_media_id
            
        # Add tags if provided
        if tags:
            post_data['tags'] = tags
            
        # Create post
        provider.debug(f"Sending request to create post: {post_data['title']}")
        response = requests.post(
            f"{credentials['json_url']}/posts",
            headers=credentials['headers'],
            json=post_data
        )
        
        # Check response
        if response.status_code == 201:
            post_data = response.json()
            post_id = post_data['id']
            post_url = post_data.get('link', '')
            provider.success(f"Post created successfully. ID: {post_id}")
            if post_url:
                provider.success(f"WordPress Post URL: {post_url}")
            return post_data
        else:
            provider.error(f"Failed to create post. Status code: {response.status_code}")
            provider.error(f"Response: {response.text}")
            return None
            
    except Exception as e:
        provider.error(f"Error creating WordPress post: {str(e)}")
        provider.error(f"Detailed traceback:\n{traceback.format_exc()}")
        return None

def image_url_replacer(html_content: str, new_image_urls: List[str]) -> str:
    """
    Replaces the src attributes of <img> tags in the HTML content with unique image URLs,
    removes extra <img> tags and their <figcaption> if there are more <img> tags than new images.

    Args:
        html_content (str): Original HTML content with <img> tags.
        new_image_urls (List[str]): List of new image URLs.

    Returns:
        str: Updated HTML content.
    """
    used_urls = set()
    unique_urls = [url for url in new_image_urls if url not in used_urls and not used_urls.add(url)]
    index = 0

    def replacer(match):
        nonlocal index
        if index < len(unique_urls):
            original = match.group(0)
            new_src = unique_urls[index]
            index += 1
            return re.sub(r'src=(["\'])(.*?)\1', f'src="{new_src}"', original)
        else:
            return ''  # Temporarily remove; we'll delete unused figures after this

    # Replace <img> tags one by one
    img_pattern = r'<img\s+[^>]*src=(["\'])(.*?)\1[^>]*>'
    updated_html = re.sub(img_pattern, replacer, html_content)

    # Now remove any <figure> blocks that are missing <img> tags (i.e., were replaced with '')
    # This handles the full figure: <figure>...</figure>
    figure_pattern = r'<figure>\s*(?:<img[^>]*>\s*)?(<figcaption>.*?</figcaption>)?\s*</figure>'
    def clean_empty_figures(match):
        figure_content = match.group(0)
        if '<img' not in figure_content:
            return ''  # Remove the entire <figure> block
        return figure_content

    updated_html = re.sub(figure_pattern, clean_empty_figures, updated_html, flags=re.DOTALL)

    return updated_html


def count_image_sources(html_content: str) -> int:
    """
    Counts the number of <img> tags with a src attribute in the HTML content.

    Args:
        html_content (str): The HTML content to search.

    Returns:
        int: Number of <img> tags with a src attribute.
    """
    pattern = r'<img\s+[^>]*src=(["\'])(.*?)\1'
    matches = re.findall(pattern, html_content)
    return len(matches)

def post_to_wordpress(
    config: ImageConfig,
    website_name: str,
    Username: str,
    App_pass: str,
    categories: str,
    author: str,
    status: str,
    article: Dict[str, str],
    feature_image: Optional[str] = None,
    body_images: Optional[List[Dict[str, str]]] = None,
    meta_description: Optional[str] = None,
    wordpress_excerpt: Optional[str] = None,
    tags: Optional[List[str]] = None,
    keyword: Optional[str] = None,
    use_keyword_for_url: bool = True,
    url_duplicate_handling: str = "increment"
) -> Optional[Dict]:
    """Create a WordPress post with the provided content and settings.

    Args:
        config: Image configuration
        website_name: WordPress site domain
        Username: WordPress username
        App_pass: WordPress application password
        categories: Category ID(s)
        author: Author ID
        status: Post status (draft/publish)
        article: Dictionary containing article content
        feature_image: URL or ID of feature image
        body_images: List of body image URLs or IDs
        meta_description: SEO meta description
        wordpress_excerpt: WordPress excerpt
        tags: List of tags
        keyword: Main keyword to use as slug for permalink
        use_keyword_for_url: Whether to use keyword for URL instead of full title
        url_duplicate_handling: How to handle duplicate URLs ("increment" or "uuid")
        
    Returns:
        Optional[Dict]: Response from WordPress API or None if failed
    """
    try:
        provider.info(f"Starting WordPress post creation for: {article['title']}")
        
        # Get WordPress credentials
        credentials = get_wordpress_credentials(website_name, Username, App_pass)
        
         # Initialize CLIP model
        processor = CLIPProcessor.from_pretrained(config.image_caption_instance)
        model = CLIPModel.from_pretrained(config.image_caption_instance)
        
        if processor and model:
            provider.info("Clip models initialized for image captioning")
        
        
        # Upload feature image if provided
        featured_media_id = None
        if feature_image:
            provider.info("Processing feature image...")
            if isinstance(feature_image, str):
                # Upload the image
                media_data = upload_media_to_wordpress(
                    article['title'],
                    processor,
                    model,
                    keyword,
                    feature_image,
                    alt_text=f"Feature image for {article['title']}",
                    caption="",
                    credentials=credentials
                )
                
                if media_data:
                    featured_media_id = media_data['id']
                    provider.success(f"Feature image uploaded successfully. ID: {featured_media_id}")
                else:
                    provider.warning("Failed to upload feature image, continuing without it")
            elif isinstance(feature_image, (int, str)) and str(feature_image).isdigit():
                featured_media_id = int(feature_image)
                provider.debug(f"Using existing media ID for feature image: {featured_media_id}")
        
        # Prepare post data
        provider.debug("Preparing post data")
        post_data = {
            'title': article['title'],
            'content': article['content'],
            'status': status,
            'categories': categories.split(',') if isinstance(categories, str) else categories,
            'author': author,
            'format': 'standard'
        }
        
        # Set slug (permalink) using url_utils
        if keyword:
            slug = generate_post_url(
                title=article['title'],
                keyword=keyword,
                use_keyword=use_keyword_for_url,
                handling_method=url_duplicate_handling
            )
            post_data['slug'] = slug
            provider.debug(f"Setting custom permalink slug: {slug}")
        
        # Add featured media if available
        if featured_media_id:
            provider.debug(f"Setting featured media ID: {featured_media_id}")
            post_data['featured_media'] = featured_media_id
           
            
        # Upload and embed body images
        count = count_image_sources(post_data['content'])
        new_body_image_selection = randomize_image_selection(body_images)

        new_urls = []
        used_urls = []
        if body_images and isinstance(new_body_image_selection, list):
            provider.info("Processing body images...")
            for i, body_img in enumerate(body_images):
                
                if len(new_urls) == count:
                    break
                
                if body_img not in used_urls:
                    
                    media_data = upload_media_to_wordpress(
                        article['title'],
                        processor,
                        model,
                        keyword,
                        body_img['file'],
                        alt_text=f"Body image {i+1} for {article['title']}",
                        caption="",
                        credentials=credentials
                    )
                    
                    if media_data:
                        
                        image_url = media_data.get("source_url")
                        new_urls.append(image_url)
                        if image_url:
                            provider.success(f"Body image {i+1} uploaded: {image_url}")
                            used_urls.append(body_img)
                    else:
                        provider.warning(f"Failed to upload body image {i+1}")
                    
        if len(new_urls) != 0:
            post_data['content'] = image_url_replacer(post_data['content'],new_urls)
            
         
        # Always clean up
        if os.path.exists("ai_images"):
            shutil.rmtree("ai_images")
        
        
        # Add tags if provided
        if tags:
            provider.debug(f"Adding {len(tags)} tags")
            post_data['tags'] = tags
        
        # Add meta description if provided
        if meta_description:
            provider.debug("Adding meta description")
            if 'meta' not in post_data:
                post_data['meta'] = {}
            post_data['meta']['_yoast_wpseo_metadesc'] = meta_description
            
        # Add excerpt if provided
        if wordpress_excerpt:
            provider.debug("Adding WordPress excerpt")
            post_data['excerpt'] = wordpress_excerpt
        
        # Make the request
        provider.debug("Sending post creation request")
        
        # Add Content-Type header for JSON request
        json_headers = credentials['headers'].copy()
        json_headers['Content-Type'] = 'application/json'
        
        response = requests.post(
            f"{credentials['json_url']}/posts",
            headers=json_headers,
            json=post_data,
            timeout=60
        )
        
        # Check response
        if response.status_code in [200, 201]:
            post_data = response.json()
            post_id = post_data['id']
            post_url = post_data.get('link', '')
            provider.success(f"Successfully posted to WordPress. Post ID: {post_id}")
            if post_url:
                provider.success(f"WordPress Post URL: {post_url}")
            return post_data
        else:
            error_msg = f"Failed to post to WordPress: {response.status_code}"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_msg += f" - {error_data.get('message', response.text)}"
            except:
                error_msg += f" - {response.text}"
            provider.error(error_msg)
            return None
            
    except Exception as e:
        provider.error(f"Error posting to WordPress: {str(e)}")
        provider.error(f"Detailed traceback:\n{traceback.format_exc()}")
        return None