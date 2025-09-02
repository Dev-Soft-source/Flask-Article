# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import requests
import os
import base64
from typing import Dict, Optional, List
import mimetypes
import sys
import json
from article_generator.image_handler import generate_image_description,generate_filename,ImageConfig,generate_ai_filename
from transformers import CLIPProcessor, CLIPModel
from urllib.parse import urlparse
import time
import shutil



# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .logger import logger

def get_wordpress_credentials(website_name: str, Username: str, App_pass: str) -> Dict[str, str]:
    """
    Get WordPress API credentials.
    
    Returns:
        Dict[str, str]: WordPress credentials
    """
    logger.debug(f"Generating WordPress credentials for website: {website_name}")
    credentials = {
        'json_url': f'https://{website_name}/wp-json/wp/v2',
        'headers': {
            'Authorization': 'Basic ' + base64.b64encode(
                f'{Username}:{App_pass}'.encode()
            ).decode('ascii')
        }
    }
    logger.debug("WordPress credentials generated successfully")
    return credentials

def list_wordpress_users(website_name: str, Username: str, App_pass: str) -> List[Dict[str, any]]:
    """
    Get a list of WordPress users.
    
    Args:
        website_name (str): WordPress site URL
        Username (str): WordPress username
        App_pass (str): WordPress application password
        
    Returns:
        List[Dict[str, any]]: List of WordPress users with their details
    """
    logger.info(f"Fetching WordPress users from: {website_name}")
    
    credentials = get_wordpress_credentials(website_name, Username, App_pass)
    
    try:
        # Request users data
        response = requests.get(
            f"{credentials['json_url']}/users",
            headers=credentials['headers'],
            params={'per_page': 100}  # Fetch up to 100 users
        )
        response.raise_for_status()
        
        users = response.json()
        logger.success(f"Successfully retrieved {len(users)} WordPress users")
        
        # Format user information
        formatted_users = []
        for user in users:
            formatted_users.append({
                'id': user['id'],
                'name': user['name'],
                'slug': user['slug'],
                'roles': user.get('roles', [])
            })
            logger.debug(f"User found: ID {user['id']}, Name: {user['name']}, Roles: {', '.join(user.get('roles', []))}")
        
        return formatted_users
        
    except requests.exceptions.RequestException as e:
        error_msg = f"HTTP Error while fetching WordPress users: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f"\nResponse Status: {e.response.status_code}"
            try:
                error_msg += f"\nResponse Body: {e.response.json()}"
            except:
                error_msg += f"\nResponse Text: {e.response.text}"
        logger.error(error_msg)
        return []
    except Exception as e:
        logger.error(f"Error fetching WordPress users: {str(e)}")
        return []

def upload_media_to_wordpress(
    processor: CLIPProcessor,
    model: CLIPModel,
    keyword: str,
    file_path_or_url: str,
    alt_text: str,
    caption: str,
    credentials: Dict[str, str],
    max_retries: int = 3,
    retry_delay: int = 5
) -> Optional[Dict]:
    """
    Upload media file or URL to WordPress.
    
    Args:
        file_path_or_url (str): Path to media file or URL
        alt_text (str): Alternative text for the image
        caption (str): Image caption
        credentials (Dict[str, str]): WordPress credentials
        max_retries (int): Maximum number of upload attempts
        retry_delay (int): Delay between retries in seconds
    Returns:
        Optional[Dict]: Media data if successful, None otherwise
    """
    
    try:
        logger.info(f"Starting media upload from: {file_path_or_url}")
        
        # Create temporary directory if it doesn't exist
       
        if not os.path.exists('tmp_images') and not file_path_or_url.startswith('ai_images'):
            os.makedirs('tmp_images')
            
        # Download image if it's a URL
        if bool(urlparse(file_path_or_url).scheme):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Referer": "https://www.google.com/",
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9"
                }
                # Download directly from the provided URL first
                response = requests.get(file_path_or_url,headers=headers, stream=True,timeout=60)
                response.raise_for_status()
                    
                filename = f"{'temp_image'}.jpg"
                file_path = os.path.join('tmp_images', filename)
            
                # Save the file
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
               
                logger.debug(f"Image downloaded successfully to: {file_path}")
                
                if file_path:
                   logger.info("Generating meaningful alt text")
                   
                   alt_text = generate_image_description(processor, model, file_path, keyword)
                   
                   if alt_text:
                      logger.info("Generated meaningful alt text successfully")
                      file_path = generate_filename(alt_text)
                      logger.info(f"file named as {file_path}")
                      
            except Exception as e:
                logger.error(f"Failed to download image: {str(e)}")
                return None
        else:
            
            file_path = file_path_or_url
            
            logger.debug(f"Image found: {file_path}")
            
            if file_path:
                logger.info("Generating meaningful alt text")
                
                alt_text = generate_image_description(processor, model, file_path, keyword)
                
                if alt_text:
                    logger.info("Generated meaningful alt text successfully")
                    file_path = generate_ai_filename(alt_text,file_path)
                    logger.info(f"file named as {file_path}")
            
        # Upload to WordPress
        for attempt in range(max_retries):
            try:
                logger.debug(f"Uploading media to WordPress (attempt {attempt + 1}/{max_retries})")
                
                # Prepare the file upload
                with open(file_path, 'rb') as f:
                    files = {'file': f}
                    
                    # Upload media
                    upload_response = requests.post(
                        f"{credentials['json_url']}/media",
                        headers=credentials['headers'],
                        files=files,
                        timeout=60
                    )
                
                if upload_response.status_code == 201:
                    media_data = upload_response.json()
                    logger.success(f"Media uploaded successfully. ID: {media_data['id']}")
                    
                    # Update media metadata if needed
                    if alt_text or caption:
                        update_data = {}
                        if alt_text:
                            update_data['alt_text'] = alt_text
                        if caption:
                            update_data['caption'] = caption
                            
                        if update_data:
                            update_response = requests.post(
                                f"{credentials['json_url']}/media/{media_data['id']}",
                                headers=credentials['headers'],
                                json=update_data
                            )
                            if update_response.status_code == 200:
                                logger.debug("Media metadata updated successfully")
                    
                    # Clean up temporary file
                    if file_path.startswith('tmp_images') or file_path.startswith("ai_images"):
                         try:
                            os.remove(file_path)
                            logger.debug(f"Temporary file removed: {file_path}")
                         except Exception as e:
                            logger.warning(f"Failed to remove temporary file: {str(e)}")

                        
                    return media_data
                    
                elif attempt < max_retries - 1:
                    logger.warning(f"Upload failed (attempt {attempt + 1}). Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Max retries reached. Upload failed. Status code: {upload_response.status_code}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error during upload: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return None
                    
    except Exception as e:
        logger.error(f"Error uploading media: {str(e)}")
        return None
    finally:
        # Clean up temporary directory if empty
        try:
            if os.path.exists('tmp_images') and not os.listdir('tmp_images'):
                os.rmdir('tmp_images')
            # # Always clean up
            # if os.path.exists("ai_images"):
            #     shutil.rmtree("ai_images")
        except:
            pass

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
    credentials: Optional[Dict[str, str]] = None
) -> Optional[Dict]:
    """
    Create a new WordPress post.
    
    Args:
        title (str): Post title
        content (str): Post content
        categories (str): Category IDs
        author (str): Author ID
        status (str): Post status
        website_name (str): WordPress site URL
        Username (str): WordPress username
        App_pass (str): WordPress application password
        featured_media_id (int, optional): ID of featured image
        credentials (Dict[str, str], optional): WordPress credentials
    Returns:
        Optional[Dict]: Post data if successful, None otherwise
    """
    logger.info(f"Creating WordPress post: {title}")
    
    if not credentials:
        logger.debug("No credentials provided, generating new credentials")
        credentials = get_wordpress_credentials(website_name, Username, App_pass)
        
    try:
        # Prepare post data
        logger.debug("Preparing post data")
        post_data = {
            'title': title,
            'content': content,
            'status': status,
            'categories': categories,
            'author': author
        }
        
        # Add featured media if provided
        if featured_media_id:
            logger.debug(f"Adding featured media ID: {featured_media_id}")
            post_data['featured_media'] = featured_media_id
        
        # Create post
        logger.debug("Sending post creation request")
        response = requests.post(
            f"{credentials['json_url']}/posts",
            headers=credentials['headers'],
            json=post_data
        )
        response.raise_for_status()
        
        post_data = response.json()
        post_id = post_data['id']
        post_url = post_data.get('link', '')
        logger.success(f"Post created successfully. ID: {post_id}")
        if post_url:
            logger.success(f"WordPress Post URL: {post_url}")
        return post_data
        
    except requests.exceptions.RequestException as e:
        error_msg = f"HTTP Error: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f"\nResponse Status: {e.response.status_code}"
            try:
                error_msg += f"\nResponse Body: {e.response.json()}"
            except:
                error_msg += f"\nResponse Text: {e.response.text}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
    except Exception as e:
        import traceback
        error_msg = f"Error creating post: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
 

import re   

def image_url_replacer(html_content: str, new_image_urls: List[str]) -> str:
    """
    Replaces the src attributes of <img> tags in the HTML content with the provided list of new image URLs.

    Args:
        html_content (str): Original HTML content with <img> tags.
        new_image_urls (List[str]): List of new image URLs to replace in order of appearance.

    Returns:
        str: Updated HTML content with replaced image URLs.
    """
    def replacer(match):
        nonlocal index
        if index < len(new_image_urls):
            original = match.group(0)
            new_src = new_image_urls[index]
            index += 1
            return re.sub(r'src=(["\'])(.*?)\1', f'src="{new_src}"', original)
        return match.group(0)

    # Pattern to match <img ... src="..." ...>
    pattern = r'<img\s+[^>]*src=(["\'])(.*?)\1[^>]*>'
    index = 0
    return re.sub(pattern, replacer, html_content)



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
    body_images: Optional[List[str]] = None,
    meta_description: Optional[str] = None,
    wordpress_excerpt: Optional[str] = None,
    keyword: Optional[str] = None
) -> Optional[Dict]:
    """
    Post an article to WordPress using the REST API.
    
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
        keyword: Main keyword to use as slug for permalink
        
    Returns:
        Optional[Dict]: Response from WordPress API or None if failed
    """
    try:
        logger.info(f"Starting WordPress post creation for: {article['title']}")
        
        # Get WordPress credentials
        credentials = get_wordpress_credentials(website_name, Username, App_pass)
        
        # Initialize CLIP model
        processor = CLIPProcessor.from_pretrained(config.image_caption_instance)
        model = CLIPModel.from_pretrained(config.image_caption_instance)
        
        if processor and model:
            logger.info("Clip models initialized for image captioning")
        
        # Upload feature image if provided
        featured_media_id = None
        if feature_image:
            logger.info("Processing feature image...")
            
            if isinstance(feature_image, str):
                # Upload the image
                media_data = upload_media_to_wordpress(
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
                    logger.success(f"Feature image uploaded successfully. ID: {featured_media_id}")
                else:
                    logger.warning("Failed to upload feature image, continuing without it")
            elif isinstance(feature_image, (int, str)) and str(feature_image).isdigit():
                featured_media_id = int(feature_image)
                logger.debug(f"Using existing media ID for feature image: {featured_media_id}")


                
        # Prepare post data
        logger.debug("Preparing post data")
        post_data = {
            'title': article['title'],
            'content':article['content'],
            'status': status,
            'categories': categories,
            'author': author,
            'format': 'standard'
        }
        
        # Set slug (permalink) to use just the keyword
        if keyword:
            # Clean the keyword to be URL-friendly
            clean_keyword = keyword.lower().strip()
            # Replace spaces and special characters with hyphens
            import re
            slug = re.sub(r'[^a-z0-9]+', '-', clean_keyword).strip('-')
            post_data['slug'] = slug
            logger.debug(f"Setting custom permalink slug: {slug}")
        
        # Add featured media if available
        if featured_media_id:
            logger.debug(f"Setting featured media ID: {featured_media_id}")
            post_data['featured_media'] = featured_media_id
        
        # Upload and embed body images
        count = count_image_sources(post_data['content'])
        
        used_urls = []
        new_urls = []
        if body_images and isinstance(body_images, list):
            logger.info("Processing body images...")
            for i, body_img in enumerate(body_images):
                if len(new_urls) == count:
                    break
                 
                if body_img not in used_urls:
                    media_data = upload_media_to_wordpress(
                        processor,
                        model,
                        keyword,
                        file_path_or_url=body_img,
                        alt_text=f"Body image {i+1} for {article['title']}",
                        caption="",
                        credentials=credentials
                    )
                    if media_data:
                        image_url = media_data.get("source_url")
                        new_urls.append(image_url)
                        if image_url:
                            logger.success(f"Body image {i+1} uploaded: {image_url}")
                            used_urls.append(body_img)

                    else:
                        logger.warning(f"Failed to upload body image {i+1}")
                    
        if len(new_urls) != 0:
            post_data['content'] = image_url_replacer(post_data['content'],new_urls)
            
        
        # Always clean up
        if os.path.exists("ai_images"):
            shutil.rmtree("ai_images")
            
        # Add meta description if provided
        if meta_description:
            logger.debug("Adding meta description")
            if 'meta' not in post_data:
                post_data['meta'] = {}
            post_data['meta']['_yoast_wpseo_metadesc'] = meta_description
            
        # Add excerpt if provided
        if wordpress_excerpt:
            logger.debug("Adding WordPress excerpt")
            post_data['excerpt'] = wordpress_excerpt
        
        # Make the request
        logger.debug("Sending post creation request")
        response = requests.post(
            f"{credentials['json_url']}/posts",
            headers=credentials['headers'],
            json=post_data,
            timeout=60
        )
        
        # Check response
        if response.status_code in [200, 201]:
            post_data = response.json()
            post_id = post_data['id']
            post_url = post_data.get('link', '')
            logger.success(f"Successfully posted to WordPress. Post ID: {post_id}")
            if post_url:
                logger.success(f"WordPress Post URL: {post_url}")
            return post_data
        else:
            error_msg = f"Failed to post to WordPress: {response.status_code}"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_msg += f" - {error_data.get('message', response.text)}"
            except:
                error_msg += f" - {response.text}"
            logger.error(error_msg)
            return None
            
    except Exception as e:
        logger.error(f"Error posting to WordPress: {str(e)}")
        return None