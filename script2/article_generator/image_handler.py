# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import requests
import os
import sys
import tempfile
from typing import List, Dict, Optional, Tuple
import random
from dataclasses import dataclass
from utils.rich_provider import provider
from utils.rate_limiter import unsplash_rate_limiter,openverse_rate_limiter,pexels_rate_limiter,pixabay_rate_limiter,hugging_face_rate_limiter
import base64
import torch
import time
from transformers import CLIPProcessor, CLIPModel
import re
import random
from io import BytesIO
import shutil


# Import PIL for image compression
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    provider.warning("PIL/Pillow not installed. Image compression will be disabled.")
    PIL_AVAILABLE = False

@dataclass
class ImageConfig:
    """Configuration for image generation and handling."""
      # Image settings 
    image_source: str = "Stock"
    stock_primary_source: str = "openverse"
    secondary_source_image: bool = True
    
    # AI image setting
    huggingface_model:str = ""
    huggingface_api_key:str = ""
    
    # Image sources api keys
    unsplash_api_key:str = ""
    pexels_api_key:str = ""
    pixabay_api_key:str = ""
    giphy_api_key:str = ""
    
    # Image captioning settings 
    image_caption_instance:str = ""
    
    
    enable_image_generation: bool = False
    max_number_of_images: int = 7
    orientation: str = "landscape"
    order_by: str = "relevant"
    image_api: bool = True
    # Image alignment options: "aligncenter", "alignleft", "alignright"
    alignment: str = "aligncenter"
    # Image compression options
    enable_image_compression: bool = False
    compression_quality: int = 70  # 0-100, higher is better quality but larger file size
    # Prevent duplicate images in the same article
    prevent_duplicate_images: bool = False

def get_image_list_unsplash(
    keyword: str,
    config: ImageConfig,
    num_images: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Get a list of images from Unsplash API.
    
    Args:
        keyword (str): Search keyword for images
        config (ImageConfig): Image configuration settings
        num_images (int, optional): Number of images to fetch
    Returns:
        List[Dict[str, str]]: List of image data
    """
    if not config.enable_image_generation:
        provider.debug("Image generation is disabled")
        return []
    
    provider.info(f"Fetching {num_images} images for keyword: {keyword}")
    
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": keyword,
        "per_page": num_images,
        "orientation": config.orientation if config.orientation != "any" else None,
        "order_by": config.order_by
    }
    
    headers = {
        "Authorization": f"Client-ID {config.unsplash_api_key}",
        "Accept-Version": "v1"
    }
    
    try:
        provider.debug(f"Sending request to Unsplash API with params: {params}")
            
        # Define the API call function
        def _make_api_call():
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        
        # Use rate limiter if enabled
        if hasattr(config, 'enable_rate_limiting') and config.enable_rate_limiting and unsplash_rate_limiter:
            provider.debug("Using rate limiter for Unsplash API call")
            data = unsplash_rate_limiter.execute_with_rate_limit(_make_api_call)
        else:
            data = _make_api_call()
        
        results = []
        for photo in data.get("results", []):
            results.append({
                "id": photo["id"],
                "url": photo["urls"]["regular"],
                "download_url": photo["links"]["download"],
                "alt_description": photo.get("alt_description", keyword),
                "width": photo["width"],
                "height": photo["height"],
                "photographer": photo["user"]["name"],
                "photographer_url": photo["user"]["links"]["html"]
            })
        
        provider.success(f"Found {len(results)} images for keyword: {keyword}")
        return results
        
    except Exception as e:
        provider.error(f"Failed to fetch images: {str(e)}")
        return []


def get_image_list_openverse(
    keyword: str,
    config: ImageConfig,
    num_images: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Get a list of images from Openverse API, filtered by orientation.

    Args:
        keyword (str): Search keyword for images.
        config (ImageConfig): Image configuration settings.
        num_images (Optional[int]): Number of images to fetch (max 20 per page).

    Returns:
        List[Dict[str, str]]: List of image data dictionaries.
    """
    if not config.enable_image_generation:
        provider.debug("Image generation is disabled")
        return []

    # Validate inputs
    if not keyword:
        provider.error("Keyword cannot be empty")
        return []
    if num_images > 20:
         count = 20
    else:
         count = num_images # Openverse API limits to 20 per page

    provider.info(f"Fetching up to {count} images for keyword: {keyword}")

    url = "https://api.openverse.engineering/v1/images"
    params = {
        "q": keyword,
        "page_size": count,
        "page": 1  # Start with page 1
    }

    results = []
    try:
        while len(results) < count:
            provider.debug(f"Sending request to Openverse API with params: {params}")

            def _make_api_call():
                response = requests.get(url, params=params)
                response.raise_for_status()
                return response.json()

            # Use rate limiter if enabled
            if hasattr(config, 'enable_rate_limiting') and config.enable_rate_limiting and openverse_rate_limiter:
                provider.debug("Using rate limiter for Openverse API call")
                data = openverse_rate_limiter.execute_with_rate_limit(_make_api_call)
            else:
                data = _make_api_call()

            if not data.get("results", []):
                provider.info("No more results available")
                break

            total_results = data.get("result_count", 0)
            provider.debug(f"Found {total_results} total results on page {params['page']}")

            for photo in data.get("results", []):
                if len(results) >= count:
                    break

                width = photo.get("width", 0)
                height = photo.get("height", 0)
                is_landscape = config.orientation == "landscape"

                # Check orientation based on dimensions
                if width == 0 or height == 0:
                    # Dimensions missing; try downloading the image
                    try:
                        response = requests.get(photo["url"], timeout=5)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                        width, height = image.size
                    except Exception as e:
                        provider.warning(f"Failed to fetch image dimensions for {photo['id']}: {str(e)}")
                        continue  # Skip if image download fails
                    
                # Skip square images
                if width == height:
                    continue

                # Filter based on orientation
                if is_landscape and width <= height:
                    continue  # Skip non-landscape images
                if not is_landscape and height <= width:
                    continue  # Skip non-portrait images

                results.append({
                    "id": photo["id"],
                    "url": photo["url"],
                    "download_url": photo["url"],
                    "alt_description": photo.get("title", keyword),
                    "width": width,
                    "height": height,
                    "photographer": photo.get("creator", "Unknown"),
                    "photographer_url": photo.get("creator_url", "")
                })
                # provider.debug(f"Processed image {photo['id']} by {photo.get('creator')}")

            # Move to next page if needed
            params["page"] += 1
            if len(results) < count and total_results <= params["page"] * count:
                provider.info("No more pages available")
                break

        provider.success(f"Found {len(results)} images for keyword: {keyword}")
        return results

    except Exception as e:
        provider.error(f"Failed to fetch images: {str(e)}")
        return []

def get_image_list_pexels(
    keyword: str,
    config: ImageConfig,
    num_images: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Get a list of images from Pexels API.
    """
    if not config.enable_image_generation:
        provider.debug("Image generation is disabled")
        return []

    provider.info(f"Fetching {num_images} images for keyword: {keyword}")

    url = "https://api.pexels.com/v1/search"
    params = {
        "query": keyword,
        "per_page": num_images,
        "orientation": config.orientation
    }

    headers = {
        "Authorization": config.pexels_api_key
    }

    try:
        provider.debug(f"Sending request to Pexels API with params: {params}")

        def _make_api_call():
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()

         # Use rate limiter if enabled
        if hasattr(config, 'enable_rate_limiting') and config.enable_rate_limiting and pexels_rate_limiter:
            provider.debug("Using rate limiter for Pexels API call")
            data = pexels_rate_limiter.execute_with_rate_limit(_make_api_call)
        else:
            data = _make_api_call()

        results = []
        for photo in data.get("photos", []):
            results.append({
                "id": str(photo["id"]),
                "url": photo["src"]["large"],
                "download_url": photo["src"]["original"],
                "alt_description": photo.get("alt", keyword),
                "width": photo.get("width", 0),
                "height": photo.get("height", 0),
                "photographer": photo["photographer"],
                "photographer_url": photo["photographer_url"]
            })

        provider.success(f"Found {len(results)} images for keyword: {keyword}")
        return results

    except Exception as e:
        provider.error(f"Failed to fetch images: {str(e)}")
        return []
    
def clean_and_prepare_folder(folder: str) -> None:
    """
    Clean up a folder (delete it if it exists) and recreate it.
    """
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    
    
def generate_ai_image_list(
    keyword: str,
    config: ImageConfig,
    num_images: Optional[int] = None,
    count_supplied:int = 0
) -> List[Dict[str, str]]:
    """
    Generate a list of AI-generated images using the Hugging Face API.

    Args:
        keyword (str): The subject for image generation.
        config (ImageConfig): Configuration containing API keys and model info.
        num_images (Optional[int]): Number of images to generate.
        hugging_face_rate_limiter: Optional rate limiter for API calls.

    Returns:
        List[Dict[str, str]]: List of generated image metadata.
    """
    if not config.enable_image_generation:
        provider.debug("AI image generation is disabled.")
        return []

    if not num_images or num_images <= 0:
        provider.debug("No valid num_images specified. Returning empty list.")
        return []

    folder = 'ai_images'
    clean_and_prepare_folder(folder)

    provider.info(f"Generating {num_images} AI images for keyword: {keyword}")

    images: List[Dict[str, str]] = []
    count = count_supplied + 1

    for i in range(num_images):
        
        try:
            prompt = get_prompt(keyword, count)
            provider.debug(f"Sending request to Hugging Face with prompt: {prompt}")

            # API call wrapped in function for rate limiter
            def make_api_call():
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{config.huggingface_model}",
                    headers={
                        "Authorization": f"Bearer {config.huggingface_api_key}",
                        "Accept": "image/png"
                    },
                    json={
                        "inputs": prompt,
                        "options": {"wait_for_model": True}
                    }
                )
                if response.status_code != 200:
                    provider.error(
                        f"API error: {response.status_code} {response.reason}. "
                        f"Response content: {response.text[:200]}"
                    )
                    response.raise_for_status()
                return response

            # Execute API call (with rate limiter if provided)
            data = (
                hugging_face_rate_limiter.execute_with_rate_limit(make_api_call)
                if hugging_face_rate_limiter
                else make_api_call()
            )

            image_data = data.content
            if not image_data:
                provider.warning("No image data returned from Hugging Face.")
                continue

          
            # if it's already binary PNG
            image_bytes = image_data

            filename = f"temp_image_{count}.png"
            ai_file = os.path.join(folder, filename)

            with open(ai_file, "wb") as f:
                f.write(image_bytes)

            # Split prompt for alt text
            prompt_used = prompt.split(",")
            prompt_used[0] = "A" if prompt_used else "AI Image"

            image_info = {
                "id": f"ai_{count}",
                "url": ai_file,
                "thumb": ai_file,
                "alt": " ".join(prompt_used),
                "photographer": "AI Generator",
                "photographer_url": "https://huggingface.co"
            }

            images.append(image_info)
            provider.debug(f"Generated image saved as {ai_file}")
            
            count += 1


        except Exception as e:
            provider.error(f"Error generating AI image {i}: {str(e)}")

    provider.info(f"Successfully generated {len(images)} AI images.")
    return images
 


def get_image_list_pixabay(
    keyword: str,
    config: ImageConfig,
    num_images: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Get a list of images from Pixabay API.
    """
    if not config.enable_image_generation:
        provider.debug("Image generation is disabled")
        return []

    provider.info(f"Fetching {num_images} images for keyword: {keyword}")

    url = "https://pixabay.com/api/"
    orientation = "horizontal" if config.orientation == "landscape" else "vertical"
    params = {
        "key": config.pixabay_api_key,
        "q": keyword,
        "per_page": num_images,
        "image_type": "photo",
        "safesearch": "true",
        "orientation": orientation
    }

    try:
        provider.debug(f"Sending request to Pixabay API with params: {params}")

        def _make_api_call():
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()

         # Use rate limiter if enabled
        if hasattr(config, 'enable_rate_limiting') and config.enable_rate_limiting and pixabay_rate_limiter:
            provider.debug("Using rate limiter for Pixabay API call")
            data = pixabay_rate_limiter.execute_with_rate_limit(_make_api_call)
        else:
            data = _make_api_call()

        results = []
        for photo in data.get("hits", []):
            results.append({
                "id": str(photo["id"]),
                "url": photo["largeImageURL"],
                "download_url": photo["largeImageURL"],
                "alt_description": photo.get("tags", keyword),
                "width": photo.get("imageWidth", 0),
                "height": photo.get("imageHeight", 0),
                "photographer": photo.get("user", "Unknown"),
                "photographer_url": f"https://pixabay.com/users/{photo.get('user', '')}-{photo.get('user_id', '')}/"
            })

        provider.success(f"Found {len(results)} images for keyword: {keyword}")
        return results

    except Exception as e:
        provider.error(f"Failed to fetch images: {str(e)}")
        return []


def download_image(url: str, save_path: str = None, compress: bool = False, quality: int = 70) -> str:
    """
    Download an image from URL and save it.
    
    Args:
        url (str): Image URL
        save_path (str, optional): Path to save the image. If None, uses OS temp directory.
        compress (bool): Whether to compress the image after downloading
        quality (int): Compression quality (0-100, higher is better quality)
    Returns:
        str: Path to the saved image
    """
    try:
        provider.debug(f"Downloading image from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # If no save path provided, create one in the OS temp directory
        if not save_path:
            # Get file extension from URL or default to .jpg
            file_ext = os.path.splitext(url.split('?')[0])[1] or '.jpg'
            # Create a temporary file with the correct extension
            fd, save_path = tempfile.mkstemp(suffix=file_ext)
            os.close(fd)  # Close the file descriptor
        
        provider.debug(f"Saving image to: {save_path}")
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Apply compression if enabled
        if compress and PIL_AVAILABLE:
            provider.debug(f"Compressing downloaded image with quality: {quality}")
            compress_image(save_path, quality)
                
        provider.success(f"Image downloaded successfully to: {save_path}")
        return save_path
    except Exception as e:
        provider.error(f"Error downloading image: {str(e)}")
        return ""

def get_image_list_from_stock_sources(
    keyword: str,
    config: ImageConfig,
    total_images: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Fetch image list from the selected stock image source.

    Args:
        keyword (str): Search term
        config (ImageConfig): Configuration object
        total_images (int, optional): Number of images to fetch
    
    Returns:
        List[Dict[str, str]]: List of image data dictionaries
    """
    images = []
    img = config.stock_primary_source.lower() 
    sourceUsed = ""

    if img == "unsplash":
        images = get_image_list_unsplash(keyword, config, total_images)
        sourceUsed = "Unsplash"
    elif img == "openverse":
        images = get_image_list_openverse(keyword, config, total_images)
        sourceUsed = "Openverse"
    elif img == "pexels":
        images = get_image_list_pexels(keyword, config, total_images)
        sourceUsed = "Pexels"
    elif img == "pixabay":
        images = get_image_list_pixabay(keyword, config, total_images)
        sourceUsed = "Pixabay"
    else:
        provider.warning(f"Unsupported image source: {config.stock_primary_source}")
        sourceUsed = "Unknown"
    
    return images,sourceUsed,

def process_body_image(
    image_data: Dict[str, str],
    keyword: str,
    sourceUsed: str,
    index: int,
    save_dir: str = None,
    timestamp: str = None,
    alignment: str = "aligncenter",
    compress: bool = False,
    compression_quality: int = 70
) -> Optional[Dict[str, str]]:
    """
    Process body image for article.
    
    Args:
        image_data (Dict[str, str]): Image metadata
        keyword (str): Article keyword
        index (int): Image index
        save_dir (str, optional): Directory to save images. If None, uses OS temp directory.
        timestamp (str, optional): Timestamp for unique filenames
        alignment (str): Image alignment class (aligncenter, alignleft, alignright)
        compress (bool): Whether to compress the image
        compression_quality (int): Compression quality (0-100)
    Returns:
        Optional[Dict[str, str]]: Processed image data or None
    """
    provider.debug(f"Processing body image {index} for keyword: {keyword}")
    
    try:
        # Download the image to temp directory
        image_url = image_data.get("url")
        if not image_url:
            provider.error(f"Missing URL for body image {index}")
            return None
            
        # Return image data for WordPress
        processed_data = {
            "file": image_data.get("url"),  # Local file path for WordPress upload
            "url": image_data.get("url"),   # Original URL for markdown display
            "alt": image_data.get("alt") or f"{keyword} {index}",
            "caption": f"Photo by {image_data.get('photographer', 'Unknown')} on {sourceUsed}",
            "photographer": image_data.get("photographer", "Unknown"),
            "photographer_url": image_data.get("photographer_url", ""),
            "alignment": alignment  # Add alignment class
        }
        
        if not processed_data["file"] or not processed_data["url"]:
            provider.error(f"Missing required URLs for body image {index}")
            return None
            
        # provider.debug(f"Processed body image {index} with URL: {processed_data['url']} and alignment: {alignment}")
        return processed_data
        
    except Exception as e:
        provider.error(f"Error processing body image {index}: {str(e)}")
        return None

def process_feature_image(
    image_data: Dict[str, str],
    keyword: str,
    sourceUsed: str,
    save_dir: str = None,
    timestamp: str = None,
    alignment: str = "aligncenter",
    compress: bool = False,
    compression_quality: int = 70
) -> Optional[Dict[str, str]]:
    """
    Process feature image for article.
    
    Args:
        image_data (Dict[str, str]): Image metadata
        keyword (str): Article keyword
        save_dir (str, optional): Directory to save images. If None, uses OS temp directory.
        timestamp (str, optional): Timestamp for unique filenames
        alignment (str): Image alignment class (aligncenter, alignleft, alignright)
        compress (bool): Whether to compress the image
        compression_quality (int): Compression quality (0-100)
    Returns:
        Optional[Dict[str, str]]: Processed image data or None
    """
    provider.debug(f"Processing feature image for keyword: {keyword}")
    
    try:
        # Download the image to temp directory
        image_url = image_data.get("url")
        if not image_url:
            provider.error("Missing URL for feature image")
            return None
            
        # Return image data for WordPress
        processed_data = {
            "file": image_data.get("url"),  # Local file path for WordPress upload
            "url": image_data.get("url"),   # Original URL for markdown display
            "alt": image_data.get("alt") or f"Featured image for {keyword}",
            "caption": f"Photo by {image_data.get('photographer', 'Unknown')} on {sourceUsed}",
            "photographer": image_data.get("photographer", "Unknown"),
            "photographer_url": image_data.get("photographer_url", ""),
            "alignment": alignment  # Add alignment class
        }
        
        if not processed_data["file"] or not processed_data["url"]:
            provider.error("Missing required URLs for feature image")
            return None
            
        provider.debug(f"Processed feature image with URL: {processed_data['url']} and alignment: {alignment}")
        return processed_data
                
    except Exception as e:
        provider.error(f"Error processing feature image: {str(e)}")
        return None
    
    
# def get_image_list_by_source(
#     keyword: str,
#     config: ImageConfig,
#     total_images: Optional[int] = None
# ) -> List[Dict[str, str]]:
#     """
#     Fetch image list from the selected image source.

#     Args:
#         keyword (str): Search term
#         config (ImageConfig): Configuration object
#         total_images (int, optional): Number of images to fetch
    
#     Returns:
#         List[Dict[str, str]]: List of image data dictionaries
#     """
#     images = []

#     if config.image_src == "Unsplash":
#         images = get_image_list_unsplash(keyword, config, total_images)

#     elif config.image_src == "Openverse":
#         images = get_image_list_openverse(keyword, config, total_images)

#     elif config.image_src == "Pexels":
#         images = get_image_list_pexels(keyword, config, total_images)

#     elif config.image_src == "Pixabay":
#         images = get_image_list_pixabay(keyword, config, total_images)

#     else:
#         provider.error(f"Unsupported image source: {config.image_src}")
    
#     return images



def fetch_image(
    config: ImageConfig,
    image_source: str, 
    primary_source: str,
    query: str, 
    num_of_image: int,
    use_secondary_library: bool = True,
    count:int = 0
    ) -> Optional[Dict]:
    """
    Fetches an image (or list of images) based on user-defined source and query.

    This method supports both AI-generated images and stock photo sources
    (Unsplash, Pexels, Pixabay, Openverse). It prioritizes a primary source
    and optionally falls back to other supported stock sources if no results are found.

    Args:
        config (ImageConfig): Configuration object containing API keys and supported sources.
        image_source (str): Either "stock_images" or "imageai" to determine fetch strategy.
        primary_source (str): Preferred source (e.g., "unsplash", "pexels", "pixabay", "openverse").
        query (str): Search keyword for image retrieval.
        num_of_image (int): Number of images to retrieve from source.
        use_secondary_library (bool, optional): Whether to attempt fallback sources. Defaults to True.

    Returns:
        Optional[Dict]: A dictionary containing image data (URL, source, license, etc.) 
        or None if no suitable image is found.
    """
    
    if image_source.lower() == "imageai":
        return generate_ai_image_list(query,config,num_images=num_of_image),"AI"

    if primary_source.lower() not in  ["unsplash", "pexels", "pixabay", "openverse"]:
       provider.error("Invalid primary image source")
       config.stock_primary_source = "unsplash"


    result = get_image_list_from_stock_sources(keyword=query,config=config,total_images=num_of_image)
    
    if result or not use_secondary_library:
        return result

    # Fallback
    for source in ["unsplash", "pexels", "pixabay", "openverse"]:
        if source == primary_source:
            continue
        result = get_image_list_from_stock_sources(keyword=query,config=config,total_images=num_of_image)
        if result:
            return result

    return None

def generate_image_description(
    processor: CLIPProcessor,
    model: CLIPModel, 
    image_path: str,
    keyword: str
    ) -> str:
    """
    Generates a description or relevance score for an image based on a keyword using the CLIP model.

    Parameters:
    - processor (CLIPProcessor): The CLIP processor used to preprocess the image and text.
    - model (CLIPModel): The pre-trained CLIP model for computing image-text similarity.
    - image_path (str): Path to the image file to be evaluated.
    - keyword (str): The text prompt or keyword to compare the image against.

    Returns:
    - str: A description, relevance score, or label indicating how well the image matches the keyword.
    """
    start_time = time.time()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    
    # Generate image embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    provider.info(f"detected {keyword}")
    
    # Define candidate text descriptions for CLIP to score
    candidate_texts = [
        f"This is about {keyword}",
        f"This shows {keyword}",
        f"This captures {keyword}",
        f"This reflects {keyword}",
        f"This features {keyword}",
        f"This highlights {keyword}",
        f"This explores {keyword}",
        f"This represents {keyword}",
        f"This emphasizes {keyword}",
        f"This illustrates {keyword}",
        f"This showcases {keyword}",
        f"This conveys {keyword}",
        f"This expresses {keyword}",
        f"This depicts {keyword}",
        f"This focuses on {keyword}",
        f"This portrays {keyword}",
        f"This revolves around {keyword}",
        f"This presents {keyword}",
        f"This centers on {keyword}",
        f"This discusses {keyword}",
        f"This investigates {keyword}",
        f"This examines {keyword}",
        f"This delves into {keyword}",
        f"This narrates {keyword}",
        f"This explains {keyword}",
        f"This visualizes {keyword}",
        f"This demonstrates {keyword}",
        f"This uncovers {keyword}",
        f"This manifests {keyword}",
        f"This reveals {keyword}",
    ]
    
    # Prepare text inputs for CLIP
    text_inputs = processor(text=candidate_texts, return_tensors="pt", padding=True)
    
    # Generate text embeddings and compute similarity
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        similarities = torch.nn.functional.cosine_similarity(image_features, text_features)
    
    # Select the most similar description
    best_description_idx = similarities.argmax().item()
    description = candidate_texts[best_description_idx]
    
    # Clean description for alt text
    alt_text = description.capitalize()
    if not alt_text.endswith('.'):
        alt_text += '.'
    
    provider.info(f"Alt text generation time: {time.time() - start_time:.2f} seconds")
    return alt_text


def generate_filename(
    alt_text: str, 
    tmp_file: str = os.path.join('tmp_images', 'temp_image.jpg')
    ) -> str:
    """Check if temporary file exists, rename it based on alt text, and return the new file path."""
    
    # Convert alt text to lowercase, remove special characters, replace spaces with hyphens
    filename = re.sub(r'[^\w\s-]', '', alt_text.lower())
    filename = re.sub(r'\s+', '-', filename).strip('-')
    new_filename = f"{filename}.jpg"
    
    # Generate the fully qualified path for the new file
    new_file_path = os.path.join('tmp_images', new_filename)
    
    # Check if the temporary file exists and rename it
    if os.path.exists(tmp_file):
        try:
            os.rename(tmp_file, new_file_path)
            provider.info(f"Renamed {tmp_file} to {new_file_path}")
            return new_file_path
        except OSError as e:
            provider.error(f"Error renaming file: {e}")
            return tmp_file  
    else:
        provider.error (f"Temporary file {tmp_file} does not exist")
        return tmp_file  


def generate_ai_filename(
    alt_text: str, 
    tmp_file: str
    ) -> str:
    """Check if temporary file exists, rename it based on alt text, and return the new file path."""
    
    # Convert alt text to lowercase, remove special characters, replace spaces with hyphens
    filename = re.sub(r'[^\w\s-]', '', alt_text.lower())
    filename = re.sub(r'\s+', '-', filename).strip('-') 
    count = tmp_file.split(".")[0].split("_")[3]

    new_filename = f"{filename}-{count}.jpg"
    
    # Generate the fully qualified path for the new file
    new_file_path = os.path.join('ai_images', new_filename)
    
    
    # Check if the temporary file exists and rename it
    if os.path.exists(tmp_file):
        try:
            os.rename(tmp_file, new_file_path)
            provider.info(f"Renamed {tmp_file} to {new_file_path}")
            return new_file_path
        except OSError as e:
            provider.error(f"Error renaming file: {e}")
            return tmp_file  
    else:
        provider.error (f"Temporary file {tmp_file} does not exist")
        return tmp_file  


def image_randomizer(input_list):
    """
    Picks a random index from a list based on its length.
    
    Args:
        input_list (list): The input list.
        
    Returns:
        int: A random index (0 to len(input_list) - 1), or None if the list is empty.
    """
    if not input_list:  # Check if list is empty
        return None
    return random.randint(0, len(input_list) - 1)

def randomize_image_selection(lst):
    """
    Returns a new randomly shuffled list using random.shuffle.
    
    Args:
        lst: List to be shuffled
    
    Returns:
        A new shuffled list
    """
    new_list = lst.copy()
    random.shuffle(new_list)
    return new_list

 
# def get_article_images(
#     keyword: str,
#     config: ImageConfig,
#     num_images: Optional[int] = None,
#     save_dir: str = None,
#     timestamp: str = None
# ) -> Tuple[Optional[Dict[str, str]], List[Dict[str, str]]]:
#     """
#     Get images for an article.
    
#     Args:
#         keyword (str): Article keyword
#         config (ImageConfig): Image configuration
#         num_images (int, optional): Number of images to fetch
#         save_dir (str, optional): Directory to save images. If None, uses OS temp directory.
#         timestamp (str, optional): Timestamp for unique filenames
#     Returns:
#         Tuple[Optional[Dict[str, str]], List[Dict[str, str]]]: Feature image and body images
#     """
#     try:
#         # Get images from any source
#         images = get_image_list_by_source(keyword, config, num_images)
#         if not images:
#             provider.warning(f"No images found for keyword: {keyword}")
#             return None, []
            
#         # Process feature image
#         feature_image = None
#         if images:
#             feature_image = process_feature_image(
#                 images[0], 
#                 keyword, 
#                 save_dir, 
#                 timestamp,
#                 alignment=config.alignment,
#                 compress=config.enable_image_compression,
#                 compression_quality=config.compression_quality
#             )
            
#         # Create a set to track used images for duplicate prevention
#         used_image_ids = set()
#         if feature_image:
#             # Add feature image ID to used images
#             for image in images:
#                 if image.get("url") == feature_image["url"]:
#                     used_image_ids.add(image.get("id", ""))
#                     break
                    
#         # Process body images
#         body_images = []
#         if len(images) > 1:
#             for i, image_data in enumerate(images[1:], 1):
#                 # Skip duplicate images if prevention is enabled
#                 if config.prevent_duplicate_images and image_data.get("id", "") in used_image_ids:
#                     provider.debug(f"Skipping duplicate image: {image_data.get('id', '')}")
#                     continue
                    
#                 body_image = process_body_image(
#                     image_data, 
#                     keyword, 
#                     i, 
#                     save_dir, 
#                     timestamp,
#                     alignment=config.alignment,
#                     compress=config.enable_image_compression,
#                     compression_quality=config.compression_quality
#                 )
                
#                 if body_image:
#                     body_images.append(body_image)
#                     # Add to used images set
#                     used_image_ids.add(image_data.get("id", ""))
                    
#         return feature_image, body_images
        
#     except Exception as e:
#         provider.error(f"Error getting article images: {str(e)}")
#         return None, []

def compress_image(image_path: str, quality: int = 70) -> str:
    """
    Compress an image using PIL/Pillow to reduce file size.
    
    Args:
        image_path (str): Path to the image file
        quality (int): Compression quality (0-100, higher is better quality)
    Returns:
        str: Path to the compressed image (same as input if compression failed)
    """
    if not PIL_AVAILABLE:
        provider.warning("PIL/Pillow not available. Skipping image compression.")
        return image_path
        
    if not os.path.exists(image_path):
        provider.error(f"Image file not found: {image_path}")
        return image_path
        
    try:
        provider.debug(f"Compressing image: {image_path} with quality {quality}")
        
        # Get file extension and create output path
        file_name, file_ext = os.path.splitext(image_path)
        compressed_path = f"{file_name}_compressed{file_ext}"
        
        # Open image and compress
        with Image.open(image_path) as img:
            # Preserve image format
            img_format = img.format if img.format else 'JPEG'
            
            # Save with compression
            img.save(
                compressed_path,
                format=img_format,
                optimize=True,
                quality=quality
            )
            
        # Check if compression was effective
        original_size = os.path.getsize(image_path)
        compressed_size = os.path.getsize(compressed_path)
        
        if compressed_size < original_size:
            reduction = (1 - (compressed_size / original_size)) * 100
            provider.success(f"Image compressed successfully: {reduction:.1f}% reduction ({original_size} -> {compressed_size} bytes)")
            
            # Replace original with compressed version
            os.replace(compressed_path, image_path)
            return image_path
        else:
            # If compression didn't reduce size, delete compressed file and keep original
            provider.debug("Compression didn't reduce file size. Keeping original.")
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            return image_path
            
    except Exception as e:
        provider.error(f"Error compressing image: {str(e)}")
        # Return original path if compression fails
        return image_path
    
    
import random

def get_prompt(keyword: str, count: int = None) -> str:
    prompts = [
        # blending realistic camera/lens effects with cinematic moods
        "Generate a 1000x1000, photorealistic wide‑angle dusk scene of: {keyword}, with cinematic depth of field, global illumination, subtle fog, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic macro close‑up of: {keyword}, glowing under neon night lights, with realistic surface imperfections and specular highlights, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic aerial cityscape featuring: {keyword}, during golden hour with physically‑based reflections, cinematic shadows, and lens flares, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic over‑the‑shoulder shot of: {keyword}, inside a misty forest with volumetric light rays, realistic depth haze, and soft motion blur, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic underwater view of: {keyword}, shimmering caustics, cinematic color grading, and subsurface scattering, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic rainy street scene featuring: {keyword}, with real‑world puddle reflections, HDR lighting, cinematic bokeh, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic close‑up render of: {keyword}, in a futuristic room with PBR materials, soft rim lighting, and micro‑surface detail, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic desert sunset frame of: {keyword}, warm global illumination, natural shadow gradients, cinematic lens glow, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic snowy landscape with: {keyword}, light scattering through fog, realistic icy textures, cinematic cold highlights, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic ancient ruins at sunrise featuring: {keyword}, with ray‑traced shadows, cinematic grain, and realistic weathering on surfaces, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic rooftop shot of: {keyword}, neon glow, chromatic aberration on edges, cinematic composition, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic immersive low‑angle twilight view of: {keyword}, with HDR tonemapping, cinematic depth of field, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic stormy seascape with: {keyword}, realistic foam simulation, soft bloom lighting, cinematic atmosphere, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic lantern‑lit alley scene with: {keyword}, dynamic shadows, photorealistic roughness maps, cinematic mood, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic glass‑shard perspective of: {keyword}, with depth‑based refractions, lens distortion, cinematic light play, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic hyper‑detailed aerial night view of: {keyword}, with global illumination and city bloom effects, cinematic tone, high‑resolution.",
        "Generate a 1000x1000, photorealistic over‑the‑shoulder cathedral shot of: {keyword}, with realistic soft shadows, light scattering through stained glass, cinematic vibe, ultra‑detailed.",
        "Generate a 1000x1000, photorealistic mountain pass at golden hour featuring: {keyword}, volumetric mist, PBR rock textures, cinematic gradient lighting, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic cyberpunk market scene with: {keyword}, wet asphalt reflections, subsurface scattering on materials, cinematic neon hues, ultra‑detailed, high‑resolution.",
        "Generate a 1000x1000, photorealistic rain‑streaked window shot of: {keyword}, with realistic depth of field, chromatic aberration, soft lens bloom, ultra‑detailed, high‑resolution."
    ]
    
    extra_prompts = [
        "Generate a 1000x1000, photorealistic handheld-style shot of: {keyword}, with natural motion blur, HDR tonemapping, cinematic contrast, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic telephoto lens capture of: {keyword}, shallow depth of field, film grain and soft vignetting, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic rainy evening close-up of: {keyword}, water droplets on lens, cinematic reflections, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic high-speed capture of: {keyword}, with motion trails, realistic lighting physics, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic fog-drenched mountain view featuring: {keyword}, cinematic depth haze, lens flares, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic vintage film look of: {keyword}, light leaks, cinematic tone mapping, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic drone-style perspective of: {keyword}, sharp dynamic shadows, cinematic color grading, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic urban alleyway shot of: {keyword}, puddles with ray-traced reflections, cinematic bloom, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic golden-hour street scene with: {keyword}, long soft shadows, chromatic aberration, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic backlit silhouette of: {keyword}, atmospheric fog and cinematic flares, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic panoramic concept of: {keyword}, with volumetric god rays, HDR light balance, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic filmic render of: {keyword}, with realistic subsurface scattering, cinematic imperfections, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic top-down architectural shot of: {keyword}, soft ambient occlusion, cinematic sharpness, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic atmospheric nightscape of: {keyword}, starry sky lighting and cinematic tones, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic low‑light interior shot of: {keyword}, with candle flicker effects, realistic shadows, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic sci-fi laboratory environment with: {keyword}, PBR metallic textures, cinematic gleam, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic shallow-focus portrait of: {keyword}, creamy bokeh, cinematic highlights, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic rainy forest clearing featuring: {keyword}, glistening foliage and depth haze, cinematic mood, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic aerial dusk panorama of: {keyword}, with HDR exposure and soft glow, ultra-detailed, high-resolution.",
        "Generate a 1000x1000, photorealistic extreme close-up texture of: {keyword}, showing micro-details, cinematic grading, ultra-detailed, high-resolution."
    ]
    
    prompts.extend(extra_prompts)

    if count is None:
        count = random.randint(0, len(prompts) - 1)
    elif count < 0 or count >= len(prompts):
        raise IndexError(f"Count must be between 0 and {len(prompts)-1}")

    return prompts[count].format(keyword=keyword)
