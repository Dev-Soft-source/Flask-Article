# Image Configuration Guide for CopyscriptAI

This guide explains how to customize image settings in your CopyscriptAI project.

## Overview

CopyscriptAI supports various image customization options:

1. **Image Alignment**: Control how images are aligned in your articles
2. **Image Compression**: Reduce image file sizes while maintaining quality
3. **Duplicate Prevention**: Avoid using the same image multiple times in one article

## How to Configure Image Settings

You can customize image settings by modifying the `Config` object in your `main.py` file:

### Basic Image Settings

```python
config = Config(
    # Enable/disable image features
    enable_image_generation=True,  # Master toggle for image generation
    add_image_into_article=True,   # Include images in the final article
    
    # Image quantity settings
    randomize_images=True,         # Use random number of images between min and max
    max_number_of_images=5,        # Maximum number of images per article
    
    # Image style settings
    orientation="landscape",       # Image orientation: "landscape", "portrait", "squarish"
    order_by="relevant",           # How to order images: "relevant" or "random"
)
```

### Advanced Image Settings

```python
config = Config(
    # ... basic settings ...
    
    # Image alignment
    image_alignment="aligncenter", # "aligncenter", "alignleft", or "alignright"
    
    # Image compression
    enable_image_compression=True, # Enable compression to reduce file sizes
    image_compression_quality=75,  # Quality (0-100, higher is better quality)
    
    # Duplicate prevention
    prevent_duplicate_images=True, # Prevent using same image multiple times
)
```

## Detailed Options

### Image Alignment

- `aligncenter`: Centers the image (default)
- `alignleft`: Aligns the image to the left
- `alignright`: Aligns the image to the right

### Image Compression

Image compression requires the PIL/Pillow library:

```bash
pip install Pillow
```

Compression settings:
- `enable_image_compression`: Turn compression on/off
- `image_compression_quality`: Quality level (0-100)
  - Higher values = better quality but larger file size
  - Lower values = smaller file size but lower quality
  - 70-80 is usually a good balance

The system will only keep the compressed version if it's smaller than the original.

### Duplicate Prevention

When `prevent_duplicate_images` is enabled, the system tracks which images have been used in the current article and ensures each image appears only once.

## Example Configuration

Here's a complete example with all image settings:

```python
config = Config(
    # ... other settings ...
    
    # Basic image settings
    enable_image_generation=True,
    add_image_into_article=True,
    randomize_images=True,
    max_number_of_images=6,
    orientation="landscape",
    order_by="relevant",
    
    # Advanced image settings
    image_alignment="aligncenter",
    enable_image_compression=True,
    image_compression_quality=75,
    prevent_duplicate_images=True,
    
    # ... other settings ...
)
```

This configuration will:
1. Enable image generation with 3-6 images per article
2. Center-align all images
3. Compress images with quality level 75
4. Prevent duplicate images in the same article

## Notes

- Image API key must be set in your `.env` file or passed to the config
- Make sure PIL/Pillow is installed if using compression
- These settings apply to both feature images and body images
