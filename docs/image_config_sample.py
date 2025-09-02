# Image Configuration Sample for CopyscriptAI

# Add this code to your main.py file in script1 or script2 to customize image settings
# These settings should be added inside the Config initialization block

# For script1 (around line 80 in main.py):
config = Config(
    # ... other configuration settings ...
    
    # Image Settings - Feature Toggles
    enable_image_generation=True,  # Set to True to enable image generation
    add_image_into_article=True,   # Set to True to include images in articles
    
    # Image Settings - Basic Parameters
    randomize_images=True,         # Use random number of images between min and max
    max_number_of_images=5,        # Maximum number of images per article
    orientation="landscape",       # Image orientation: "landscape", "portrait", or "squarish"
    order_by="relevant",           # How to order images: "relevant" or "random"
    
    # Image Settings - Advanced Options
    image_alignment="aligncenter", # Image alignment: "aligncenter", "alignleft", or "alignright"
    
    # Image Settings - Compression (requires PIL/Pillow)
    enable_image_compression=True, # Enable image compression to reduce file sizes
    image_compression_quality=75,  # Compression quality (0-100, higher is better quality)
    
    # Image Settings - Duplicate Prevention
    prevent_duplicate_images=True, # Prevent duplicate images in the same article
    
    # ... other configuration settings ...
)

# For script2, the format is similar but used in its main.py
