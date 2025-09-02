#!/usr/bin/env python3
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

"""
Image API Connection Test Script

This script tests the Image API connection using the credentials in the .env file.
"""

import os
import sys
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from ..article_generator.image_handler import get_image_list_by_source,ImageConfig

# Initialize console
console = Console()

def main():
    """Main function to test Image API connection."""
    console.print(
        Panel.fit(
            "[bold blue]Image API Connection Test[/]",
            border_style="blue",
        )
    )
    
    # Load environment variables
    load_dotenv()
    
    # Get Unsplash API key
    img_api_key = os.getenv('IMAGE_API_KEY')
    
    # Check if API key is available
    if not img_api_key:
        console.print("[bold red]Error:[/] Image API key is missing in .env file")
        return 1
    
    config = ImageConfig(
        image_src="Unplash",
        enable_image_generation=True,
        randomize_images=False,
        max_number_of_images=5,
        prevent_duplicate_images=True,
        image_api_key=os.environ.get("IMAGE_API_KEY", "")
    )

    
    # Test connection
    console.print("[yellow]Testing Image API connection...[/]")
    try:
        images = get_image_list_by_source("Dog",config=config,total_images=5)
        
        if len(images) != 0:
            console.print(f"[bold green]Success![/] Connected to {config.image_src} API")
            console.print(f'Successfully retrieved {len(images)} images')
            return 0
        else:
            console.print(f"[bold red]Error:[/] Failed to connect to {config.image_src}. Check api configs")
            return 1
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
