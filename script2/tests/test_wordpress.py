#!/usr/bin/env python3
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

"""
WordPress Connection Test Script

This script tests the WordPress connection using the credentials in the .env file.
"""

import os
import sys
import base64
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Initialize console
console = Console()

def main():
    """Main function to test WordPress connection."""
    console.print(
        Panel.fit(
            "[bold blue]WordPress Connection Test[/]",
            border_style="blue",
        )
    )
    
    # Load environment variables
    load_dotenv()
    
    # Get WordPress credentials
    WP_WEBSITE_NAME = os.getenv('WP_WEBSITE_NAME')
    WP_USERNAME = os.getenv('WP_USERNAME')
    wp_app_pass = os.getenv('WP_APP_PASS')
    
    # Check if credentials are available
    if not all([WP_WEBSITE_NAME, WP_USERNAME, wp_app_pass]):
        console.print("[bold red]Error:[/] WordPress credentials are missing in .env file")
        console.print(f"WP_WEBSITE_NAME: {'[green]Found[/]' if WP_WEBSITE_NAME else '[red]Missing[/]'}")
        console.print(f"WP_USERNAME: {'[green]Found[/]' if WP_USERNAME else '[red]Missing[/]'}")
        console.print(f"WP_APP_PASS: {'[green]Found[/]' if wp_app_pass else '[red]Missing[/]'}")
        return 1
    
    # Format website URL
    if not WP_WEBSITE_NAME.startswith(('http://', 'https://')):
        WP_WEBSITE_NAME = f"https://{WP_WEBSITE_NAME}"
        
    # Remove trailing slash if present
    if WP_WEBSITE_NAME.endswith('/'):
        WP_WEBSITE_NAME = WP_WEBSITE_NAME[:-1]
        
    # Create authentication token
    auth_token = base64.b64encode(f"{WP_USERNAME}:{wp_app_pass}".encode()).decode()
    
    # Test connection
    console.print("[yellow]Testing WordPress connection...[/]")
    try:
        response = requests.get(
            f"{WP_WEBSITE_NAME}/wp-json/wp/v2/users/me",
            headers={"Authorization": f"Basic {auth_token}"},
            timeout=10
        )
        
        if response.status_code == 200:
            user_data = response.json()
            console.print(f"[bold green]Success![/] Connected to WordPress as: {user_data.get('name', 'Unknown')}")
            console.print(f"User ID: {user_data.get('id', 'Unknown')}")
            console.print(f"User roles: {', '.join(user_data.get('roles', []))}")
            return 0
        else:
            console.print(f"[bold red]Error:[/] Failed to connect to WordPress. Status code: {response.status_code}")
            console.print(f"Response: {response.text}")
            return 1
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
