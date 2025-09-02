#!/usr/bin/env python3
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import json
import requests
import base64
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console
console = Console()

def get_wordpress_credentials(website_name: str, username: str, app_pass: str):
    """
    Get WordPress API credentials.
    
    Args:
        website_name (str): WordPress site URL
        username (str): WordPress username
        app_pass (str): WordPress application password
        
    Returns:
        Dict: WordPress credentials with URL and headers
    """
    # Format website URL
    if not website_name.startswith(('http://', 'https://')):
        website_name = f"https://{website_name}"
        
    # Remove trailing slash if present
    if website_name.endswith('/'):
        website_name = website_name[:-1]
        
    # Create credentials
    credentials = {
        'json_url': f'{website_name}/wp-json/wp/v2',
        'headers': {
            'Authorization': 'Basic ' + base64.b64encode(
                f'{username}:{app_pass}'.encode()
            ).decode('utf-8')
        }
    }
    return credentials

def list_wordpress_users():
    """
    List all WordPress users from the site configured in .env
    """
    # Get WordPress credentials from .env
    website_name = os.getenv('WP_WEBSITE_NAME')
    username = os.getenv('WP_USERNAME')
    app_pass = os.getenv('WP_APP_PASS')
    
    if not all([website_name, username, app_pass]):
        console.print("[red]Error:[/] WordPress credentials not found in .env file")
        console.print("Please ensure you have set the following variables in your .env file:")
        console.print("- WP_WEBSITE_NAME")
        console.print("- WP_USERNAME")
        console.print("- WP_APP_PASS")
        return
    
    try:
        # Get WordPress credentials
        credentials = get_wordpress_credentials(website_name, username, app_pass)
        
        # Fetch WordPress users
        console.print("[yellow]Fetching WordPress users...[/]")
        response = requests.get(
            f"{credentials['json_url']}/users",
            headers=credentials['headers'],
            params={"per_page": 100}  # Increase if you have more than 100 users
        )
        
        response.raise_for_status()
        users = response.json()
        
        if not users:
            console.print("[yellow]No WordPress users found[/]")
            return
        
        # Create and populate a table with user information
        table = Table(title="WordPress Users")
        table.add_column("ID", style="cyan")
        table.add_column("Username", style="green")
        table.add_column("Name", style="blue")
        table.add_column("Role", style="magenta")
        
        for user in users:
            table.add_row(
                str(user['id']),
                user['slug'],
                user['name'],
                ', '.join(user['roles']) if 'roles' in user else 'N/A'
            )
        
        console.print(table)
        
        # Print usage instructions
        console.print("\n[bold green]How to use a custom WordPress author:[/]")
        console.print("1. [yellow]Add to .env file:[/] WP_CUSTOM_AUTHOR=<author_id>")
        console.print("2. [yellow]Or run with environment variable:[/] WP_CUSTOM_AUTHOR=<author_id> python main.py <csv_file>")
        console.print("\nDefault author ID is set to", os.getenv('WP_AUTHOR', '1'), "(from WP_AUTHOR in .env)")
        
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error:[/] Failed to fetch WordPress users: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                console.print(f"[red]API Error:[/] {json.dumps(error_data, indent=2)}")
            except:
                console.print(f"[red]Response:[/] {e.response.text}")
    except Exception as e:
        console.print(f"[red]Error:[/] {str(e)}")

if __name__ == "__main__":
    console.print("[bold blue]WordPress User Listing Tool[/]", 
                 "Get a list of available WordPress users that can be used as authors")
    list_wordpress_users()
