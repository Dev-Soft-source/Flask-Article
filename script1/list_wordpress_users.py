#!/usr/bin/env python
# filepath: /home/abuh/Documents/Python/LLM_article_gen_2/scripts/script1/list_wordpress_users.py
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from rich.console import Console
from rich.table import Table
import os
from dotenv import load_dotenv
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

from article_generator.wordpress_handler import list_wordpress_users

# Initialize Rich console
console = Console()

def main():
    """
    List all WordPress users from the configured WordPress site.
    """
    try:
        # Get WordPress credentials from environment variables
        website_name = os.getenv('WP_WEBSITE_NAME')
        username = os.getenv('WP_USERNAME')
        app_pass = os.getenv('WP_APP_PASS')
        
        if not all([website_name, username, app_pass]):
            console.print("[bold red]Error:[/] WordPress credentials are not properly configured in .env file")
            console.print("Please ensure the following variables are set:")
            console.print("  - WP_WEBSITE_NAME: Your WordPress site domain")
            console.print("  - WP_USERNAME: Your WordPress username")
            console.print("  - WP_APP_PASS: Your WordPress application password")
            return 1
        
        # Get WordPress users
        console.print("[cyan]Fetching WordPress users...[/]")
        users = list_wordpress_users(website_name, username, app_pass)
        
        if not users:
            console.print("[bold yellow]No WordPress users found or unable to fetch users.[/]")
            return 1
        
        # Display users in a table
        table = Table(title="WordPress Users")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Slug", style="blue")
        table.add_column("Roles", style="magenta")
        
        for user in users:
            table.add_row(
                str(user['id']),
                user['name'],
                user['slug'],
                ", ".join(user['roles'])
            )
        
        console.print(table)
        
        # Display instructions for using a custom author
        console.print("\n[bold cyan]How to use a custom WordPress author:[/]")
        console.print("1. To set a custom author for all articles in a batch, use the WP_CUSTOM_AUTHOR environment variable:")
        console.print("   [yellow]export WP_CUSTOM_AUTHOR=2[/] (replace '2' with the desired author ID)")
        console.print("\n2. To use the default author (from .env file), don't set WP_CUSTOM_AUTHOR")
        
        return 0
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
