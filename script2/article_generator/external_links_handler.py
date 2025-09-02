# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from typing import List, Dict, Optional
from serpapi import GoogleSearch
from utils.rich_provider import provider
from config import Config
import traceback
import re

def generate_external_links_section(
    keyword: str,
    config: Config,
    output_format: str = 'wordpress'
) -> Optional[str]:
    """
    Generate a section with relevant external links using SerpAPI.
    
    Args:
        keyword (str): Search keyword
        config (Config): Configuration object containing API keys and settings
        output_format (str): Output format ('wordpress' or 'markdown')
    Returns:
        Optional[str]: Formatted external links section or None on error
    """
    if not config.add_external_links_into_article:
        provider.info("External links generation is disabled in configuration")
        return None
        
    provider.info(f"Generating external links section for keyword: {keyword}")
    
    # Try with a more specific search query first
    enhanced_keyword = f"{keyword} tips resources guide"
    provider.debug(f"Using enhanced search query: '{enhanced_keyword}'")
    
    try:
        # Try with enhanced keyword first
        external_links = _get_external_links(enhanced_keyword, config.serp_api_key, output_format)
        
        # If first attempt failed, try with original keyword
        if not external_links:
            provider.debug(f"First attempt failed. Trying with original keyword: '{keyword}'")
            external_links = _get_external_links(keyword, config.serp_api_key, output_format)
        
        # If still no results, try with a different query format
        if not external_links:
            fallback_keyword = f"best {keyword} guide"
            provider.debug(f"Second attempt failed. Trying with fallback keyword: '{fallback_keyword}'")
            external_links = _get_external_links(fallback_keyword, config.serp_api_key, output_format)
            
        # If all attempts failed, create a placeholder
        if not external_links:
            provider.warning("All attempts to generate external links failed. Creating placeholder.")
            if output_format == 'wordpress':
                external_links = "\n".join([
                    '<!-- wp:heading -->',
                    '<h2>External Resources</h2>',
                    '<!-- /wp:heading -->',
                    '<!-- wp:paragraph -->',
                    '<p>Additional resources related to this topic will be updated soon.</p>',
                    '<!-- /wp:paragraph -->'
                ])
            else:  # markdown
                external_links = "## External Resources\n\nAdditional resources related to this topic will be updated soon."
                
        provider.success(f"Generated external links section with length: {len(external_links) if external_links else 0}")
        return external_links
            
    except Exception as e:
        provider.error(f"Error generating external links: {str(e)}")
        provider.error(f"Traceback: {traceback.format_exc()}")
        # Return placeholder instead of None to ensure the section is included
        if output_format == 'wordpress':
            return "\n".join([
                '<!-- wp:heading -->',
                '<h2>External Resources</h2>',
                '<!-- /wp:heading -->',
                '<!-- wp:paragraph -->',
                '<p>Additional resources related to this topic will be updated soon.</p>',
                '<!-- /wp:paragraph -->'
            ])
        else:  # markdown
            return "## External Resources\n\nAdditional resources related to this topic will be updated soon."

def _get_external_links(keyword: str, serp_api_key: str, output_format: str = 'wordpress') -> Optional[str]:
    """
    Helper function to fetch and format external links for a keyword.
    
    Args:
        keyword (str): Search keyword
        serp_api_key (str): SerpAPI key
        output_format (str): Output format ('wordpress' or 'markdown')
    Returns:
        Optional[str]: Formatted external links section or None if no results
    """
    try:
        # Set up search parameters with improved settings
        params = {
            "engine": "google",
            "q": keyword,
            "api_key": serp_api_key,
            "gl": "us",      # Set to US results
            "hl": "en",      # Set to English
            "num": 10,       # Get more results to filter from
            "safe": "active", # Safe search
            "google_domain": "google.com",
            "no_cache": True  # Bypass cache for fresh results
        }
        
        # Perform search
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Check for organic_results and also other potentially useful sections
        links = []
        sources = []
        
        if "organic_results" in results:
            sources.append(("organic_results", results["organic_results"]))
        
        if "related_questions" in results:
            related_questions = []
            for q in results["related_questions"]:
                if q.get("link"):
                    related_questions.append({
                        "title": q.get("question", "Related Question"),
                        "link": q.get("link"),
                        "snippet": q.get("snippet", "")
                    })
            if related_questions:
                sources.append(("related_questions", related_questions))
        
        # Skip if no results found at all
        if not sources:
            provider.warning("No usable results found for keyword")
            return None
        
        # Process all sources of results
        for source_name, source_results in sources:
            provider.debug(f"Processing results from {source_name}")
            
            for i, result in enumerate(source_results[:5]):
                title = result.get("title", "")
                if title is None:
                    title = ""
                title = title.strip()
                
                link = result.get("link", "")
                if link is None:
                    link = ""
                link = link.strip()
                
                snippet = result.get("snippet", "")
                if snippet is None:
                    snippet = ""
                snippet = snippet.strip()
                
                if not title or not link:
                    continue
                    
                # Skip results from the domain containing the keyword (likely competitors)
                domain = link.split('/')[2] if len(link.split('/')) > 2 else ""
                keyword_parts = keyword.lower().replace(" ", "")
                if keyword_parts in domain.lower().replace(".", ""):
                    continue
                
                # Skip results from common unwanted domains
                unwanted_domains = ["pinterest", "quora", "facebook", "twitter", "instagram", "youtube", "tiktok"]
                if any(unwanted in domain.lower() for unwanted in unwanted_domains):
                    continue
                
                # Store only the necessary data 
                links.append({
                    "title": title,
                    "link": link
                })
        
        if not links:
            provider.warning("No valid links found after filtering")
            return None
            
        # Format links in the requested output format
        # Add section header
        if output_format == 'wordpress':
            content = [
                '''<!-- wp:heading -->''',
                '''<h2>External Resources</h2>''',
                '''<!-- /wp:heading -->''',
                '''<!-- wp:paragraph -->''',
                '''<p>Here are some helpful resources for more information about this topic:</p>''',
                '''<!-- /wp:paragraph -->''',
                '''<!-- wp:list -->''',
                '''<ul>'''
            ]
            
            # Add list items
            for link_data in links[:5]:  # Limit to 5 links
                title = link_data["title"]
                link_url = link_data["link"]
                content.append(f'<li><a href="{link_url}" target="_blank" rel="noopener noreferrer">{title}</a></li>')
            
            content.append('</ul>')
            content.append('<!-- /wp:list -->')
            
            formatted_section = "\n".join(content)
        else:  # markdown
            content = ["## External Resources", "", "Here are some helpful resources for more information about this topic:", ""]
            
            # Add list items
            for link_data in links[:5]:  # Limit to 5 links
                title = link_data["title"]
                link_url = link_data["link"]
                content.append(f"- [{title}]({link_url})")
            
            formatted_section = "\n".join(content)
        
        provider.success(f"Generated external links section with {len(links[:5])} links")
        return formatted_section
        
    except Exception as e:
        provider.error(f"Error in _get_external_links: {str(e)}")
        provider.error(f"Stack trace:\n{traceback.format_exc()}")
        return None
