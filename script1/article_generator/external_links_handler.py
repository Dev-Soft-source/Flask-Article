# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from serpapi import GoogleSearch
from typing import List, Dict
import re
from .logger import logger

def generate_external_links_section(keyword: str, serp_api_key: str, output_format: str = 'markdown') -> str:
    """
    Generate a section with external links related to the keyword.
    
    Args:
        keyword (str): Main keyword for the article
        serp_api_key (str): SerpAPI key for search
        output_format (str): Output format ('markdown' or 'wordpress')
    Returns:
        str: Formatted external links section
    """
    logger.info(f"Generating external links for keyword: {keyword}")
    logger.debug(f"Using SerpAPI key: {serp_api_key[:5]}... (truncated)")
    logger.debug(f"Output format: {output_format}")
    
    # Validate input parameters
    if not keyword or not keyword.strip():
        logger.error("Empty keyword provided for external links")
        return ""
        
    if not serp_api_key or not serp_api_key.strip():
        logger.error("Empty SerpAPI key provided")
        return ""
    
    try:
        # Set up search parameters - Add more parameters to improve results
        params = {
            "engine": "google",
            "q": keyword,
            "api_key": serp_api_key,
            "gl": "us",      # Set to US results
            "hl": "en",      # Set to English
            "num": 10,       # Get 10 results (increased from 5)
            "safe": "active", # Safe search
            "google_domain": "google.com",
            "no_cache": True  # Bypass cache for fresh results
        }
        
        logger.debug(f"SerpAPI search parameters: {params}")
        
        # Perform search
        logger.debug("Executing SerpAPI Google search...")
        search = GoogleSearch(params)
        results = search.get_dict()
        logger.debug(f"Search results received, status: {'success' if 'organic_results' in results else 'failed'}")
        
        # Log any error messages from the API
        if "error" in results:
            logger.error(f"SerpAPI error: {results['error']}")
            return ""
        
        # Log the response keys for debugging
        logger.debug(f"Response contains keys: {list(results.keys())}")
        
        # Extract organic results
        links = []
        
        # Check for different result types, not just organic_results
        result_sources = []
        
        if "organic_results" in results:
            result_sources.append(("organic_results", results["organic_results"]))
            logger.debug(f"Found {len(results['organic_results'])} organic results")
        
        # Also check for other potential sources of links
        if "related_questions" in results:
            logger.debug(f"Found {len(results['related_questions'])} related questions")
            related_questions = []
            for q in results["related_questions"]:
                if q.get("link"):
                    related_questions.append({
                        "title": q.get("question", "Related Question"),
                        "link": q.get("link"),
                        "snippet": q.get("snippet", "")
                    })
            if related_questions:
                result_sources.append(("related_questions", related_questions))
        
        # Process all sources of results
        for source_name, source_results in result_sources:
            logger.debug(f"Processing results from {source_name}")
            
            for i, result in enumerate(source_results[:7]):  # Limit to 7 results per source
                title = result.get("title", "")
                link = result.get("link", "")
                snippet = result.get("snippet", "")
                
                logger.debug(f"Processing {source_name} result #{i+1}: {title[:30]}... (truncated)")
                
                # Skip results that don't have both title and link
                if not title or not link:
                    logger.debug(f"Skipping {source_name} result #{i+1}: Missing title or link")
                    continue
                    
                # Skip results that contain the keyword in the domain (likely competitors)
                domain = link.split('/')[2] if len(link.split('/')) > 2 else ""
                if keyword.lower().replace(" ", "") in domain.lower().replace(".", ""):
                    logger.debug(f"Skipping {source_name} result #{i+1}: Keyword found in domain {domain}")
                    continue
                
                # Skip results from common unwanted domains
                unwanted_domains = ["pinterest", "quora", "facebook", "twitter", "instagram", "youtube", "tiktok"]
                if any(unwanted in domain.lower() for unwanted in unwanted_domains):
                    logger.debug(f"Skipping {source_name} result #{i+1}: Domain '{domain}' is in unwanted list")
                    continue
                
                logger.debug(f"Adding {source_name} result #{i+1} to links")
                
                # Format links based on output format
                if output_format == 'wordpress':
                    links.extend([
                        '''<!-- wp:paragraph -->''',
                        f'''<p><a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a></p>''',
                        '''<!-- /wp:paragraph -->'''
                    ])
                else:  # markdown
                    links.append(f"- [{title}]({link})")
            
        # Log how many links we found
        logger.success(f"Found {len(links)} external links in total")
        
        # Return formatted links
        if links:
            if output_format == 'wordpress':
                # Format as WordPress blocks
                content = []
                
                # Add heading
                content.append('<!-- wp:heading -->')
                content.append('<h2>External Resources</h2>')
                content.append('<!-- /wp:heading -->')
                
                # Add introduction
                content.append('<!-- wp:paragraph -->')
                content.append('<p>Here are some helpful resources for more information about this topic:</p>')
                content.append('<!-- /wp:paragraph -->')
                
                # Add list
                content.append('<!-- wp:list -->')
                content.append('<ul>')
                
                # Limit to 5 best links to avoid overwhelming the reader
                # The problem is here - links contains strings, not dictionaries
                # Process links based on their format
                link_count = 0
                for i in range(min(len(links), 15)):  # Check up to 15 items to find 5 valid links
                    if link_count >= 5:  # Only include 5 links maximum
                        break
                        
                    link_data = links[i]
                    
                    # Skip non-link elements like WordPress formatting tags
                    if not link_data.startswith('<p><a href=') and not link_data.startswith('- ['):
                        continue
                    
                    link_count += 1
                    
                    if output_format == 'wordpress':
                        # The links are already properly formatted as WordPress paragraphs
                        # Just extract the content and reformat for the list
                        if '<p><a href=' in link_data:
                            # Extract from WordPress paragraph format
                            match = re.search(r'<a href="([^"]+)" target="_blank" rel="noopener noreferrer">([^<]+)</a>', link_data)
                            if match:
                                link_url = match.group(1)
                                title = match.group(2)
                                content.append(f'<li><a href="{link_url}" target="_blank" rel="noopener noreferrer">{title}</a></li>')
                    else:
                        # Extract from markdown format
                        match = re.search(r'\[(.*?)\]\((.*?)\)', link_data)
                        if match:
                            title = match.group(1)
                            link_url = match.group(2)
                            content.append(f'<li><a href="{link_url}" target="_blank" rel="noopener noreferrer">{title}</a></li>')
                
                content.append('</ul>')
                content.append('<!-- /wp:list -->')
                
                formatted_output = '\n'.join(content)
                logger.debug(f"Formatted WordPress output (length: {len(formatted_output)})")
                logger.debug(f"First 100 chars: {formatted_output[:100]}...")
                
                # Validate the output
                if not formatted_output.strip():
                    logger.error("Generated empty WordPress external links content")
                    return ""
                
                return formatted_output
            else:
                # For markdown, add a heading and format as a list
                md_content = ["## External Resources", ""]
                for link in links[:5]:  # Limit to 5 best links for markdown too
                    md_content.append(f"- {link}")
                
                formatted_output = "\n".join(md_content)
                logger.debug(f"Formatted Markdown output (length: {len(formatted_output)})")
                
                # Validate the output
                if not formatted_output.strip():
                    logger.error("Generated empty Markdown external links content")
                    return ""
                
                return formatted_output
        else:
            logger.warning("No links to return after filtering")
            return ""
            
    except Exception as e:
        logger.error(f"Error generating external links: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "" 