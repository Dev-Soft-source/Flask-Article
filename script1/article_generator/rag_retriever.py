# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import requests
from bs4 import BeautifulSoup
import trafilatura
import time
import random
import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from article_generator.logger import logger
from sentence_transformers import SentenceTransformer
import faiss
from serpapi import GoogleSearch
from utils.rate_limiter import serpapi_rate_limiter,duck_duck_go_rate_limiter
from ddgs import DDGS
from http.client import HTTPException



class WebContentRetriever:
    """Retrieves web content related to a keyword using search engines and web scraping."""
    
    def __init__(self, config):
        """Initialize the retriever with configuration."""
        self.config = config
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
        }
        
        # Create directories for cached content
        os.makedirs("cache", exist_ok=True)
        os.makedirs(self.config.rag_cache_dir, exist_ok=True)
        
        # Initialize embedding model
        try:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.config.rag_embedding_model)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.success(f"Embedding model loaded with dimension {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback if model fails to load - we'll handle this in the search method
            self.embedding_model = None
            self.embedding_dimension = self.config.rag_embedding_dimension  # Use configured dimension
        
        # Initialize FAISS index
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            logger.info("FAISS index initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.index = None 
    

    def _is_valid_url(self, url: str) -> bool:
        """Placeholder for URL validation (implement as needed)."""
        return bool(url.startswith(('http://', 'https://')))

    def _get_fallback_urls(self, keyword: str, num_results: int) -> List[str]:
        """Placeholder for fallback URLs (implement as needed)."""
        logger.warning(f"Fallback URLs for keyword: {keyword}")
        return []

    def search_duckduckgo(self, keyword: str, num_results: int = 5) -> List[str]:
        """
        Search DuckDuckGo for URLs related to a keyword.

        Args:
            keyword: Search keyword.
            num_results: Maximum number of results to return (default: 5).

        Returns:
            List of unique, valid URLs.

        Raises:
            SearchError: If the search fails after retries or encounters an unrecoverable error.
        """
        logger.info(f"Searching DuckDuckGo for: {keyword}")

        urls = set()  # Use set to ensure uniqueness
        attempt = 0
        max_retries = 5  # Reduced retries for faster failure
        initial_delay = 10 # Lower initial delay for quicker retries

        # Validate input
        if not keyword.strip():
            logger.error("Empty keyword provided")
            raise ValueError("Keyword cannot be empty")
        
        if num_results < 1:
            logger.error("Number of results must be positive")
            raise ValueError("Number of results must be positive")

        with DDGS() as ddgs:  # Use context manager for cleaner resource handling
            while attempt < max_retries:
                try:
                    # Fetch more results than needed to account for filtering
                    results = ddgs.text(keyword, region="wt-wt", max_results=num_results * 2)
                    logger.debug(f"Received {len(results)} results on attempt {attempt + 1}")

                    if not results:
                        logger.warning(f"No results found on attempt {attempt + 1}")
                    else:
                        for result in results:
                            url = result.get("href")
                            if url and self._is_valid_url(url):
                                urls.add(url)  # Add to set for uniqueness
                                if len(urls) >= num_results:
                                    break
                        if urls:
                            break  # Exit retry loop if sufficient URLs are found

                except HTTPException as e:
                    if "429" in str(e):  # Explicitly check for rate limit
                        attempt += 1
                        if attempt >= max_retries:
                            logger.error(f"Max retries ({max_retries}) reached for rate limit: {e}")
                            break
                        delay = initial_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limit hit: {e}. Retrying after {delay}s (Attempt {attempt}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.error(f"HTTP error in DuckDuckGo search: {e}")
                        break
                except Exception as e:
                    logger.error(f"Unexpected error in DuckDuckGo search: {e}")
                    break
                attempt += 1

        if urls:
            logger.info(f"Found {len(urls)} unique URLs for keyword: {keyword}")
            return list(urls)[:num_results]

        logger.warning(f"No valid URLs found for keyword: {keyword}, using fallback")
        return self._get_fallback_urls(keyword, num_results)

    def search_google(self, keyword: str, num_results: int = 5) -> List[str]:
        """
        Search Google for URLs related to a keyword using SerpAPI.
        
        Args:
            keyword: Search keyword
            num_results: Maximum number of results to return
            
        Returns:
            List of URLs
        """
        logger.info(f"Searching Google for: {keyword} using SerpAPI")
        
        try:
            # Check if serp_api_key is available in config
            if not hasattr(self.config, 'serp_api_key') or not self.config.serp_api_key:
                logger.error("SerpAPI key is missing in configuration")
                return self._get_fallback_urls(keyword, num_results)
            
            # Set up SerpAPI search parameters
            params = {
                "engine": "google",
                "q": keyword,
                "api_key": self.config.serp_api_key,
                "gl": "us",      # Set to US results
                "hl": "en",      # Set to English
                "num": num_results * 2,  # Request more to account for filtering
                "safe": "active", # Safe search
                "google_domain": "google.com",
                "no_cache": True  # Bypass cache for fresh results
            }
            
            logger.debug(f"SerpAPI search parameters: {params}")
            logger.debug("Executing SerpAPI Google search...")

            # Perform search with rate limiting
            if serpapi_rate_limiter:
                logger.debug("Using rate limiter for SerpAPI call")

                def make_api_call():
                    search = GoogleSearch(params)
                    return search.get_dict()

                results = serpapi_rate_limiter.execute_with_rate_limit(make_api_call)
            else:
                search = GoogleSearch(params)
                results = search.get_dict()
            
            # Log any error messages from the API
            if "error" in results:
                logger.error(f"SerpAPI error: {results['error']}")
                return self._get_fallback_urls(keyword, num_results)
                
            # Extract organic results
            urls = []
            
            # Process organic search results
            if "organic_results" in results:
                logger.debug(f"Found {len(results['organic_results'])} organic results")
                
                for result in results["organic_results"]:
                    url = result.get("link", "")
                    if url and self._is_valid_url(url) and url not in urls:
                        urls.append(url)
                        if len(urls) >= num_results:
                            break
            
            # If we don't have enough results, try knowledge graph if available
            if len(urls) < num_results and "knowledge_graph" in results:
                if "website" in results["knowledge_graph"]:
                    website = results["knowledge_graph"]["website"]
                    if website and self._is_valid_url(website) and website not in urls:
                        urls.append(website)
            
            # If still not enough results, check related questions
            if len(urls) < num_results and "related_questions" in results:
                for question in results["related_questions"]:
                    if "link" in question:
                        url = question["link"]
                        if url and self._is_valid_url(url) and url not in urls:
                            urls.append(url)
                            if len(urls) >= num_results:
                                break
            
            # If still not enough results, use fallback URLs
            if not urls:
                logger.warning("No URLs found in SerpAPI results, using fallback URLs")
                return self._get_fallback_urls(keyword, num_results)
            
            logger.success(f"Found {len(urls)} URLs via SerpAPI for keyword: {keyword}")
            return urls[:num_results]
            
        except Exception as e:
            logger.error(f"Error during SerpAPI search: {e}")
            return self._get_fallback_urls(keyword, num_results)
    
    def _get_fallback_urls(self, keyword: str, num_results: int = 5) -> List[str]:
        """
        Get fallback URLs when search fails.
        
        Args:
            keyword: Search keyword
            num_results: Maximum number of results to return
            
        Returns:
            List of fallback URLs
        """
        fallback_urls = []
        for url_template in self.config.rag_fallback_urls:
            fallback_urls.append(f"{url_template}{keyword.replace(' ', '%20')}")
            
        # Filter valid URLs
        valid_urls = [url for url in fallback_urls if self._is_valid_url(url)]
        logger.warning(f"Using {len(valid_urls[:num_results])} fallback URLs")
        return valid_urls[:num_results]

    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid for our purposes.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is valid, False otherwise
        """
        # Skip video results, images, PDFs, etc.
        skip_domains = [
            'youtube.com', 'maps.google', 'google.com/shopping',
            'facebook.com', 'twitter.com', 'instagram.com',
            'wikipedia.org', 'en.wikipedia', 'wiki/', 'wikimedia.org',
            'pdf', 'download', 'login', 'signin'
        ]
        
        return all(domain not in url.lower() for domain in skip_domains)

    def extract_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract content from a URL using trafilatura.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Dictionary with extracted content and metadata, or None if extraction failed
        """
        logger.info(f"Extracting content from: {url}")
        try:
            # Use consistent, transparent headers
            standard_headers = {
                'User-Agent': 'ArticleGenerator/1.0 (Educational Research Tool)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            
            # Download the content directly using requests
            try:
                response = requests.get(url, headers=standard_headers, timeout=15)
                success = response.status_code == 200
                logger.log_url_access(url, response.status_code, success)
                
                if not success:
                    logger.warning(f"Could not download content from {url}")
                    return None
                
                # Now pass the content to trafilatura for extraction
                downloaded = response.text
            except Exception as req_err:
                logger.error(f"Error downloading content: {req_err}")
                logger.log_url_access(url, 0, False)
                return None
            
            # First try with trafilatura's extract
            try:
                content = trafilatura.extract(downloaded, include_links=False, include_images=False, output_format='txt')
            except Exception as traf_err:
                logger.error(f"Error with trafilatura extraction: {traf_err}")
                content = None
            
            # If trafilatura fails, try BeautifulSoup
            if not content or len(content.strip()) < 100:
                logger.warning(f"Trafilatura extraction yielded insufficient content, trying BeautifulSoup")
                try:
                    soup = BeautifulSoup(downloaded, 'html.parser')
                    
                    # Remove common non-content elements
                    for element in soup.select('script, style, nav, footer, header, aside'):
                        element.extract()
                    
                    # Get content from common content containers
                    for container in ['article', 'main', '.content', '#content', '.post', '.entry']:
                        main_content = soup.select(container)
                        if main_content:
                            content = '\n\n'.join([elem.get_text(separator='\n') for elem in main_content])
                            content = ' '.join(content.split())  # Normalize whitespace
                            break
                    
                    # If still no content, get the body
                    if not content or len(content.strip()) < 100:
                        content = soup.body.get_text(separator='\n')
                        content = ' '.join(content.split())  # Normalize whitespace
                except Exception as bs_err:
                    logger.error(f"BeautifulSoup extraction failed: {bs_err}")
            
            if not content or len(content.strip()) < 100:  # Still ensure we have meaningful content
                logger.warning(f"No meaningful content extracted from {url}")
                return None
            
            # Extract metadata if available
            try:
                metadata = trafilatura.extract_metadata(downloaded)
                if metadata:
                    title = metadata.title if hasattr(metadata, 'title') and metadata.title else "Unknown Title"
                    author = metadata.author if hasattr(metadata, 'author') and metadata.author else "Unknown Author"
                    date = metadata.date if hasattr(metadata, 'date') and metadata.date else None
                    description = metadata.description if hasattr(metadata, 'description') and metadata.description else ""
                else:
                    title = "Unknown Title"
                    author = "Unknown Author"
                    date = None
                    description = ""
            except Exception as meta_err:
                logger.warning(f"Error extracting metadata: {meta_err}")
                # Set default values
                title = "Unknown Title"
                author = "Unknown Author"
                date = None
                description = ""
                
                # Try to get title from BeautifulSoup
                try:
                    soup = BeautifulSoup(downloaded, 'html.parser') if 'soup' not in locals() else soup
                    if soup.title and soup.title.string:
                        title = soup.title.string.strip()
                except Exception as bs_err:
                    logger.error(f"BeautifulSoup title extraction failed: {bs_err}")
            
            # Log content scraping
            content_length = len(content)
            logger.log_content_scraping(url, content_length, title)
            
            # Create result dictionary
            result = {
                "url": url,
                "content": content,
                "title": title,
                "author": author,
                "date": date,
                "description": description,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            # Log the failure
            logger.log_url_access(url, 0, False)
            return None

    def get_articles(self, keyword: str, num_articles: int = 5) -> List[Dict[str, Any]]:
        """
        Get articles related to a keyword by searching and extracting content.
        
        Args:
            keyword: Search keyword
            num_articles: Maximum number of articles to return
            
        Returns:
            List of dictionaries containing article content and metadata
        """
        logger.info(f"Retrieving articles for keyword: {keyword}")
        
        # Search for URLs
        if self.config.rag_article_retriever_engine == "Duckduckgo":
            urls = self.search_duckduckgo(keyword, 5 * 2) # Get more URLs than needed in case some fail
        else:
            urls = self.search_google(keyword, num_articles * 2)  # Get more URLs than needed in case some fail
        
        if not urls:
            logger.warning(f"No URLs found for keyword: {keyword}")
            return []
        
        # Extract content from each URL
        articles = []
        for url in urls:
            # Extract content without random delays
            article = self.extract_content(url)
            if article:
                articles.append(article)
                logger.success(f"Successfully extracted article from {url}")
                
                if len(articles) >= num_articles:
                    break
        
        logger.success(f"Retrieved {len(articles)} articles for keyword: {keyword}")
        return articles

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Maximum number of words per chunk
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of text chunks
        """
        # Use configured chunk size if not specified
        if chunk_size is None:
            chunk_size = self.config.rag_chunk_size
            
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks

    def create_embeddings(self, articles: List[Dict[str, Any]]) -> Tuple[List[str], np.ndarray, List[Dict[str, Any]]]:
        """
        Create embeddings for article chunks.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Tuple of (text chunks, embeddings array, metadata list)
        """
        logger.info("Creating embeddings for articles")
        
        if not self.embedding_model:
            logger.error("Embedding model not available")
            return [], np.array([]), []
        
        all_chunks = []
        all_metadata = []
        
        # Process each article
        for article in articles:
            # Combine title and content
            full_text = f"{article['title']}\n\n{article['content']}"
            
            # Split into chunks
            chunks = self.chunk_text(full_text)
            
            # Add chunks and metadata
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "url": article["url"],
                    "title": article["title"],
                    "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk
                })
        
        # Create embeddings
        try:
            embeddings = self.embedding_model.encode(all_chunks)
            logger.success(f"Created {len(all_chunks)} chunk embeddings")
            
            # Log embedding generation
            logger.log_embedding_generation(
                text_chunks=len(all_chunks),
                vector_dim=self.embedding_dimension,
                source=f"Articles for '{articles[0]['title']}' and others"
            )
            
            return all_chunks, embeddings, all_metadata
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return [], np.array([]), []

    def create_fallback_content(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Create synthetic article content for when web content retrieval fails.
        
        Args:
            keyword: Search keyword
            
        Returns:
            List containing a single synthetic article
        """
        logger.warning(f"Using fallback content generation for: {keyword}")
        
        # Create a basic article with general information about the topic
        title = f"Information about {keyword}"
        
        content = f"""
This is general information about {keyword}. 

When dealing with {keyword}, it's important to understand the basics. 
There are several key aspects to consider.

First, {keyword} requires proper preparation and planning.
Second, having the right tools and resources is essential.
Third, following best practices helps ensure successful outcomes.

Remember to always prioritize safety and ethical considerations when working with {keyword}.
        """
        
        # Create article dictionary
        article = {
            "url": "fallback://synthetic-content",
            "content": content,
            "title": title,
            "author": "System Generated",
            "date": None,
            "description": f"General information about {keyword}",
            "timestamp": time.time()
        }
        
        return [article]
        
    def build_knowledge_base(self, keyword: str, num_articles: int = 5) -> bool:
        """
        Build a knowledge base for a keyword by retrieving articles and creating embeddings.
        
        Args:
            keyword: Search keyword
            num_articles: Maximum number of articles to retrieve
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Building knowledge base for: {keyword}")
        
        # Get articles
        articles = self.get_articles(keyword, num_articles)
        
        # If no articles found, try fallback content generation
        if not articles:
            logger.warning("No articles found for knowledge base, using fallback content")
            articles = self.create_fallback_content(keyword)
            
            # If still no articles, return failure
            if not articles:
                logger.warning("No fallback content generated")
                return False
        
        # Create embeddings
        chunks, embeddings, metadata = self.create_embeddings(articles)
        
        if len(chunks) == 0:
            logger.warning("No chunks created for knowledge base")
            return False
        
        # Store data
        self.chunks = chunks
        self.metadata = metadata
        
        # Build FAISS index
        try:
            if self.index:
                self.index = faiss.IndexFlatL2(self.embedding_dimension)
            self.index.add(embeddings.astype('float32'))
            logger.success(f"Knowledge base built with {len(chunks)} chunks from {len(articles)} articles")
            return True
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return False

    def search_knowledge_base(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing relevant chunks and metadata
        """
        logger.info(f"Searching knowledge base for: {query}")
        
        if not self.embedding_model or not self.index or not hasattr(self, 'chunks'):
            logger.error("Knowledge base not initialized")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Format results
            results = []
            top_score = 0
            
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.chunks):  # Valid index check
                    score = float(scores[0][i])
                    if i == 0:
                        top_score = score
                    
                    results.append({
                        "content": self.chunks[idx],
                        "metadata": self.metadata[idx],
                        "score": score
                    })
            
            # Log RAG search
            logger.log_rag_search(query, len(results), top_score)
            logger.success(f"Found {len(results)} relevant chunks")
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []

    def get_context_for_generation(self, keyword: str, num_articles: int = 5, num_chunks: int = 5) -> str:
        """
        Get context for article generation by building and searching a knowledge base.
        
        Args:
            keyword: Search keyword
            num_articles: Maximum number of articles to retrieve
            num_chunks: Number of relevant chunks to include in context
            
        Returns:
            Formatted context string for article generation
        """
        logger.info(f"Getting context for article generation: {keyword}")
        
        try:
            # Check if embedding model and index are initialized
            if not self.embedding_model or not self.index:
                logger.warning("Embedding model or FAISS index not properly initialized")
                return ""
                
            # Build knowledge base
            success = self.build_knowledge_base(keyword, num_articles)
            
            if not success:
                logger.warning("Failed to build knowledge base, returning empty context")
                return ""
            
            # Check if chunks attribute exists
            if not hasattr(self, 'chunks') or len(self.chunks) == 0:
                logger.warning("No chunks available in knowledge base")
                return ""
                
            # Search knowledge base
            results = self.search_knowledge_base(keyword, num_chunks)
            
            if not results:
                logger.warning("No relevant chunks found, returning empty context")
                return ""
            
            # Format context
            context = f"# Research Context for '{keyword}'\n\n"
            
            sources = []
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                sources.append(metadata["url"])
                context += f"## Source {i}: {metadata['title']}\n"
                context += f"URL: {metadata['url']}\n\n"
                context += f"{result['content']}\n\n"
                context += "---\n\n"
            
            # Log RAG context generation
            logger.log_rag_context(
                context_length=len(context),
                chunks_used=len(results),
                sources=sources
            )
            
            logger.success(f"Generated context with {len(results)} chunks")
            return context
            
        except Exception as e:
            logger.error(f"Error in get_context_for_generation: {str(e)}")
            return ""

    def debug_response(self, url: str) -> None:
        """
        Debug function to check what is being returned from a URL request.
        
        Args:
            url: URL to check
        """
        try:
            # Use consistent headers
            standard_headers = {
                'User-Agent': 'ArticleGenerator/1.0 (Educational Research Tool)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            
            # Make the request
            logger.info(f"Making debug request to: {url}")
            response = requests.get(url, headers=standard_headers, timeout=15)
            
            # Log response details
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
            
            # Extract and log a sample of the content
            content = response.text
            content_sample = content[:5000] + "..." if len(content) > 5000 else content
            logger.info(f"Response content sample:\n{content_sample}")
            
            # Try to extract search results using our existing methods
            soup = BeautifulSoup(content, 'html.parser')
            
            # Method 1: Look for result divs with links
            results_div_g = [link['href'] for link in soup.select('div.g a') if link.get('href', '').startswith('http')]
            logger.info(f"Found {len(results_div_g)} URLs using div.g selector: {results_div_g[:5]}")
            
            # Method 2: Extract from /url?q= format
            results_url_q = []
            for a in soup.select('a'):
                href = a.get('href', '')
                if href.startswith('/url?q='):
                    url = href.split('/url?q=')[1].split('&')[0]
                    results_url_q.append(url)
            logger.info(f"Found {len(results_url_q)} URLs using /url?q= format: {results_url_q[:5]}")
            
            # Method 3: Look for cite tags within search results
            results_cite = ['https://' + cite.text.strip() for cite in soup.select('cite')]
            logger.info(f"Found {len(results_cite)} URLs using cite tags: {results_cite[:5]}")
            
            # Save the response to a file for further inspection
            with open("google_response.html", "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Saved complete response to google_response.html")
            
        except Exception as e:
            logger.error(f"Error in debug_response: {e}")

    def retrieve_relevant_content(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve content relevant to a query.
        
        Args:
            query: Query to search for
            k: Number of results to retrieve
            
        Returns:
            List of dictionaries containing relevant content and metadata
        """
        # Use configured number of chunks if not specified
        if k is None:
            k = self.config.rag_num_chunks
            
        logger.info(f"Retrieving content for query: {query}")
        
        if not self.index or not self.vectors or len(self.chunks) == 0:
            logger.warning("No content in knowledge base to search")
            return [] 