import os
from typing import List, Dict, Any, Optional
from serpapi import GoogleSearch
import trafilatura
from bs4 import BeautifulSoup
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from article_generator.logger import logger
import time
from ddgs import DDGS
from http.client import HTTPException


class WebContentRetriever:
    def __init__(self, config):
        """Initialize the WebContentRetriever with necessary components."""
        self.serp_api_key = config.serp_api_key
        self.cache_dir = "cache/rag_cache"  # Default cache 
        self.config = config
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
            logger.success(f"Embedding model loaded with dimension {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.embedding_model = None
            self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Initialize FAISS index
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            self.text_chunks = []
            self.chunk_metadata = []
            logger.success("FAISS index initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {str(e)}")
            self.index = None

    def search_duckduckgo(self, keyword: str, num_results: int = 10) -> List[str]:
        """
        Search DuckDuckGo for URLs related to a keyword.

        Args:
            keyword: Search keyword.
            num_results: Maximum number of results to return (default: 10).

        Returns:
            List of unique, valid URLs.

        Raises:
            ValueError: If keyword is empty or num_results is invalid.
        """
        logger.info(f"Searching DuckDuckGo for: {keyword}")

        # Validate input
        if not keyword.strip():
            logger.error("Empty keyword provided")
            raise ValueError("Keyword cannot be empty")
        if num_results < 1:
            logger.error("Number of results must be positive")
            raise ValueError("Number of results must be positive")

        urls = set()  # Use set for efficient uniqueness
        attempt = 0
        max_retries = 5  # Reduced for faster failure
        initial_delay = 10  # Optimized initial delay

        with DDGS() as ddgs:  # Context manager for resource cleanup
            while attempt < max_retries:
                try:
                    logger.debug(f"Attempt {attempt + 1}/{max_retries} for keyword: {keyword}")
                    # Fetch more results to account for filtering
                    results = ddgs.text(keyword, region="wt-wt", max_results=num_results * 2)
                    results = list(results)  # Convert iterator to list early
                    logger.debug(f"Retrieved {len(results)} results from DDGS")

                    if not results:
                        logger.warning(f"No results returned on attempt {attempt + 1}")
                    else:
                        for result in results:
                            url = result.get("href")
                            if url and self._is_valid_url(url):
                                urls.add(url)
                                logger.debug(f"Added URL: {url}")
                                if len(urls) >= num_results:
                                    break
                        if urls:
                            break  # Exit retry loop if sufficient URLs found

                except HTTPException as e:
                    if "429" in str(e):  # Explicit rate limit check
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
                    logger.error(f"Unexpected error in DuckDuckGo search: {type(e).__name__}: {e}")
                    break

                attempt += 1
                time.sleep(0.2)  # Minimal delay between attempts to avoid overwhelming server

        if urls:
            logger.info(f"Found {len(urls)} unique URLs for keyword: {keyword}")
            return list(urls)[:num_results]

        logger.warning(f"No valid URLs found for keyword: {keyword}, using fallback")
        return self._get_fallback_urls(keyword)

    def search_google(self, keyword: str) -> List[str]:
        """Search Google using SerpAPI and return relevant URLs."""
        logger.info(f"Searching Google for: {keyword} using SerpAPI")
        
        if not self.serp_api_key:
            logger.warning("No SerpAPI key provided, using fallback URLs")
            return self._get_fallback_urls(keyword)

        params = {
            "api_key": self.serp_api_key,
            "engine": "google",
            "q": keyword,
            "num": 10,
            "gl": "us",
            "hl": "en"
        }

        try:
            logger.debug(f"SerpAPI search parameters: {params}")
            search = GoogleSearch(params)
            results = search.get_dict()
            
            urls = []
            
            # Extract organic results
            if "organic_results" in results:
                logger.debug(f"Found {len(results['organic_results'])} organic results")
                urls.extend([result["link"] for result in results["organic_results"]])
            
            # Extract knowledge graph data
            if "knowledge_graph" in results:
                logger.debug("Found knowledge graph data")
                kg = results["knowledge_graph"]
                if "source" in kg:
                    urls.append(kg["source"])
            
            # Extract related questions
            if "related_questions" in results:
                logger.debug(f"Found {len(results['related_questions'])} related questions")
                for question in results["related_questions"]:
                    if "link" in question:
                        urls.append(question["link"])
            
            # Filter and validate URLs
            valid_urls = [url for url in urls if self._is_valid_url(url)]
            
            if not valid_urls:
                logger.warning("No valid URLs found, using fallback URLs")
                return self._get_fallback_urls(keyword)
            
            logger.success(f"Found {len(valid_urls[:5])} valid URLs via SerpAPI")
            return valid_urls[:5]  # Return top 5 URLs
            
        except Exception as e:
            logger.error(f"Error in Google search: {str(e)}")
            return self._get_fallback_urls(keyword)

    def _get_fallback_urls(self, keyword: str) -> List[str]:
        """Return a list of reliable fallback URLs."""
        logger.warning(f"Using fallback URLs for keyword: {keyword}")
        fallback_urls = [
            f"https://www.britannica.com/topic/{keyword.replace(' ', '-')}",
            f"https://www.sciencedirect.com/search?qs={keyword.replace(' ', '%20')}",
            f"https://www.nationalgeographic.com/search?q={keyword.replace(' ', '%20')}",
            f"https://www.ncbi.nlm.nih.gov/search/all/?term={keyword.replace(' ', '%20')}",
            f"https://scholar.google.com/scholar?q={keyword.replace(' ', '+')}"
        ]
        logger.debug(f"Generated {len(fallback_urls)} fallback URLs")
        return fallback_urls

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for our purposes."""
        # Skip video results, images, PDFs, etc.
        skip_domains = [
            'youtube.com', 'maps.google', 'google.com/shopping',
            'facebook.com', 'twitter.com', 'instagram.com',
            'tiktok.com', 'pinterest.com', 'reddit.com',
            'wikipedia.org', 'en.wikipedia', 'wiki/', 'wikimedia.org',
            'pdf', 'download', 'login', 'signin'
        ]
        
        # Check both conditions: URL starts with http/https AND doesn't contain any of the skip domains
        return url.startswith(('http://', 'https://')) and all(domain not in url.lower() for domain in skip_domains)

    def extract_content(self, url: str) -> Optional[str]:
        """Extract content from a URL using trafilatura."""
        logger.info(f"Extracting content from: {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                logger.debug(f"Successfully downloaded content from {url}")
                content = trafilatura.extract(downloaded)
                if content:
                    content_length = len(content)
                    # Try to extract title from HTML
                    title = None
                    try:
                        soup = BeautifulSoup(downloaded, 'html.parser')
                        title = soup.title.string if soup.title else None
                    except:
                        pass
                    
                    logger.log_content_scraping(url, content_length, title)
                    return content
                
                # Fallback to BeautifulSoup if trafilatura fails
                logger.warning(f"Trafilatura extraction failed for {url}, falling back to BeautifulSoup")
                soup = BeautifulSoup(downloaded, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                logger.log_content_scraping(url, len(text), soup.title.string if soup.title else None)
                return text
            else:
                logger.error(f"Failed to download content from {url}")
                logger.log_url_access(url, 0, False)
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            logger.log_url_access(url, 0, False)
        return None

    def build_knowledge_base(self, keyword: str) -> None:
        """Build knowledge base from web content."""
        logger.info(f"Building knowledge base for keyword: {keyword}")
        
         # Search for URLs
        if self.config.rag_article_retriever_engine == "Duckduckgo":
            urls = self.search_duckduckgo(keyword) # Get more URLs than needed in case some fail
        else:
            urls = self.search_google(keyword)  # Get more URLs than needed in case some fail
        
        total_chunks = 0
        successful_urls = 0
        
        for url in urls:
            content = self.extract_content(url)
            if content:
                successful_urls += 1
                chunks = self._chunk_text(content)
                logger.debug(f"Created {len(chunks)} text chunks from {url}")
                
                if chunks:
                    embeddings = self._create_embeddings(chunks)
                    
                    if embeddings:
                        # Add to FAISS index
                        self.index.add(np.array(embeddings).astype('float32'))
                        self.text_chunks.extend(chunks)
                        self.chunk_metadata.extend([{"url": url} for _ in chunks])
                        total_chunks += len(chunks)
                        logger.log_embedding_generation(len(chunks), self.embedding_dimension, url)
        
        logger.success(f"Knowledge base built with {total_chunks} chunks from {successful_urls} URLs")
        
        # Display stats
        logger.display_rag_stats()

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks."""
        logger.debug(f"Chunking text with chunk size: {chunk_size}")
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += 1
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.debug(f"Created {len(chunks)} text chunks")
        return chunks

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks."""
        logger.debug(f"Creating embeddings for {len(texts)} text chunks")
        try:
            embeddings = self.embedding_model.encode(texts).tolist()
            logger.debug(f"Successfully created {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return []

    def retrieve_relevant_content(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant content for a query."""
        logger.info(f"Retrieving relevant content for query: '{query}'")
        
        # Check if we have content in the knowledge base
        if not self.index or not hasattr(self, 'text_chunks') or len(self.text_chunks) == 0 or self.index.ntotal == 0:
            logger.warning("No content in knowledge base, cannot retrieve")
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in FAISS index
            distances, indices = self.index.search(
                np.array([query_embedding]).astype('float32'), k
            )
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.text_chunks):
                    results.append({
                        "text": self.text_chunks[idx],
                        "metadata": self.chunk_metadata[idx],
                        "distance": float(distances[0][i])
                    })
            
            # Calculate top score (convert distance to similarity score)
            top_score = 0
            if results:
                # Lower distance is better, so convert to similarity
                top_score = 1.0 / (1.0 + min(r["distance"] for r in results))
            
            logger.log_rag_search(query, len(results), top_score)
            logger.success(f"Retrieved {len(results)} relevant chunks for query")
            
            # Log the context information
            if results:
                context_length = sum(len(r["text"]) for r in results)
                sources = list(set(r["metadata"]["url"] for r in results if "url" in r["metadata"]))
                logger.log_rag_context(context_length, len(results), sources)
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving content: {str(e)}")
            return [] 