# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import requests
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urlparse
import re
import time
import random
import logging
import os
import sys
import json
from typing import List, Dict, Optional
import threading
import shutil
from utils.error_utils import ErrorHandler

# Initialize error handler
error_handler = ErrorHandler()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Import webdriver_manager for fallback
from webdriver_manager.chrome import ChromeDriverManager

# Import termcolor for colored output
from termcolor import colored

# Configure logging - minimal output to console
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger('ArticleExtractor')


# Improved spinner class with an enable flag for interactive terminals.
class Spinner:
    def __init__(self, message="Processing", enabled: Optional[bool] = None):
        """
        Args:
            message: Message to show during spinner rotation.
            enabled: Whether to enable the spinner output. By default, it is enabled only if sys.stdout.isatty() is True.
        """
        self.message = message
        self.spinning = False
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_thread = None
        self.current_count = 0
        self.total_count = 0
        
        self.enabled = enabled if enabled is not None else sys.stdout.isatty()
        # Get terminal width for proper clearing if enabled
        self.term_width = shutil.get_terminal_size().columns if self.enabled else 0
        
    def spin(self):
        if not self.enabled:
            return
        while self.spinning:
            for char in self.spinner_chars:
                if not self.spinning:
                    break
                # Create status message
                if self.total_count > 0:
                    status = f"{self.message} [{self.current_count}/{self.total_count}] {char}"
                else:
                    status = f"{self.message} {char}"
                # Calculate padding to clear the entire line
                padding = ' ' * (self.term_width - len(status) - 1)
                # Write the status with padding and return to start of line
                sys.stdout.write('\r' + colored(status, 'cyan') + padding)
                sys.stdout.flush()
                time.sleep(0.1)
    
    def start(self):
        if not self.enabled:
            return
        self.spinning = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
    
    def update(self, current, total: Optional[int] = None):
        self.current_count = current
        if total is not None:
            self.total_count = total
    
    def stop(self):
        if not self.enabled:
            return
        self.spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()
        # Clear the entire line and move to beginning
        sys.stdout.write('\r' + ' ' * self.term_width + '\r')
        sys.stdout.flush()


# -------------------------
# HTTP Request Handler
# -------------------------
class RequestHandler:
    """Handles all HTTP requests with proper error handling and retries."""

    def __init__(self, user_agent: Optional[str] = None, max_retries: int = 3, retry_delay: int = 2):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def get(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay + random.uniform(0, 1))
                else:
                    return None
        return None


# -------------------------
# Selenium Searcher
# -------------------------
class SeleniumSearcher:
    """Handles keyword search on DuckDuckGo using Selenium and extracts result URLs."""

    def __init__(self, headless: bool = True, browser_type: str = "chrome"):
        self.headless = headless
        self.browser_type = browser_type.lower()
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        try:
            if self.browser_type == "chrome":
                options = Options()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")
                options.add_argument("--disable-notifications")
                options.add_argument("--disable-popup-blocking")
                options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/91.0.4472.124 Safari/537.36")

                current_dir = os.path.dirname(os.path.abspath(__file__))
                driver_filename = "chromedriver.exe" if os.name == 'nt' else "chromedriver"
                driver_path = os.path.join(current_dir, driver_filename)
                if os.path.exists(driver_path):
                    service = Service(driver_path)
                else:
                    service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                raise ValueError(f"Unsupported browser type: {self.browser_type}")

            self.driver.set_page_load_timeout(30)
        except Exception as e:
            error_handler.handle_error(e, severity="error")
            raise

    def search(self, query: str, max_results: int = 10) -> List[str]:
        if not self.driver:
            return []

        try:
            self.driver.get("https://duckduckgo.com/")
            search_box = None
            selectors = [
                (By.ID, "searchbox_input"),
                (By.NAME, "q"),
                (By.XPATH, "//input[@type='text']"),
                (By.CSS_SELECTOR, "input[type='text']")
            ]
            for by, selector in selectors:
                try:
                    search_box = WebDriverWait(self.driver, 3).until(EC.presence_of_element_located((by, selector)))
                    if search_box:
                        break
                except:
                    continue

            if not search_box:
                return []

            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            time.sleep(3)  # Allow search results to load

            results = []
            try:
                result_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[data-testid='result-title-a']")
                for element in result_elements:
                    url = element.get_attribute("href")
                    if url and not url.startswith("javascript:") and "duckduckgo.com" not in url:
                        results.append(url)
                        if len(results) >= max_results:
                            break
            except Exception:
                pass

            if not results:
                alternative_selectors = [
                    "article h2 a",
                    "article a[href]",
                    "h2 a"
                ]
                for selector in alternative_selectors:
                    try:
                        result_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in result_elements:
                            url = element.get_attribute("href")
                            if url and not url.startswith("javascript:") and "duckduckgo.com" not in url:
                                results.append(url)
                                if len(results) >= max_results:
                                    break
                        if results:
                            break
                    except Exception:
                        continue

            if not results:
                all_links = self.driver.find_elements(By.TAG_NAME, "a")
                for link in all_links:
                    try:
                        url = link.get_attribute("href")
                        if url and not url.startswith("javascript:") and "duckduckgo.com" not in url:
                            if not any(skip in url for skip in ["duckduckgo.com", "/settings", "/about", "/privacy"]):
                                results.append(url)
                                if len(results) >= max_results:
                                    break
                    except Exception:
                        continue

            return results

        except TimeoutException:
            return []
        except Exception:
            return []

    def filter_urls(self, urls: List[str], excluded_domains: Optional[List[str]] = None) -> List[str]:
        if not excluded_domains:
            excluded_domains = [
                'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
                'linkedin.com', 'pinterest.com', 'reddit.com', 'quora.com'
            ]
        filtered_urls = []
        for url in urls:
            domain = urlparse(url).netloc
            if not any(excluded in domain for excluded in excluded_domains):
                filtered_urls.append(url)
        return filtered_urls

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass


# -------------------------
# Content Extractor
# -------------------------
class ContentExtractor:
    """Extracts and processes article content from HTML."""

    def __init__(self, request_handler: RequestHandler, max_content_length: Optional[int] = None):
        self.request_handler = request_handler
        self.max_content_length = max_content_length

    def extract_domain(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc

    def extract_article_content(self, url: str) -> Optional[Dict[str, str]]:
        response = self.request_handler.get(url)
        if not response:
            return None

        html = response.text
        try:
            extracted_text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
                include_links=True,
                include_images=True
            )
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            if extracted_text:
                content = extracted_text
            else:
                for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                    tag.decompose()
                main_content = None
                for container in ["main", "article", "div.content", "div.post", "div.entry", "#content", ".post-content"]:
                    try:
                        main_content = soup.select_one(container)
                        if main_content:
                            break
                    except Exception:
                        continue
                if main_content:
                    text = main_content.get_text(separator=' ', strip=True)
                else:
                    text = soup.get_text(separator=' ', strip=True)
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
            if self.max_content_length:
                content = content[:self.max_content_length]
            content = self._clean_content(content)
            domain = self.extract_domain(url)
            return {"title": title, "content": content, "domain": domain, "url": url}
        except Exception:
            return None

    def _clean_content(self, content: str) -> str:
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        return content.strip()


# -------------------------
# Article Extractor
# -------------------------
class ArticleExtractor:
    """
    Combines searching and content extraction.
    Initialize with a keyword and the desired number of search results.
    """

    def __init__(self, 
                 keyword: str, 
                 max_search_results: int = 5, 
                 max_content_length: Optional[int] = None, 
                 headless: bool = True,
                 spinner_enabled: Optional[bool] = None):
        self.keyword = keyword
        self.max_search_results = max_search_results
        self.request_handler = RequestHandler()
        self.searcher = SeleniumSearcher(headless=headless)
        self.content_extractor = ContentExtractor(self.request_handler, max_content_length)
        # The spinner will automatically be enabled only if sys.stdout is a TTY
        self.spinner = Spinner("Searching and extracting articles", enabled=spinner_enabled)

    def search_and_extract(self) -> List[Dict[str, str]]:
        self.spinner.start()
        try:
            search_urls = self.searcher.search(self.keyword, max_results=self.max_search_results * 2)
            if not search_urls:
                self.spinner.stop()
                error_handler.handle_error(Exception("No search results found"), severity="warning")
                return []
            filtered_urls = self.searcher.filter_urls(search_urls)
            urls_to_process = filtered_urls[:self.max_search_results]
            if not urls_to_process:
                self.spinner.stop()
                error_handler.handle_error(Exception("No valid URLs found after filtering"), severity="warning")
                return []
            self.spinner.update(0, len(urls_to_process))
            articles = []
            for i, url in enumerate(urls_to_process, 1):
                self.spinner.update(i)
                article_data = self.content_extractor.extract_article_content(url)
                if article_data:
                    articles.append(article_data)
                time.sleep(random.uniform(0.5, 1.5))
            self.spinner.stop()
            # print on screen that articles have been extracted, and show the details too
            if articles:
                error_handler.handle_error(Exception(f"Extracted {len(articles)} articles"), severity="info")
                for i, article in enumerate(articles, 1):
                    title = article['title'][:50] + "..." if len(article['title']) > 50 else article['title']
                    error_handler.handle_error(Exception(f"{i}. {title} ({article['domain']})"), severity="info")
            else:
                error_handler.handle_error(Exception("No articles extracted"), severity="warning")
            return articles
        except Exception as e:
            self.spinner.stop()
            error_handler.handle_error(e, severity="error")
            return []

    def extract_from_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        self.spinner.update(0, len(urls))
        self.spinner.start()
        try:
            articles = []
            for i, url in enumerate(urls, 1):
                self.spinner.update(i)
                article_data = self.content_extractor.extract_article_content(url)
                if article_data:
                    articles.append(article_data)
                time.sleep(random.uniform(0.5, 1.5))
            self.spinner.stop()
            return articles
        except Exception as e:
            self.spinner.stop()
            error_handler.handle_error(e, severity="error")
            return []

    def close(self):
        self.searcher.close()


# -------------------------
# Save Utility Function
# -------------------------
def save_articles_to_file(articles: List[Dict[str, str]], filename: str = "articles.txt") -> None:
    with open(filename, "a", encoding="utf-8") as f:
        for article in articles:
            f.write(f"Title: {article['title']}\n")
            f.write(f"Domain: {article['domain']}\n")
            f.write(f"URL: {article['url']}\n")
            f.write("Content:\n")
            f.write(article['content'] + "\n\n")
            f.write("-" * 80 + "\n\n")
    error_handler.handle_error(Exception(f"Saved {len(articles)} articles to {filename}"), severity="info")


# -------------------------
# Main Execution Block
# -------------------------
if __name__ == '__main__':
    error_handler.handle_error(Exception("ARTICLE EXTRACTOR"), severity="info")
    error_handler.handle_error(Exception("=" * 50), severity="info")
    
    extractor = None
    try:
        search_keyword = "Cascading Stylesheet"
        max_results = 10
        
        error_handler.handle_error(Exception(f"Keyword: {search_keyword}"), severity="info")
        error_handler.handle_error(Exception(f"Max results: {max_results}"), severity="info")
        
        extractor = ArticleExtractor(
            keyword=search_keyword, 
            max_search_results=max_results, 
            max_content_length=50000, 
            headless=True
        )
        
        extracted_articles = extractor.search_and_extract()
        
        # Uncomment to save articles to file
        # save_articles_to_file(extracted_articles, "test.txt")
        
    except Exception as e:
        error_handler.handle_error(e, severity="error")
    finally:
        if extractor is not None:
            extractor.close()
