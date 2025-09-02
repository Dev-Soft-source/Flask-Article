from typing import List, Dict, Optional, Union
from ddgs import DDGS


class DuckDuckGoSearcher:
    """
    A utility class to perform various types of searches using DuckDuckGo.

    Supports:
    - Web search
    - Image search
    - Video search
    - News search

    This class uses the `duckduckgo-search` package to query DuckDuckGo's unofficial API.
    """

    def __init__(self):
        """
        Initialize the DuckDuckGo search engine wrapper.
        """
        self.ddgs = DDGS()

    def web_search(self, query: str, max_results: int = 10, region: str = "wt-wt") -> List[Dict[str, Union[str, None]]]:
        """
        Perform a web search using DuckDuckGo.

        Args:
            query (str): The search term to look for.
            max_results (int): Maximum number of results to return.
            region (str): Region code for localization (e.g., "wt-wt", "us-en", etc.).

        Returns:
            List[Dict[str, Union[str, None]]]: A list of dictionaries containing web search results.
        """
        results = self.ddgs.text(query, region=region, max_results=max_results)
        return list(results)

    def image_search(self, query: str, max_results: int = 10, region: str = "wt-wt", size: Optional[str] = None) -> List[Dict[str, Union[str, int]]]:
        """
        Perform an image search using DuckDuckGo.

        Args:
            query (str): The image search keyword.
            max_results (int): Maximum number of results to return.
            region (str): Region code for localization.
            size (Optional[str]): Optional size filter (e.g., "Small", "Medium", "Large").

        Returns:
            List[Dict[str, Union[str, int]]]: A list of dictionaries with image metadata.
        """
        results = self.ddgs.images(query, region=region, max_results=max_results, size=size)
        return list(results)

    def video_search(self, query: str, max_results: int = 10, region: str = "wt-wt") -> List[Dict[str, Union[str, None]]]:
        """
        Perform a video search using DuckDuckGo.

        Args:
            query (str): The search keyword.
            max_results (int): Maximum number of video results.
            region (str): Region code for localization.

        Returns:
            List[Dict[str, Union[str, None]]]: A list of video result dictionaries.
        """
        results = self.ddgs.videos(query, region=region, max_results=max_results)
        return list(results)

    def news_search(self, query: str, max_results: int = 10, region: str = "wt-wt") -> List[Dict[str, Union[str, None]]]:
        """
        Perform a news search using DuckDuckGo.

        Args:
            query (str): The news topic or keyword.
            max_results (int): Maximum number of news articles to return.
            region (str): Region code for localization.

        Returns:
            List[Dict[str, Union[str, None]]]: A list of news result dictionaries.
        """
        results = self.ddgs.news(query, region=region, max_results=max_results)
        return list(results)
