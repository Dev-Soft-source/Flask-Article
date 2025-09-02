# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from serpapi import GoogleSearch
from typing import List, Dict, Optional
import sys
import os
import json
import hashlib
from datetime import datetime, timedelta
from .logger import logger
import time
from utils.rate_limiter import serpapi_rate_limiter
from utils.error_utils import ErrorHandler

# Initialize error handler
error_handler = ErrorHandler()

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PAACache:
    """Handles caching of PAA questions to reduce API calls."""

    def __init__(self):
        """Initialize the cache system."""
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache', 'paa')
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"PAA cache initialized at: {self.cache_dir}")

    def _get_cache_key(self, keyword: str) -> str:
        """Generate a cache key for a keyword."""
        return hashlib.md5(keyword.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _is_cache_valid(self, cache_data: Dict) -> bool:
        """Check if cached data is still valid (less than 24 hours old)."""
        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        return datetime.now() - cache_time < timedelta(hours=24)

    def get(self, keyword: str) -> Optional[List[str]]:
        """Try to get PAA questions from cache."""
        try:
            cache_key = self._get_cache_key(keyword)
            cache_path = self._get_cache_path(cache_key)

            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                if self._is_cache_valid(cache_data):
                    logger.info(f"Using cached PAA data for: '{keyword}'")
                    return cache_data['questions']

            return None
        except Exception as e:
            logger.warning(f"Cache read error: {str(e)}")
            return None

    def save(self, keyword: str, questions: List[str]) -> None:
        """Save PAA questions to cache."""
        try:
            cache_key = self._get_cache_key(keyword)
            cache_path = self._get_cache_path(cache_key)

            cache_data = {
                'keyword': keyword,
                'questions': questions,
                'timestamp': datetime.now().isoformat()
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.debug(f"Cached PAA data for: '{keyword}'")
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")

# Initialize global cache instance
paa_cache = PAACache()

def get_paa_questions(keyword: str, serp_api_key: str, config=None) -> List[str]:
    """
    Get "People Also Ask" questions from Google using SerpAPI.

    Args:
        keyword (str): Search keyword
        serp_api_key (str): SerpAPI key
        config: Configuration object with PAA settings
    Returns:
        List[str]: List of PAA questions
    """
    logger.info(f"Getting PAA questions for keyword: {keyword}")

    # Try to get from cache first
    cached_questions = paa_cache.get(keyword)
    if cached_questions:
        # Apply config-based filtering to cached results if config is provided
        if config:
            import random
            if config.paa_use_random_range:
                num_questions = random.randint(config.paa_min_questions, config.paa_max_questions)
            else:
                num_questions = config.paa_max_questions
            return cached_questions[:num_questions]
        return cached_questions

    try:
        # Set up search parameters
        params = {
            "engine": "google",
            "q": keyword,
            "api_key": serp_api_key,
            "gl": "us",  # Set to US results
            "hl": "en"   # Set to English
        }

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

        # Extract PAA questions (only the questions, not answers)
        if "related_questions" in results:
            questions = []
            for q in results["related_questions"]:
                if "question" in q:
                    questions.append(q["question"])
            logger.success(f"Found {len(questions)} PAA questions")

            # Cache the results
            paa_cache.save(keyword, questions)

            # Apply config-based filtering if config is provided
            if config:
                import random
                if config.paa_use_random_range:
                    num_questions = random.randint(config.paa_min_questions, config.paa_max_questions)
                    logger.info(f"Using random range: selecting {num_questions} questions out of {len(questions)}")
                else:
                    num_questions = config.paa_max_questions
                    logger.info(f"Using max questions: selecting {num_questions} questions out of {len(questions)}")
                
                # Randomly select questions if we have more than needed
                if len(questions) > num_questions:
                    questions = random.sample(questions, num_questions)
                else:
                    questions = questions[:num_questions]
                
                logger.info(f"Final PAA questions count: {len(questions)}")

            return questions
        else:
            logger.warning("No PAA questions found in search results")
            return []

    except Exception as e:
        logger.error(f"Error getting PAA questions: {str(e)}")
        return []

def generate_answer_for_question(context, question: str, keyword: str, web_context:str = None) -> str:
    """
    Generate an answer for a PAA question using GPT.

    Args:
        context: Article generation context
        question (str): The question to answer
        keyword (str): Main article keyword for context
    Returns:
        str: Generated answer
    """
    from article_generator.content_generator import gpt_completion

    try:
        # Get paragraphs_per_section from context's config
        paragraphs_per_section = getattr(context.config, 'paragraphs_per_section', 2)
        
        # Adjust max tokens based on number of paragraphs
        max_tokens = paragraphs_per_section * 200
        
        # Calculate approximate word count for all paragraphs
        paragraphs_word_count = paragraphs_per_section * 100
        
        prompt_start = ""
        if web_context:
            prompt_start = 'Follow the web context for all the information and data used to answer this question: '+ web_context
        
        prompt = f"""{prompt_start}
        \n\nMake sure to read the context before answering the question.
        You are writing an answer for a "People Also Ask" section in an article about "{keyword}".
        The question is: "{question}"

        Write a concise, informative answer that directly addresses this question.
        The answer should be helpful, accurate, and provide value to the reader.
        Format your answer in {paragraphs_per_section} paragraph(s) using plain text paragraphs separated by double line breaks.
        Each paragraph should be about 80-120 words for a total of approximately {paragraphs_word_count} words.

        Remember to:
        - Be direct and answer the question clearly
        - Structure your answer logically across {paragraphs_per_section} paragraph(s)
        - Include relevant facts or details
        - Write in a natural, engaging style
        - Maintain the context of the main article topic: "{keyword}"
        - Use appropriate tone and language for the audience
        - Use only plain text paragraphs - NO HTML tags apart from `<strong>`
        - Make sure to only answer with reference to the provided context, nothing by yourself alone.
        - Think of this context as 100% correct.
        - `<strong>`: Emphasize 1-2 key phrases/keywords per paragraph.
        
        Structure guidelines for a {paragraphs_per_section}-paragraph answer:
        - First paragraph: Directly address the question and provide the core answer
        - Middle paragraph(s): Expand with details, examples, or supporting information
        - Last paragraph: Summarize or provide final insights/recommendations
        """

        # save prompt to a file prompt.txt
        with open("prompt.txt", "w") as f:
            f.write(prompt)
            logger.debug(f"Prompt written to prompt.txt")

        # Generate answer using GPT with adjusted max_tokens - using standard paragraph generation
        # Note: We're explicitly using "content_generation" type to avoid paragraph headings
        answer = gpt_completion(
            context=context,
            prompt=prompt,
            generation_type="content_generation",  # Use content_generation to avoid paragraph headings
            max_tokens=max_tokens
        )

        error_handler.handle_error(Exception(answer), severity="info")

        return answer.strip()

    except Exception as e:
        error_handler.handle_error(e, severity="error")
        return f"Sorry, we couldn't generate an answer for this question at this time."

def generate_paa_section(
    keyword: str,
    serp_api_key: str,
    context=None,
    web_context=None,
    config=None,
) -> str:
    """
    Generate a complete PAA section with questions and GPT-generated answers.

    Args:
        keyword (str): Main article keyword
        serp_api_key (str): SerpAPI key
        context: Article generation context
        web_context: Additional web context for answers
        config: Configuration object with PAA settings
    Returns:
        str: Formatted PAA section
    """
    logger.info(f"Generating PAA section for keyword: {keyword}")

    time.sleep(2)  # Sleep to avoid hitting API rate limits

    # print context in file context.txt
    if web_context:
        with open("context.txt", "w", encoding="utf-8") as f:
            f.write(str(web_context))
            logger.debug(f"Context written to context.txt")

    try:
        # Get PAA questions
        questions = get_paa_questions(keyword, serp_api_key, config)

        if not questions:
            logger.warning("No PAA questions found")
            return ""

        # Format the PAA section with GPT-generated answers
        logger.debug("Formatting PAA section with GPT-generated answers")

        # Create the main section with a clear hierarchy - using ## for the main heading to fit with article structure
        paa_section = "## People Also Ask\n\n"

        for i, question_item in enumerate(questions, 1):
            # Handle both string and dictionary formats (for backward compatibility with cache)
            if isinstance(question_item, dict) and 'question' in question_item:
                question = question_item['question']
            else:
                question = question_item

            if context:
                # Generate answer using GPT
                if web_context:
                    answer = generate_answer_for_question(context, question, keyword, web_context)
                else:
                    answer = generate_answer_for_question(context, question, keyword)

                # Format question with bigger font (h3) and bold for emphasis
                paa_section += f"### **{question}**\n\n"
                paa_section += f"{answer}\n\n"

                logger.debug(f"Added GPT-generated answer for question {i}/{len(questions)}")
            else:
                # If no context is provided, just add placeholder
                logger.warning("No context provided for GPT answer generation")
                paa_section += f"### **{question}**\n\n"
                paa_section += "Answer not available.\n\n"

        logger.success(f"PAA section generation complete ({len(questions)} Q&A pairs)")
        return paa_section.strip()

    except Exception as e:
        error_handler.handle_error(e, severity="error")
        return ""