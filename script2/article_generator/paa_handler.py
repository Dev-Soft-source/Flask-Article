# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from typing import List, Optional, Dict, Any
import json
import os
import hashlib
from datetime import datetime, timedelta
from utils.api_utils import SerpAPI
from utils.text_utils import TextProcessor
from article_generator.logger import logger
from config import Config
from utils.ai_utils import generate_completion

class PAAHandler:
    """Handles 'People Also Ask' content generation using SERP API and GPT."""
    
    def __init__(self, config: Config):
        """Initialize the PAA handler with configuration."""
        self.config = config
        self.serp_api = SerpAPI(config)
        self.text_processor = TextProcessor(config)
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache', 'paa')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_key(self, keyword: str) -> str:
        """Generate a cache key for a keyword."""
        return hashlib.md5(keyword.encode()).hexdigest()
        
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
        
    def _is_cache_valid(self, cache_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid (less than 24 hours old)."""
        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        return datetime.now() - cache_time < timedelta(hours=24)
        
    def _get_from_cache(self, keyword: str) -> Optional[List[str]]:
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
            
    def _save_to_cache(self, keyword: str, questions: List[str]) -> None:
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
                
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")
            
    def _extract_question(self, question: Dict[str, Any]) -> Optional[str]:
        """Extract and validate a question from PAA response."""
        try:
            if not isinstance(question, dict):
                return None
                
            q_text = question.get('question', '').strip()
            
            if not q_text:
                return None
                
            return q_text
            
        except Exception:
            return None
            
    def get_paa_questions(self, keyword: str, num_questions: int = 5) -> List[str]:
        """Fetches related questions from Google's 'People Also Ask' section."""
        if not self.config.add_paa_paragraphs_into_article:
            logger.info("PAA generation is disabled in configuration")
            return []
            
        # Calculate actual number of questions to fetch based on config
        if hasattr(self.config, 'paa_use_random_range') and self.config.paa_use_random_range:
            import random
            min_questions = getattr(self.config, 'paa_min_questions', 3)
            max_questions = getattr(self.config, 'paa_max_questions', 5)
            num_questions = random.randint(min_questions, max_questions)
            logger.info(f"Using random range for PAA questions: {num_questions} (range: {min_questions}-{max_questions})")
        else:
            # Use the max_questions from config if available, otherwise use the passed parameter
            num_questions = getattr(self.config, 'paa_max_questions', num_questions)
            
        logger.info(f"Fetching PAA questions for: '{keyword}' (target: {num_questions} questions)")
        
        # Try to get from cache first
        cached_questions = self._get_from_cache(keyword)
        if cached_questions:
            # Apply config-based filtering to cached results
            if len(cached_questions) >= num_questions:
                if hasattr(self.config, 'paa_use_random_range') and self.config.paa_use_random_range:
                    import random
                    selected_questions = random.sample(cached_questions, num_questions)
                    logger.info(f"Randomly selected {len(selected_questions)} questions from {len(cached_questions)} cached questions")
                    return selected_questions
                else:
                    return cached_questions[:num_questions]
            else:
                logger.info(f"Cached questions ({len(cached_questions)}) fewer than requested ({num_questions}), using all cached")
                return cached_questions
        
        try:
            results = self.serp_api.perform_search(keyword)
            if not results:
                logger.warning(f"No search results found for: '{keyword}'")
                return []
                
            if 'related_questions' not in results:
                logger.warning(f"No PAA questions found for: '{keyword}'")
                return []
                
            questions = []
            for q in results['related_questions']:
                question = self._extract_question(q)
                if question:
                    questions.append(question)
                    
            if questions:
                # Cache the results
                self._save_to_cache(keyword, questions)
                
                # Apply config-based selection logic
                if len(questions) >= num_questions:
                    if hasattr(self.config, 'paa_use_random_range') and self.config.paa_use_random_range:
                        import random
                        final_questions = random.sample(questions, num_questions)
                        logger.info(f"Randomly selected {len(final_questions)} questions from {len(questions)} fetched questions")
                    else:
                        final_questions = questions[:num_questions]
                        logger.info(f"Selected first {len(final_questions)} questions from {len(questions)} fetched questions")
                else:
                    final_questions = questions
                    logger.info(f"Using all {len(final_questions)} fetched questions (fewer than requested {num_questions})")
                
                # Display selected questions
                logger.success(f"Final PAA questions:\n" + "\n".join([f"- {q}" for q in final_questions]))
                return final_questions
            else:
                logger.warning(f"No valid PAA questions found for: '{keyword}'")
                return []
            
        except Exception as e:
            logger.error(f"Error fetching PAA questions: {str(e)}", show_traceback=True)
            return []
            
    def generate_answer_for_question(self, question: str, keyword: str, article_context=None, web_context: str = "") -> str:
        """Generate an answer for a PAA question using GPT."""
        try:
            tone = getattr(self.config, 'voicetone', 'neutral')
            paragraphs_per_section = getattr(self.config, 'paragraphs_per_section', 2)
            
            # Calculate max tokens based on number of paragraphs (about 150 words per paragraph)
            max_tokens = paragraphs_per_section * 200
            
            # Calculate approximate word count for all paragraphs
            paragraphs_word_count = paragraphs_per_section * 100
            
            # Format prompt template if available
            if hasattr(article_context, 'prompts') and article_context.prompts:
                prompt = article_context.prompts.format_prompt(
                    'paa_answer',
                    question=question,
                    keyword=keyword,
                    voicetone=tone,
                    articlelanguage=self.config.articlelanguage,
                    articleaudience=self.config.articleaudience,
                    pov=self.config.pointofview,
                    paragraphs=paragraphs_per_section,
                    paragraphs_word_count=paragraphs_word_count
                )
            else:
                # Create a simple, reliable prompt for PAA answers
                prompt = f"""
                {web_context and f'Follow the web context for all the information and data: {web_context}'}
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
                - Use a {tone} tone
                - Write in {self.config.articlelanguage} language for {self.config.articleaudience} audience
                - Use {self.config.pointofview} point of view
                - Use only plain text paragraphs - NO HTML tags apart from `<strong>`
                - `<strong>`: Emphasize 1-2 key phrases/keywords per paragraph.

                
                Structure guidelines for a {paragraphs_per_section}-paragraph answer:
                - First paragraph: Directly address the question and provide the core answer
                - Middle paragraph(s): Expand with details, examples, or supporting information
                - Last paragraph: Summarize or provide final insights/recommendations
                """
            
            # Generate answer using GPT
            # Determine which model to use based on whether OpenRouter is enabled
            model_to_use = self.config.openrouter_model if (hasattr(self.config, 'use_openrouter') and self.config.use_openrouter and self.config.openrouter_api_key) else self.config.openai_model
            
            answer = generate_completion(
                prompt=prompt,
                model=model_to_use,
                temperature=self.config.content_generation_temperature,
                max_tokens=max_tokens,
                article_context=article_context
            )
            
            return answer.strip()
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Error generating answer for question '{question}': {str(e)}\nError details: {error_trace}")
            return f"Sorry, we couldn't generate an answer for this question at this time."
            
    def generate_paa_section(self, keyword: str, num_questions: int = 5, article_context=None, web_context: str = "") -> Optional[str]:
        """Generates a complete PAA section with questions from SERP API and answers from GPT."""
        if not self.config.add_paa_paragraphs_into_article:
            logger.info("PAA generation is disabled in configuration")
            return None
            
        try:
            # Get PAA questions using config parameters
            questions = self.get_paa_questions(keyword, num_questions)
            if not questions:
                return None
                
            # Initialize content string (don't add heading here - let text_processor handle it)
            paa_content = ""
            
            # Add section heading - the main WordPress formatter will add the heading too
            # We'll add this as a marker for our content
            paa_content += "# People Also Ask\n\n"
            
            # Add each Q&A pair
            for question in questions:
                # Add markdown formatted question (will be processed correctly by text_processor)
                paa_content += f"## {question}\n\n"
                
                # Generate answer with GPT if context is provided
                if article_context:
                    if web_context:
                        answer = self.generate_answer_for_question(question, keyword, article_context, web_context)
                    else:
                        answer = self.generate_answer_for_question(question, keyword, article_context)
                    # Add answer paragraph (will be processed correctly by text_processor)
                    paa_content += f"{answer}\n\n"
                else:
                    # Fallback if no context is provided
                    logger.warning("No article context provided for GPT answer generation")
                    paa_content += "Answer not available at this time.\n\n"
                
            return paa_content
            
        except Exception as e:
            logger.error(f"Error generating PAA section: {str(e)}", show_traceback=True)
            return None