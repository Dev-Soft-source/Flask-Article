# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import openai
from typing import List, Dict, Optional
import sys
import os
import time
import re

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from article_generator.content_generator import ArticleContext, gpt_completion, make_openrouter_api_call
from .logger import logger
from utils.rate_limiter import openai_rate_limiter


def generate_faq_section(
    context: ArticleContext,
    keyword: str,
    faq_prompt: str,
    *,
    openai_engine: str,
    enable_token_tracking: bool = False,
    track_token_usage: bool = False,
    web_context: Optional[str] = "",
) -> str:
    """
    Generate a complete FAQ section with multiple Q&As.
    
    Args:
        context (ArticleContext): Article generation context
        keyword (str): Main article keyword
        faq_prompt (str): Template for FAQ generation
        engine (str): OpenAI engine to use
        enable_token_tracking (bool): Whether to track token usage
        track_token_usage (bool): Whether to display token usage info
    Returns:
        str: Formatted FAQ section
    """
    logger.info(f"Generating FAQ section for keyword: {keyword}")
    time.sleep(2)  # Sleep to avoid hitting the rate limit too quickly
    
    try:
        # Format the FAQ prompt with the keyword
        logger.debug("Formatting FAQ prompt")
        formatted_prompt = faq_prompt.format(
            keyword=keyword,
            faqs=keyword,
            articlanguage="English",
            articleaudience=context.config.articleaudience,
            
        )

        if web_context and web_context.strip() and web_context.strip() != "":
            formatted_prompt = 'Follow the web context for all the information and data used to answer this question: '+ web_context + '\n\n' + formatted_prompt
        
        # Track token usage if enabled
        if enable_token_tracking:
            tokens_used = len(context.encoding.encode(formatted_prompt))
            if track_token_usage:
                logger.debug(f"Token usage - Request: {tokens_used}")
        
        # Check if OpenRouter is enabled and make API call accordingly
        if context.config.use_openrouter and context.config.openrouter_api_key:
            logger.debug("Sending request to OpenRouter API for FAQ generation")
            
            # Use the configured OpenRouter model
            openrouter_model = context.config.openrouter_model
            
            response = make_openrouter_api_call(
                messages=[{"role": "user", "content": formatted_prompt}],
                model=openrouter_model,
                api_key=context.config.openrouter_api_key,
                site_url=context.config.openrouter_site_url,
                site_name=context.config.openrouter_site_name,
                temperature=context.config.faq_generation_temperature,
                max_tokens=1000,  # Reasonable length for FAQs
                top_p=context.config.faq_generation_top_p,
                frequency_penalty=context.config.faq_generation_frequency_penalty,
                presence_penalty=context.config.faq_generation_presence_penalty
            )
        else:
            # Get completion from OpenAI using specified engine
            logger.debug("Sending request to OpenAI API for FAQ generation")
            
            # Use rate limiter if available
            if context.config.enable_rate_limiting and openai_rate_limiter:
                logger.debug("Using rate limiter for OpenAI API call")
                
                def make_api_call():
                    return openai.chat.completions.create(
                        model=openai_engine,
                        messages=[{"role": "user", "content": formatted_prompt}],
                        temperature=context.config.faq_generation_temperature,
                        max_tokens=1000,  # Reasonable length for FAQs
                        top_p=context.config.faq_generation_top_p,
                        frequency_penalty=context.config.faq_generation_frequency_penalty,
                        presence_penalty=context.config.faq_generation_presence_penalty
                    )
                    
                response = openai_rate_limiter.execute_with_rate_limit(make_api_call)
            else:
                response = openai.chat.completions.create(
                    model=openai_engine,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=context.config.faq_generation_temperature,
                    max_tokens=1000,  # Reasonable length for FAQs
                    top_p=context.config.faq_generation_top_p,
                    frequency_penalty=context.config.faq_generation_frequency_penalty,
                    presence_penalty=context.config.faq_generation_presence_penalty
                )
        
        faq_text = response.choices[0].message.content.strip()
        
        # Track token usage for response if enabled
        if enable_token_tracking and track_token_usage:
            response_tokens = len(context.encoding.encode(faq_text))
            logger.debug(f"Token usage - Response: {response_tokens}")
        
        logger.debug(f"Generated FAQ text (length: {len(faq_text)} characters)")
        
        # Process the FAQ text to ensure proper formatting
        faq_items = []
        
        # First, try to parse using Q&A format
        current_question = None
        current_answer = []
        
        # Check if the text contains FAQ or Frequently Asked Questions header
        if faq_text.lower().startswith("faq") or faq_text.lower().startswith("frequently asked questions"):
            # Remove the header line
            faq_text = "\n".join(faq_text.split("\n")[1:]).strip()
        
        # Parse the FAQ text line by line
        lines = faq_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            
            if not line:
                continue
                
            # Check if this is a question line
            is_question = False
            
            # Check for explicit Q: format
            if line.startswith('Q:') or line.startswith('Question:'):
                is_question = True
                if line.startswith('Q:'):
                    current_question = line[2:].strip()
                else:
                    current_question = line[9:].strip()
            
            # Check for numbered questions (1., 2., etc.)
            elif re.match(r'^\d+\.', line) and ('?' in line):
                is_question = True
                current_question = re.sub(r'^\d+\.', '', line).strip()
            
            # Check for questions ending with ?
            elif line.endswith('?'):
                is_question = True
                current_question = line
            
            # Check for bold/strong text that might be a question
            elif (line.startswith('**') and line.endswith('**')) or (line.startswith('<strong>') and line.endswith('</strong>')):
                is_question = True
                current_question = line.replace('**', '').replace('<strong>', '').replace('</strong>', '')
            
            # If we found a question and already have a previous question, save it
            if is_question and current_answer and len(faq_items) > 0:
                faq_items[-1]['answer'] = '\n'.join(current_answer)
                current_answer = []
            
            # If we found a question, add it to the items
            if is_question:
                faq_items.append({'question': current_question, 'answer': ''})
                current_answer = []
                continue
            
            # Check if this is an explicit answer line
            if line.startswith('A:') or line.startswith('Answer:'):
                if line.startswith('A:'):
                    answer_text = line[2:].strip()
                else:
                    answer_text = line[7:].strip()
                    
                if answer_text and faq_items:
                    current_answer.append(answer_text)
            
            # Otherwise, it's part of the current answer if we have a question
            elif faq_items:
                current_answer.append(line)
        
        # Add the last answer if there is one
        if current_answer and faq_items:
            faq_items[-1]['answer'] = '\n'.join(current_answer)
        
        # If we couldn't parse using Q&A format, try to parse using a different approach
        if not faq_items:
            logger.debug("Couldn't parse using Q&A format, trying alternative parsing")
            
            # Try to split by double newlines which often separate Q&A pairs
            qa_pairs = faq_text.split('\n\n')
            
            for i in range(0, len(qa_pairs) - 1, 2):
                if i + 1 < len(qa_pairs):
                    question = qa_pairs[i].strip()
                    answer = qa_pairs[i + 1].strip()
                    
                    # Only add if the question looks like a question
                    if '?' in question or question.lower().startswith('how') or question.lower().startswith('what') or question.lower().startswith('why'):
                        faq_items.append({'question': question, 'answer': answer})
        
        # If we still couldn't parse, try one more approach - look for question marks
        if not faq_items:
            logger.debug("Trying to parse by finding question marks")
            
            # Split the text into sentences
            sentences = re.split(r'(?<=[.!?])\s+', faq_text)
            
            i = 0
            while i < len(sentences):
                sentence = sentences[i].strip()
                
                # If this sentence ends with a question mark, it's likely a question
                if sentence.endswith('?'):
                    question = sentence
                    answer_parts = []
                    
                    # Collect all following sentences until we hit another question
                    j = i + 1
                    while j < len(sentences) and not sentences[j].strip().endswith('?'):
                        answer_parts.append(sentences[j].strip())
                        j += 1
                    
                    answer = ' '.join(answer_parts)
                    if answer:
                        faq_items.append({'question': question, 'answer': answer})
                    
                    i = j
                else:
                    i += 1
        
        # Format the FAQ items as HTML
        logger.debug(f"Parsed {len(faq_items)} FAQ items")
        
        # Check if we have any FAQ items
        if not faq_items:
            logger.warning("No FAQ items were generated")
            # As a fallback, just format the entire text as a single FAQ item
            if faq_text:
                # Try to find a reasonable split for question and answer
                parts = faq_text.split('\n', 1)
                if len(parts) > 1:
                    question = parts[0].strip()
                    answer = parts[1].strip()
                else:
                    question = f"Frequently Asked Questions about {keyword}"
                    answer = faq_text
                
                faq_items = [{'question': question, 'answer': answer}]
            else:
                return ""
        
        # Format the FAQ section for WordPress
        # Don't include the heading - let the text_processor handle that
        faq_html = ''
        
        for item in faq_items:
            question = item['question'].strip()
            answer = item['answer'].strip()
            
            # Format the question and answer using Q: and A: format
            faq_html += f"Q: {question}\n\nA: {answer}\n\n"
        
        logger.success(f"FAQ section generated successfully with {len(faq_items)} items")
        return faq_html.strip()
        
    except Exception as e:
        logger.error(f"Error generating FAQ section: {str(e)}")
        return ""

def generate_faqs(context, keyword, article_content, faq_prompt_template):
    """Generate FAQs for the article."""
    logger.info("Generating FAQs...")
    
    try:
        # Format the FAQ prompt
        prompt = faq_prompt_template.format(
            keyword=keyword,
            article_content=article_content,
            articleaudience=context.articleaudience
        )
        
        # Use faq seed if seed control is enabled
        seed = context.config.faq_seed if context.config.enable_seed_control else None
        
        # Generate FAQs
        faqs = gpt_completion(
            context,
            prompt,
            temp=context.config.faq_generation_temperature,
            generation_type="faq_generation",
            seed=seed
        )
        
        if not faqs:
            logger.warning("Generated FAQs were empty")
            return ""
            
        logger.success(f"Generated FAQs successfully")
        return faqs.strip()
        
    except Exception as e:
        logger.error(f"Error generating FAQs: {str(e)}")
        return "" 