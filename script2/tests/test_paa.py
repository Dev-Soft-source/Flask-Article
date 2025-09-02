#!/usr/bin/env python3

# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import sys
import os
import argparse
from datetime import datetime
from config import Config
from utils.prompts_config import Prompts
from article_generator.paa_handler import PAAHandler
from article_generator.article_context import ArticleContext
from utils.rich_provider import provider

def load_prompts():
    """Load prompts from the prompts.py file"""
    try:
        prompts_file = os.path.join(os.path.dirname(__file__), 'prompts.py')
        prompts_globals = {}
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_content = f.read()
            exec(prompts_content, prompts_globals)
        
        # Create Prompts object with the loaded values
        prompts = Prompts(
            title=prompts_globals.get('TITLE_PROMPT', ''),
            outline=prompts_globals.get('OUTLINE_PROMPT', ''),
            introduction=prompts_globals.get('INTRODUCTION_PROMPT', ''),
            paragraph=prompts_globals.get('PARAGRAPH_PROMPT', ''),
            conclusion=prompts_globals.get('CONCLUSION_PROMPT', ''),
            faq=prompts_globals.get('FAQ_PROMPT', ''),
            system_message=prompts_globals.get('SYSTEM_MESSAGE', ''),
            meta_description=prompts_globals.get('META_DESCRIPTION_PROMPT', ''),
            wordpress_excerpt=prompts_globals.get('WORDPRESS_EXCERPT_PROMPT', ''),
            grammar=prompts_globals.get('GRAMMAR_CHECK_PROMPT', ''),
            humanize=prompts_globals.get('HUMANIZE_PROMPT', ''),
            blocknote=prompts_globals.get('BLOCK_NOTES_PROMPT', ''),
            summarize=prompts_globals.get('SUMMARIZE_PROMPT', ''),
            paa_answer=prompts_globals.get('PAA_ANSWER_PROMPT', '')
        )
        return prompts
    except Exception as e:
        provider.error(f"Error loading prompts: {str(e)}")
        sys.exit(1)

def test_paa_handler(keyword, num_questions=3, with_context=True, verbose=False):
    """Test the PAA handler functionality"""
    provider.info(f"Testing PAA handler with keyword: '{keyword}'")
    
    # Initialize configuration
    config = Config()
    
    # Load prompts
    prompts = load_prompts()
    
    # Create article context if needed
    article_context = None
    if with_context:
        article_context = ArticleContext(config=config, prompts=prompts)
        provider.info("Created article context with prompts")
        provider.info("Will use PAA_ANSWER_PROMPT from prompts.py")
    else:
        provider.info("Running without article context - will use hardcoded prompt")
    
    # Initialize PAA handler
    paa_handler = PAAHandler(config)
    provider.info("PAA handler initialized")
    
    # Print configuration
    provider.info(f"Voice tone: {config.voicetone}")
    provider.info(f"Article language: {config.articlelanguage}")
    provider.info(f"Article audience: {config.articleaudience}")
    provider.info(f"Point of view: {config.pointofview}")
    provider.info(f"OpenAI model: {config.openai_model}")
    
    try:
        # Test fetching PAA questions
        provider.info(f"Fetching PAA questions for: '{keyword}'")
        questions = paa_handler.get_paa_questions(keyword, num_questions)
        
        if not questions:
            provider.error("No PAA questions found")
            return
        
        provider.success(f"Found {len(questions)} PAA questions")
        for i, question in enumerate(questions, 1):
            provider.info(f"Question {i}: {question}")
        
        # Test generating answers for each question
        provider.info("Generating answers for each question...")
        
        for i, question in enumerate(questions, 1):
            provider.info(f"Processing question {i}: {question}")
            
            try:
                # Generate answer with detailed timing
                start_time = datetime.now()
                
                # If verbose, get prompt first to display it
                if verbose and hasattr(article_context, 'prompts') and article_context.prompts:
                    try:
                        prompt = article_context.prompts.format_prompt(
                            'paa_answer',
                            question=question,
                            keyword=keyword,
                            tone=config.voicetone,
                            language=config.articlelanguage,
                            audience=config.articleaudience,
                            pov=config.pointofview
                        )
                        provider.print(f"[yellow]Using prompt:[/]\n{prompt}")
                    except Exception as e:
                        provider.warning(f"Could not format prompt: {str(e)}")
                
                answer = paa_handler.generate_answer_for_question(question, keyword, article_context)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                if answer and not answer.startswith("Sorry"):
                    provider.success(f"Generated answer in {duration:.2f}s")
                    provider.print(f"[bold green]Q: {question}[/]")
                    provider.print(f"[cyan]A: {answer}[/]")
                else:
                    provider.warning(f"Failed to generate answer in {duration:.2f}s")
                    provider.print(f"[bold red]Q: {question}[/]")
                    provider.print(f"[red]A: {answer}[/]")
            except Exception as e:
                import traceback
                provider.error(f"Error generating answer for question '{question}': {str(e)}")
                provider.error(f"Error details: {traceback.format_exc()}")
        
        # Test generating full PAA section
        provider.info("Generating complete PAA section...")
        start_time = datetime.now()
        paa_section = paa_handler.generate_paa_section(keyword, num_questions, article_context)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if paa_section:
            provider.success(f"Generated PAA section in {duration:.2f}s")
            provider.print(f"[green]PAA Section Preview:[/]")
            provider.print(f"{paa_section[:500]}...")
        else:
            provider.error("Failed to generate PAA section")
    
    except Exception as e:
        import traceback
        provider.error(f"Error in PAA testing: {str(e)}")
        provider.error(f"Error details: {traceback.format_exc()}")

def main():
    """Main function to parse arguments and run the test"""
    parser = argparse.ArgumentParser(description='Test PAA Handler')
    parser.add_argument('keyword', help='Keyword to test PAA generation')
    parser.add_argument('--questions', type=int, default=3, help='Number of PAA questions to generate')
    parser.add_argument('--no-context', action='store_true', help='Run without article context')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output including prompts')
    
    args = parser.parse_args()
    
    test_paa_handler(args.keyword, args.questions, not args.no_context, args.verbose)

if __name__ == "__main__":
    main() 