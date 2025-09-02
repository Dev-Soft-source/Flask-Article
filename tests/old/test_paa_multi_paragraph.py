#!/usr/bin/env python3
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

"""
Test script for PAA multi-paragraph functionality.

This script tests:
1. Generation of PAA answers with multiple paragraphs based on configuration
2. Proper formatting and structure of paragraphs
3. Appropriate word count scaling based on paragraph count
"""

import sys
import os
import time
from datetime import datetime

# Add script paths to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script2'))

# Import required modules
from config import Config
from article_generator.paa_handler import PAAHandler
from article_generator.article_context import ArticleContext
from utils.rich_provider import provider
from utils.prompts_config import Prompts

def load_prompts():
    """Load prompts from the prompts.py file."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("prompts", os.path.join(os.path.dirname(__file__), '..', 'script2', 'prompts.py'))
        prompts_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompts_module)
        
        prompts_globals = prompts_module.__dict__
        
        prompts = Prompts(
            outline=prompts_globals.get('OUTLINE_PROMPT', ''),
            title=prompts_globals.get('TITLE_PROMPT', ''),
            title_craft=prompts_globals.get('TITLE_CRAFT_PROMPT', ''),
            introduction=prompts_globals.get('INTRODUCTION_PROMPT', ''),
            paragraph=prompts_globals.get('PARAGRAPH_PROMPT', ''),
            conclusion=prompts_globals.get('CONCLUSION_PROMPT', ''),
            faq=prompts_globals.get('FAQ_PROMPT', ''),
            grammar=prompts_globals.get('GRAMMAR_CHECK_PROMPT', ''),
            humanize=prompts_globals.get('HUMANIZE_PROMPT', ''),
            blocknote=prompts_globals.get('BLOCKNOTE_PROMPT', ''),
            summarize=prompts_globals.get('SUMMARIZE_PROMPT', ''),
            paa_answer=prompts_globals.get('PAA_ANSWER_PROMPT', '')
        )
        return prompts
    except Exception as e:
        provider.error(f"Error loading prompts: {str(e)}")
        sys.exit(1)

def test_paa_multi_paragraph(paragraph_count=2, verbose=True):
    """Test PAA answer generation with multiple paragraphs."""
    provider.info(f"Testing PAA multi-paragraph functionality with {paragraph_count} paragraphs")
    
    # Initialize configuration
    config = Config()
    config.paragraphs_per_section = paragraph_count
    
    # Load prompts
    prompts = load_prompts()
    
    # Create article context
    article_context = ArticleContext(config=config, prompts=prompts)
    
    # Initialize PAA handler
    paa_handler = PAAHandler(config)
    
    # Test questions
    test_questions = [
        ("What are the benefits of regular exercise?", "healthy lifestyle"),
        ("How does artificial intelligence impact society?", "artificial intelligence"),
        ("What are the most effective ways to learn a new language?", "language learning")
    ]
    
    results = []
    
    for question, keyword in test_questions:
        provider.info(f"Testing question: '{question}' for keyword '{keyword}'")
        
        try:
            # Generate answer with detailed timing
            start_time = datetime.now()
            answer = paa_handler.generate_answer_for_question(question, keyword, article_context)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Check results
            paragraphs = [p for p in answer.split("\n\n") if p.strip()]
            word_count = len(answer.split())
            
            result = {
                "question": question,
                "keyword": keyword,
                "paragraphs": len(paragraphs),
                "word_count": word_count,
                "duration": duration,
                "success": len(paragraphs) == paragraph_count
            }
            results.append(result)
            
            if verbose:
                provider.print(f"\n[bold green]Question:[/] {question}")
                provider.print(f"[bold blue]Answer ([/][bold]{len(paragraphs)}[/][bold blue] paragraphs, [/][bold]{word_count}[/][bold blue] words):[/]")
                provider.print(f"[cyan]{answer}[/]")
                provider.print(f"[yellow]Generated in {duration:.2f}s[/]")
                
                if len(paragraphs) != paragraph_count:
                    provider.warning(f"Expected {paragraph_count} paragraphs, got {len(paragraphs)}")
            
        except Exception as e:
            import traceback
            provider.error(f"Error testing question '{question}': {str(e)}")
            provider.error(f"Error details: {traceback.format_exc()}")
            results.append({
                "question": question,
                "keyword": keyword,
                "success": False,
                "error": str(e)
            })
    
    # Print summary
    provider.print("\n[bold]Test Results Summary:[/]")
    for result in results:
        if result.get("success", False):
            provider.print(f"[green]✓[/] {result['question']} - {result.get('paragraphs', 0)} paragraphs, {result.get('word_count', 0)} words")
        else:
            provider.print(f"[red]✗[/] {result['question']} - {result.get('error', 'Unknown error')}")
    
    success_count = sum(1 for r in results if r.get("success", False))
    provider.print(f"\n[bold]{'Overall Success' if success_count == len(results) else 'Partial Success'}:[/] {success_count}/{len(results)} tests passed")
    
    return results

def main():
    """Main function to parse arguments and run the test."""
    import argparse
    parser = argparse.ArgumentParser(description='Test PAA multi-paragraph functionality')
    parser.add_argument('--paragraphs', type=int, default=2, help='Number of paragraphs to generate per answer')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    test_paa_multi_paragraph(args.paragraphs, not args.quiet)

if __name__ == "__main__":
    main()
