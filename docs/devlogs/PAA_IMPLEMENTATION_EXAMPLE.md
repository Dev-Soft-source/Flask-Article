# PAA Multi-Paragraph Implementation Example

This document provides a specific implementation example showing the exact code changes needed to implement the multi-paragraph PAA answers feature for `script2`. This can be used as a reference when making the actual code changes.

## Example Implementation for script2

### 1. Changes to prompts.py

```python
# Before:
PAA_ANSWER_PROMPT = """You are writing an answer for a "People Also Ask" section in an article about "{keyword}".
The question is: "{question}"

Write a concise, informative paragraph that directly answers this question.
The answer should be helpful, accurate, and provide value to the reader.
Keep the answer to a single paragraph with about 100-150 words.

Remember to:
- Be direct and answer the question clearly
- Include relevant facts or details
- Write in a natural, engaging style
- Maintain the context of the main article topic: "{keyword}"
- Adjust tone to match: {tone}
- Use appropriate language: {language}
- Target audience expertise level: {audience}
- Use appropriate {pov} point of view
- NO color or extra formatting other than just strong, and em.
"""

# After:
PAA_ANSWER_PROMPT = """You are writing an answer for a "People Also Ask" section in an article about "{keyword}".
The question is: "{question}"

Write a concise, informative answer that directly addresses this question.
The answer should be structured into {paragraphs_per_section} paragraphs.
Each paragraph should flow naturally and provide value to the reader.
Total length should be about {paragraphs_per_section * 120} words.

Remember to:
- Be direct and answer the question clearly
- Include relevant facts or details
- Write in a natural, engaging style
- Maintain the context of the main article topic: "{keyword}"
- Adjust tone to match: {tone}
- Use appropriate language: {language}
- Target audience expertise level: {audience}
- Use appropriate {pov} point of view
- Ensure a logical flow between paragraphs
- NO color or extra formatting other than just strong, and em.
"""
```

### 2. Changes to article_generator/paa_handler.py

```python
# Before:
def generate_answer_for_question(self, question: str, keyword: str, article_context=None, web_context: str = "") -> str:
    """Generate an answer for a PAA question using GPT."""
    try:
        tone = getattr(self.config, 'voicetone', 'neutral')
        # Format prompt template if available
        if hasattr(article_context, 'prompts') and article_context.prompts:
            prompt = article_context.prompts.format_prompt(
                'paa_answer',
                question=question,
                keyword=keyword,
                tone=tone,
                language=self.config.articlelanguage,
                audience=self.config.articleaudience,
                pov=self.config.pointofview
            )
        else:
            # Create a simple, reliable prompt for PAA answers
            prompt = f"""
            {web_context & f'Follow the web context for all the information and data: {web_context}'}
            You are writing an answer for a "People Also Ask" section in an article about "{keyword}".
            The question is: "{question}"

            Write a concise, informative paragraph that directly answers this question.
            The answer should be helpful, accurate, and provide value to the reader.
            Keep the answer to a single paragraph with about 100-150 words.

            Remember to:
            - Be direct and answer the question clearly
            - Include relevant facts or details
            - Write in a natural, engaging style
            - Maintain the context of the main article topic: "{keyword}"
            - Use a {tone} tone
            - Write in {self.config.articlelanguage} language for {self.config.articleaudience} audience
            - Use {self.config.pointofview} point of view
            """
        
        # Generate answer using GPT
        # Determine which model to use based on whether OpenRouter is enabled
        model_to_use = self.config.openrouter_model if (hasattr(self.config, 'use_openrouter') and self.config.use_openrouter and self.config.openrouter_api_key) else self.config.openai_model
        
        answer = generate_completion(
            prompt=prompt,
            model=model_to_use,
            temperature=self.config.content_generation_temperature,
            max_tokens=200,
            article_context=article_context
        )
        
        return answer.strip()

# After:
def generate_answer_for_question(self, question: str, keyword: str, article_context=None, web_context: str = "") -> str:
    """Generate an answer for a PAA question using GPT."""
    try:
        tone = getattr(self.config, 'voicetone', 'neutral')
        paragraphs_per_section = getattr(self.config, 'paragraphs_per_section', 2)
        # Format prompt template if available
        if hasattr(article_context, 'prompts') and article_context.prompts:
            prompt = article_context.prompts.format_prompt(
                'paa_answer',
                question=question,
                keyword=keyword,
                tone=tone,
                language=self.config.articlelanguage,
                audience=self.config.articleaudience,
                pov=self.config.pointofview,
                paragraphs_per_section=paragraphs_per_section
            )
        else:
            # Create a simple, reliable prompt for PAA answers with multiple paragraphs
            prompt = f"""
            {web_context & f'Follow the web context for all the information and data: {web_context}'}
            You are writing an answer for a "People Also Ask" section in an article about "{keyword}".
            The question is: "{question}"

            Write a concise, informative answer that directly addresses this question.
            The answer should be structured into {paragraphs_per_section} paragraphs.
            Each paragraph should flow naturally and provide value to the reader.
            Total length should be about {paragraphs_per_section * 120} words.

            Remember to:
            - Be direct and answer the question clearly
            - Include relevant facts or details
            - Write in a natural, engaging style
            - Maintain the context of the main article topic: "{keyword}"
            - Use a {tone} tone
            - Write in {self.config.articlelanguage} language for {self.config.articleaudience} audience
            - Use {self.config.pointofview} point of view
            - Ensure a logical flow between paragraphs
            """
        
        # Generate answer using GPT with increased max_tokens to accommodate multiple paragraphs
        model_to_use = self.config.openrouter_model if (hasattr(self.config, 'use_openrouter') and self.config.use_openrouter and self.config.openrouter_api_key) else self.config.openai_model
        
        # Calculate max tokens based on paragraph count
        max_tokens = paragraphs_per_section * 200  # Approximately 200 tokens per paragraph
        
        answer = generate_completion(
            prompt=prompt,
            model=model_to_use,
            temperature=self.config.content_generation_temperature,
            max_tokens=max_tokens,
            article_context=article_context
        )
        
        return answer.strip()
```

## Test Case Example

Here's a simple test case that can be used to verify the implementation:

```python
def test_paa_multi_paragraph_answers():
    """Test generating PAA answers with multiple paragraphs."""
    # Initialize configuration
    config = Config()
    config.paragraphs_per_section = 3  # Set to generate 3 paragraphs
    
    # Initialize PAA handler
    paa_handler = PAAHandler(config)
    
    # Test question
    question = "What are the benefits of regular exercise?"
    keyword = "healthy lifestyle"
    
    # Generate answer
    answer = paa_handler.generate_answer_for_question(question, keyword)
    
    # Check result
    paragraphs = answer.split("\n\n")
    assert len(paragraphs) == 3, f"Expected 3 paragraphs, got {len(paragraphs)}"
    
    # Verify each paragraph has reasonable content
    for i, paragraph in enumerate(paragraphs):
        assert len(paragraph.split()) >= 30, f"Paragraph {i+1} too short: {len(paragraph.split())} words"
        assert len(paragraph.split()) <= 200, f"Paragraph {i+1} too long: {len(paragraph.split())} words"
    
    print(f"✅ PAA multi-paragraph test passed with {len(paragraphs)} paragraphs")
    print(f"Total word count: {len(answer.split())}")
    return True
```

This implementation preserves all existing functionality while adding the ability to generate multiple paragraphs for PAA answers that match the article's configured paragraph style.
