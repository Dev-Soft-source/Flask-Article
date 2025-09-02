# PAA Enhancement Technical Plan - Multiple Paragraphs per Answer

## Overview

This document outlines the technical plan to address client feedback regarding the People Also Ask (PAA) section generation. The client has requested that PAA answers should be formatted with the same number of paragraphs per answer as configured for the main article sections (using the `paragraphs_per_section` configuration parameter).

## Current Implementation Analysis

### How PAA Currently Works

The PAA functionality in both scripts follows this general flow:

1. SerpAPI is used to fetch "People Also Ask" questions related to the article keyword
2. The system generates answers for each question using the LLM
3. Currently, answers are formatted as **single paragraphs** (100-150 words)
4. These Q&A pairs are formatted into a Markdown section
5. When humanization is enabled, the entire PAA section is processed
6. The formatted content is then processed for WordPress output

### Configuration Parameters

Both scripts already include a `paragraphs_per_section` parameter:

- `script1/utils/config.py` - Line ~189: `paragraphs_per_section: int = 2`
- `script2/config.py` - Line ~192: `paragraphs_per_section: int = 2`

This parameter is used throughout the codebase to control how many paragraphs are generated for each article section, but it isn't currently applied to PAA answers.

## Technical Implementation

### Files to Modify

1. **PAA Handler Scripts**:
   - `/home/abuh/Documents/Python/LLM_article_gen_2/scripts/script1/article_generator/paa_handler.py`
   - `/home/abuh/Documents/Python/LLM_article_gen_2/scripts/script2/article_generator/paa_handler.py`

2. **Prompt Templates**:
   - `/home/abuh/Documents/Python/LLM_article_gen_2/scripts/script1/prompts.py`
   - `/home/abuh/Documents/Python/LLM_article_gen_2/scripts/script2/prompts.py`

### Implementation Details

#### 1. Modify PAA Answer Prompts

Both scripts need updated prompt templates that instruct the LLM to generate multiple paragraphs. The key changes will be:

**For script2/prompts.py**:
- Update `PAA_ANSWER_PROMPT` to incorporate the `paragraphs_per_section` parameter
- Change instructions from "single paragraph" to multiple paragraphs based on configuration
- Include instructions for proper paragraph distribution (balanced content)

**For script1/prompts.py**:
- Add a dedicated PAA answer prompt if one doesn't exist
- Ensure it includes the same multiple paragraph capability

#### 2. Update PAA Handler Functions

**For script2/article_generator/paa_handler.py**:
- Modify `generate_answer_for_question()` to pass the `paragraphs_per_section` parameter to the prompt
- Update the max_tokens value to accommodate multiple paragraphs (increase from 200 to ~300-400)

**For script1/article_generator/paa_handler.py**:
- Similar changes to the `generate_answer_for_question()` function
- Update max_tokens to accommodate multiple paragraphs

#### 3. Ensure Formatting Consistency

Both scripts need to ensure that:
- Paragraphs are properly separated in the generated PAA sections
- The formatting matches the rest of the article
- Humanization preserves the paragraph structure

## Detailed Code Changes

### 1. Update PAA_ANSWER_PROMPT in script2/prompts.py

```python
# PAA answer generation prompt
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

### 2. Update generate_answer_for_question() in script2/article_generator/paa_handler.py

```python
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
                pov=self.config.pointofview,
                paragraphs_per_section=self.config.paragraphs_per_section
            )
        else:
            # Create a simple, reliable prompt for PAA answers with multiple paragraphs
            prompt = f"""
            {web_context & f'Follow the web context for all the information and data: {web_context}'}
            You are writing an answer for a "People Also Ask" section in an article about "{keyword}".
            The question is: "{question}"

            Write a concise, informative answer that directly addresses this question.
            The answer should be structured into {self.config.paragraphs_per_section} paragraphs.
            Each paragraph should flow naturally and provide value to the reader.
            Total length should be about {self.config.paragraphs_per_section * 120} words.

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
        
        answer = generate_completion(
            prompt=prompt,
            model=model_to_use,
            temperature=self.config.content_generation_temperature,
            max_tokens=self.config.paragraphs_per_section * 200,  # Dynamically scale based on paragraph count
            article_context=article_context
        )
        
        return answer.strip()
```

### 3. Similar Changes for script1

The same approach will be applied to the script1 implementation, adapting to its specific structure.

## Testing Plan

1. **Unit Testing**:
   - Test PAA answer generation with different `paragraphs_per_section` values (1, 2, 3)
   - Verify paragraph separation and structure
   - Check total word count scales appropriately

2. **Integration Testing**:
   - Generate full articles with PAA sections
   - Verify PAA formatting matches article section formatting
   - Test with humanization enabled to ensure paragraph structure is preserved

3. **Regression Testing**:
   - Ensure other features still work correctly
   - Verify token usage remains efficient
   - Check compatibility with existing articles

## Implementation To-Do List

- [ ] 📝 Create a detailed technical plan document ✅
- [ ] 🔍 Review current PAA implementation in both scripts ✅
- [ ] 📌 Update PAA_ANSWER_PROMPT in script2/prompts.py ✅
- [ ] 🛠️ Add/Update PAA prompt in script1/prompts.py ✅
- [ ] 🔄 Modify generate_answer_for_question() in script2 ✅
- [ ] 🔄 Modify generate_answer_for_question() in script1 ✅
- [ ] 🧪 Implement unit tests for PAA paragraph generation ✅
- [ ] 📄 Update documentation in PAA_USER_GUIDE.md ✅
- [ ] ✅ Final testing and quality assurance ✅

## Conclusion

This implementation will enhance the PAA section by making the answers match the formatting style of the main article sections. By leveraging the existing `paragraphs_per_section` configuration parameter, we maintain consistency across the codebase while adding the requested functionality. The changes are focused and minimal, affecting only the PAA generation process without disrupting other aspects of the article generation system.
