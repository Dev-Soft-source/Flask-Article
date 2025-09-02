# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

# Technical Plan: Paragraph Headings Implementation

**Date:** June 15, 2025  
**Author:** GitHub Copilot  
**Feature:** Implementation of paragraph-level headings in article generation system

## Implementation To-Do List

### Configuration Settings
- ✓ Add `enable_paragraph_headings` boolean flag (default: True)
- ✓ Add `max_paragraph_headings_per_section` integer parameter (default: 5)
- ✓ Add `refine_paragraph_headings` boolean flag (default: True)
- ✓ Add `variable_paragraph_headings` boolean flag (default: False)

### Prompt Templates
- ✓ Create `PARAGRAPH_WITH_HEADING_PROMPT` in Script1
- ✓ Create `PARAGRAPH_WITH_HEADING_PROMPT` in Script2
- ✓ Ensure prompts include clear format instructions for `[HEADING]` and `[CONTENT]` markers
- ✓ Add HTML tag formatting instructions for `<h4>` and `<p>` tags

### Core Implementation
- ✓ Implement `generate_paragraph_with_heading()` function in Script1
- ✓ Implement `generate_paragraph_with_heading()` method in Script2
- ✓ Update `generate_section()` in Script1 to use paragraph headings when enabled
- ✓ Update `_generate_sections()` in Script2 to use paragraph headings when enabled
- ✓ Implement regex parsing for extracting heading and content
- ✓ Add fallback mechanisms for parsing errors

### Formatting Functions
- ✓ Update WordPress formatting in Script1 to handle paragraph headings
- ✓ Update WordPress formatting in Script2 to handle paragraph headings
- ✓ Update Markdown conversion to handle paragraph headings
- ✓ Create helper function for parsing paragraph with heading

### Testing
- ❌ Test with paragraph headings enabled
- ❌ Test with paragraph headings disabled
- ❌ Test error scenarios with malformed LLM responses
- ❌ Validate HTML formatting in WordPress output

### Documentation
- ✓ Document new configuration parameters
- ✓ Create usage examples
- ✓ Document fallback mechanisms
- ✓ Add sample output examples

## Progress Tracking

| Date | Task | Status | Notes |
|------|------|--------|-------|
| 2025-06-15 | Add configuration settings to Script1 | ✓ | Added to config.py |
| 2025-06-15 | Add configuration settings to Script2 | ✓ | Added to config.py |
| 2025-06-15 | Add prompt templates | ✓ | Added to both scripts' prompts.py |
| 2025-06-15 | Implement generate_paragraph_with_heading | ✓ | Added to both scripts |
| 2025-06-15 | Update generate_section in Script1 | ✓ | Added conditional logic for paragraph headings |
| 2025-06-15 | Update _generate_sections in Script2 | ✓ | Added conditional logic for paragraph headings |
| 2025-06-15 | Implement regex parsing for heading/content | ✓ | Using pattern `r'\[HEADING\](.*?)(?=\[CONTENT\])'` |
| 2025-06-15 | Add fallback mechanisms | ✓ | Added fallbacks for parsing errors |
| 2025-06-15 | Update WordPress formatting in Script1 | ✓ | Added regex for `<h4>` and `<p>` pairs |
| 2025-06-15 | Update WordPress formatting in Script2 | ✓ | Added regex for `<h4>` and `<p>` pairs |
| 2025-06-15 | Update Markdown conversion | ✓ | Added support for paragraph headings in Markdown |
| 2025-06-15 | Create helper function for parsing | ✓ | Added `parse_paragraph_with_heading` to both scripts |
| 2025-06-15 | Improve heading quality with content-first approach | ✓ | Modified prompts and parsing to generate content before headings |
| 2025-06-15 | Add parse_paragraph_with_heading helper | ✓ | Added to both scripts' text_processor.py |
| 2025-06-15 | Update Markdown conversion | ✓ | Added convert_to_markdown function |
| 2025-06-15 | Create documentation | ✓ | Created paragraph_headings_usage_guide.md |

## 1. Overview

This document outlines the technical plan for implementing paragraph headings in the article generation system using a single API call approach. This approach generates both the paragraph content and its heading simultaneously in a single LLM call, ensuring efficiency while maintaining strong relevance between headings and content.

## 2. Basic Implementation Plan

1. Add configuration parameters to control paragraph heading generation
2. Create new prompts for generating paragraph content with headings in a single API call
3. Update content generation methods to use the single-call approach for paragraph and heading generation
4. Update the WordPress and Markdown formatting functions to handle paragraph headings
5. Add validation and error handling for parsing heading and content from the response
6. Implement testing to verify formatting and structure preservation

## 3. Current System Analysis

The current article generation system follows a hierarchical structure:

1. **Article Level**: The entire article with a main title
2. **Section Level**: Multiple sections, each with its own heading
3. **Paragraph Level**: Multiple paragraphs per section (defined by `paragraphs_per_section`)

Key findings from code analysis:

- Both Script1 and Script2 use section points from the outline to guide paragraph content
- Paragraph generation is already structured to handle multiple paragraphs per section
- The `generate_paragraph` method in ContentGenerator already supports context-aware generation
- WordPress formatting functions process paragraphs individually
- Existing heading hierarchy: H1 (title) → H2 (sections) → H3 (subsections)

## 4. Technical Implementation Details

### 4.1 Configuration Changes

#### Script1: `utils/config.py`

```python
@dataclass
class Config:
    # Existing parameters...
    
    # Paragraph Heading Settings
    enable_paragraph_headings: bool = True
    max_paragraph_headings_per_section: int = 5  # Maximum number of paragraph headings per section
    refine_paragraph_headings: bool = True  # Whether to allow the LLM to refine outline-based headings
    variable_paragraph_headings: bool = False  # Whether to use a variable number of headings
```

#### Script2: `config.py`

```python
# Paragraph Heading Settings
enable_paragraph_headings: bool = True
max_paragraph_headings_per_section: int = 5  # Maximum number of paragraph headings per section
refine_paragraph_headings: bool = True  # Whether to allow the LLM to refine outline-based headings
variable_paragraph_headings: bool = False  # Whether to use a variable number of headings
```

### 4.2 Prompt Changes

#### Script1: `prompts.py`

Add the paragraph and heading generation prompt:

```python
# Add to the existing prompts in prompts.py
PARAGRAPH_WITH_HEADING_PROMPT = """You are a seasoned SEO content writer with over a decade of experience crafting high-performing, keyword-optimized content for Fortune 500 companies and leading digital brands across diverse industries.

Your expertise lies in creating engaging, informative, and strategically structured content that consistently achieves top SERP rankings while maintaining exceptional readability and user engagement metrics.

Your task is to write paragraph {current_paragraph} of {paragraphs_per_section} for the section titled "{heading}" (Section {section_number} of {total_sections}). The content should be informative, engaging. Target {articleaudience} readers with appropriate complexity and examples.

Requirements:
1. Generate ONE well-structured paragraph
2. Focus on a single main point or aspect
3. Use {articlelanguage} language
4. Target the {articleaudience} audience
5. Maintain a {voicetone} tone
6. Use {pointofview} point of view
7. Include specific details and examples
8. Natural keyword integration
9. Clear topic sentence and conclusion
10. MAKE SURE TO RETURN YOUR RESPONSE IN EXACTLY THIS FORMAT:
[CONTENT] Your paragraph content here with proper formatting as needed...
[HEADING] Brief, descriptive heading that summarizes the above content

11. DO NOT use any markdown formatting in the content, only proper HTML tags for emphasis (<strong>, <em>)
12. The heading should directly relate to the content in the paragraph

The section needs to cover these points across {paragraphs_per_section} paragraphs:
{all_points}

For this specific paragraph ({current_paragraph} of {paragraphs_per_section}), focus on:
{current_points}

This is paragraph {current_paragraph} of {paragraphs_per_section} - structure your content accordingly.
{flow_instruction}

Write a cohesive paragraph with a relevant heading that educates and engages the reader while maintaining SEO optimization.
Make sure to make it look as human-like as possible, and please avoid any hyperbolic language.
"""
```

#### Script2: `prompts.py`

Add the paragraph and heading generation prompt:

```python
# Add to prompts.py
PARAGRAPH_WITH_HEADING_PROMPT = """
{context_summary}
You are now writing Paragraph {current_paragraph} of {paragraphs_per_section} for the section titled "{subtitle}" (Section {section_number} of {total_sections}) in the context of {keyword}.

Requirements:
1. Generate ONE well-structured paragraph
2. Focus on a single main point or aspect
3. Use {articlelanguage} language
4. Target the {articleaudience} audience
5. Maintain a {voicetone} tone
6. Use {pointofview} point of view
7. Include specific details and examples
8. Natural keyword integration
9. Clear topic sentence and conclusion
10. MAKE SURE TO RETURN YOUR RESPONSE IN EXACTLY THIS FORMAT:
[CONTENT] Your paragraph content here with proper formatting as needed...
[HEADING] Brief, descriptive heading that summarizes the above content

11. DO NOT use any markdown formatting in the content, only proper HTML tags for emphasis (<strong>, <em>)
12. The heading should directly relate to the content in the paragraph

The section needs to cover these points across {paragraphs_per_section} paragraphs:
{all_points}

For this specific paragraph ({current_paragraph} of {paragraphs_per_section}), focus on:
{current_points}

This is paragraph {current_paragraph} of {paragraphs_per_section} - structure your content accordingly.
{flow_instruction}

Write a cohesive paragraph with a relevant heading that educates and engages the reader while maintaining SEO optimization.
Make sure to make it look as human-like as possible, and please avoid any hyperbolic language.
"""
```

### 4.3 Implementation in Script1

#### Update `content_generator.py`

Modify the code to implement the single API call approach:

```python
def generate_paragraph_with_heading(
    context: ArticleContext,
    heading: str,
    keyword: str,
    current_paragraph: int = 1,
    paragraphs_per_section: int = None,
    section_number: int = 1,
    total_sections: int = 1,
    section_points: List[str] = None,
    web_context: str = ""
) -> str:
    """Generate paragraph content and its heading in a single API call."""
    logger.debug(f"Generating paragraph {current_paragraph}/{paragraphs_per_section} for: {heading}")
    
    # Set defaults if not provided
    if paragraphs_per_section is None:
        paragraphs_per_section = context.paragraphs_per_section
    
    # Default empty list for section points if none provided
    if section_points is None:
        section_points = []
    
    # Format all points as a string
    all_points_str = "\n".join([f"- {point}" for point in section_points])
    
    # Distribute points across paragraphs if there are enough points
    current_points = []
    if section_points and len(section_points) > 0:
        points_per_paragraph = max(1, len(section_points) // paragraphs_per_section)
        start_idx = (current_paragraph - 1) * points_per_paragraph
        end_idx = min(start_idx + points_per_paragraph, len(section_points))
        current_points = section_points[start_idx:end_idx]
    
    # Format current points as a string
    current_points_str = "\n".join([f"- {point}" for point in current_points]) if current_points else "- General information about this topic"
    
    # Adjust flow instruction based on position in section
    if current_paragraph == 1:
        flow_instruction = "This is the first paragraph for this section. Introduce the topic clearly and set the stage for the following paragraphs."
    elif current_paragraph == paragraphs_per_section:
        flow_instruction = "This is the last paragraph for this section. Provide a mini-conclusion for this section or a smooth transition to the next section."
    else:
        flow_instruction = f"This is paragraph {current_paragraph} of {paragraphs_per_section}. Ensure a smooth transition from previous content and develop the topic further."
    
    # Create format kwargs
    format_kwargs = {
        "keyword": keyword,
        "heading": heading,
        "section_number": section_number,
        "total_sections": total_sections,
        "paragraphs_per_section": paragraphs_per_section,
        "current_paragraph": current_paragraph,
        "voicetone": context.voicetone,
        "articletype": context.articletype,
        "articlelanguage": context.articlelanguage,
        "articleaudience": context.articleaudience,
        "pointofview": context.pointofview,
        "all_points": all_points_str,
        "current_points": current_points_str,
        "flow_instruction": flow_instruction
    }
    
    # Format the prompt
    try:
        prompt = PARAGRAPH_WITH_HEADING_PROMPT.format(**format_kwargs)
    except Exception as e:
        logger.warning(f"Error formatting paragraph prompt: {e}")
        # Fallback to a simpler prompt
        prompt = f"Write paragraph {current_paragraph} of {paragraphs_per_section} for section '{heading}' about {keyword}. Return in format: [HEADING] Heading text [CONTENT] Paragraph content"
    
    # Generate the paragraph with heading
    try:
        response = gpt_completion(
            context=context,
            prompt=prompt,
            temp=context.config.content_generation_temperature,
            max_tokens=context.config.paragraph_max_tokens,
            generation_type="paragraph"
        )
        
        # Parse the response to extract heading and content
        content_match = re.search(r'\[CONTENT\](.*)', response, re.DOTALL)
        heading_match = re.search(r'\[HEADING\](.*?)(?=\[CONTENT\])', response, re.DOTALL)
        
        if content_match and heading_match:
            paragraph_content = content_match.group(1).strip()
            paragraph_heading = heading_match.group(1).strip()
            
            # Format with HTML tags
            formatted_paragraph = f'<h4>{paragraph_heading}</h4>\n\n<p>{paragraph_content}</p>'
            return formatted_paragraph
        else:
            logger.warning(f"Failed to parse heading and content from response: {response}")
            # Fallback to using the entire response as content with a generic heading
            return f'<h4>About {keyword}</h4>\n\n<p>{response.strip()}</p>'
            
    except Exception as e:
        logger.error(f"Error generating paragraph with heading: {str(e)}")
        # Return a fallback paragraph with heading in case of error
        return f'<h4>About {keyword}</h4>\n\n<p>Information about {heading} related to {keyword}.</p>'
```

#### Update `generate_section` in `content_generator.py`

```python
def generate_section(
    context: ArticleContext,
    heading: str,
    keyword: str,
    section_number: int,
    total_sections: int,
    paragraph_prompt: str,
    parsed_sections: List[Dict[str, str]],
) -> str:
    """
    Generate content for a section of the article paragraph by paragraph.

    Args:
        context: Article context object
        heading: Section heading
        keyword: Main keyword for the article
        section_number: Current section number
        total_sections: Total number of sections
        paragraph_prompt: Template for paragraph generation
        parsed_sections: List of parsed section dictionaries

    Returns:
        Generated section content
    """
    logger.info(f"Generating section {section_number}/{total_sections}: {heading}")

    # Get subsection points for this section
    section_points = []
    for section in parsed_sections:
        if section["title"] == heading:
            section_points = section["subsections"]
            break

    if not section_points and len(parsed_sections) >= section_number:
        # Fallback: get points by index if title match fails
        section_points = parsed_sections[section_number - 1]["subsections"]

    # Use paragraph seed if seed control is enabled
    seed = context.config.paragraph_seed if context.config.enable_seed_control else None

    # Generate multiple paragraphs
    paragraphs = []
    for i in range(context.paragraphs_per_section):
        current_paragraph = i + 1
        
        if context.config.enable_paragraph_headings:
            # Generate paragraph with heading in a single API call
            formatted_paragraph = generate_paragraph_with_heading(
                context=context,
                heading=heading,
                keyword=keyword,
                current_paragraph=current_paragraph,
                paragraphs_per_section=context.paragraphs_per_section,
                section_number=section_number,
                total_sections=total_sections,
                section_points=section_points
            )
            paragraphs.append(formatted_paragraph)
            continue  # Skip the rest of the loop for this paragraph
        
        # Create format kwargs for this specific paragraph (only used for regular paragraphs)
        format_kwargs = {
            "keyword": keyword,
            "heading": heading,
            "section_number": section_number,
            "total_sections": total_sections,
            "paragraphs_per_section": context.paragraphs_per_section,
            "current_paragraph": current_paragraph,
            "min_paragraph_tokens": context.min_paragraph_tokens,
            "max_paragraph_tokens": context.max_paragraph_tokens,
            "voicetone": context.voicetone,
            "articletype": context.articletype,
            "articlelanguage": context.articlelanguage,
            "articleaudience": context.articleaudience,
            "pointofview": context.pointofview,
            "all_points": "\n".join([f"- {point}" for point in section_points]),
            "total_points": len(section_points),
        }

        # Format the prompt
        try:
            prompt = paragraph_prompt.format(**format_kwargs)
        except KeyError as e:
            logger.warning(f"Missing key in paragraph prompt: {e}, using simplified prompt")
            prompt = f"Write paragraph {current_paragraph} of {context.paragraphs_per_section} for section '{heading}' about {keyword}."

        # Add outline context for reference
        outline_context = "\n".join([
            f"# Article Title: {context.article_parts['title']}",
            f"# Article Outline:",
        ] + [f"Section {j+1}: {section['title']}" for j, section in enumerate(parsed_sections)])
        
        prompt = f"{prompt}\n\n{outline_context}"

        # Generate this paragraph
        try:
            paragraph = gpt_completion(
                context=context,
                prompt=prompt,
                generation_type="paragraph",
                seed=seed,
            )
            formatted_paragraph = f'<p>{paragraph}</p>'
        except Exception as e:
            logger.error(f"Error generating paragraph {current_paragraph} for '{heading}': {str(e)}")
            # Create a fallback paragraph if generation fails
            formatted_paragraph = f'<p>Information about {heading} related to {keyword}.</p>'
        
        paragraphs.append(formatted_paragraph)

    # Join all paragraphs with double newlines
    section_content = "\n\n".join(paragraphs)

    formatted_section = f"## {heading}\n\n{section_content}"

    # Store section in context
    context.article_parts["sections"].append(formatted_section)

    logger.success(f"Generated section {section_number}: {len(section_content)} chars")
    return formatted_section
```

### 4.4 Implementation in Script2

#### Update `content_generator.py`

Modify the code to implement the single API call approach:

```python
def generate_paragraph_with_heading(self, keyword: str, subtitle: str, current_paragraph: int = 1, paragraphs_per_section: int = None, section_number: int = 1, total_sections: int = 1, section_points: List[str] = None, web_context: str = "") -> str:
    """Generates a paragraph with heading for a specific subtitle in a single API call."""
    try:
        provider.debug(f"Generating paragraph {current_paragraph}/{paragraphs_per_section} for: {subtitle}")

        # Set defaults if not provided
        if paragraphs_per_section is None:
            paragraphs_per_section = self.config.paragraphs_per_section
            
        # Default empty list for section points if none provided
        if section_points is None:
            section_points = []
            
        # Distribute points across paragraphs if there are enough points
        if section_points and len(section_points) > 1 and paragraphs_per_section > 1:
            points_per_paragraph = max(1, len(section_points) // paragraphs_per_section)
            start_idx = (current_paragraph - 1) * points_per_paragraph
            end_idx = min(start_idx + points_per_paragraph, len(section_points))
            
            # Points for this specific paragraph
            current_points = section_points[start_idx:end_idx]
            
            # Format current points as a string
            current_points_str = "\n".join([f"- {point}" for point in current_points]) if current_points else "- General information about this topic"
        else:
            current_points_str = "- Cover relevant information for this paragraph"
        
        # Format all points as a string for overall context
        all_points_str = "\n".join([f"- {point}" for point in section_points]) if section_points else "- General information about this topic"

        # Adjust flow instruction based on position in section
        if current_paragraph == 1:
            flow_instruction = "This is the first paragraph for this section. Introduce the topic clearly and set the stage for the following paragraphs."
        elif current_paragraph == paragraphs_per_section:
            flow_instruction = "This is the last paragraph for this section. Provide a mini-conclusion for this section or a smooth transition to the next section."
        else:
            flow_instruction = f"This is paragraph {current_paragraph} of {paragraphs_per_section}. Ensure a smooth transition from previous content and develop the topic further."
        
        # Prepare prompt for paragraph with heading
        prompt = self.prompts.format_prompt(
            'paragraph_with_heading',
            context_summary=self.context.get_context_summary(),
            keyword=keyword,
            subtitle=subtitle,
            articlelanguage=self.config.articlelanguage,
            articleaudience=self.config.articleaudience,
            voicetone=self.config.voicetone,
            pointofview=self.config.pointofview,
            current_paragraph=current_paragraph,
            paragraphs_per_section=paragraphs_per_section,
            section_number=section_number,
            total_sections=total_sections,
            all_points=all_points_str,
            current_points=current_points_str,
            flow_instruction=flow_instruction,
            articletype=self.config.articletype,
        )

        if web_context != "":
            prompt += f"\n\nFollow this Web context and use the data in it: {web_context}"

        # Use paragraph seed if seed control is enabled
        seed = self.config.paragraph_seed if self.config.enable_seed_control else None

        # Generate the paragraph with heading
        engine = self.config.openrouter_model if (hasattr(self.config, 'use_openrouter') and self.config.use_openrouter) else self.config.openai_model
        
        self.context.add_message("user", prompt)
        response = generate_completion(
            prompt=prompt,
            model=engine,
            temperature=self.config.content_generation_temperature,
            max_tokens=self.config.paragraph_max_tokens,
            article_context=self.context,
            seed=seed
        )
        
        # Parse the response to extract heading and content
        content_match = re.search(r'\[CONTENT\](.*)', response, re.DOTALL)
        heading_match = re.search(r'\[HEADING\](.*?)(?=\[CONTENT\])', response, re.DOTALL)
        
        if content_match and heading_match:
            paragraph_content = content_match.group(1).strip()
            paragraph_heading = heading_match.group(1).strip()
            
            # Format with HTML tags
            formatted_paragraph = f'<h4>{paragraph_heading}</h4>\n\n<p>{paragraph_content}</p>'
            return formatted_paragraph
        else:
            provider.warning(f"Failed to parse heading and content from response: {response}")
            # Fallback to using the entire response as content with a generic heading
            return f'<h4>About {keyword}</h4>\n\n<p>{response.strip()}</p>'
            
    except Exception as e:
        provider.error(f"Error generating paragraph with heading: {str(e)}")
        provider.error(f"Stack trace:\n{traceback.format_exc()}")
        return f'<h4>About {keyword}</h4>\n\n<p>Information about {subtitle} related to {keyword}.</p>'
```

#### Update `_generate_sections` in `generator.py`

```python
def _generate_sections(
    self,
    keyword: str,
    article_data: Dict[str, Any],
    heading: str,
    web_context: str = ""
) -> Optional[str]:
    """Generate content for a section with multiple paragraphs."""
    try:
        # Extract section points if available
        section_points = []
        
        # Look for section points in the parsed outline
        if "parsed_outline" in article_data and article_data["parsed_outline"]:
            parsed_outline = article_data["parsed_outline"]
            for section in parsed_outline:
                if section["title"].lower() == heading.lower():
                    section_points = section["subsections"]
                    break
        
        # Find the section number
        section_number = 1
        total_sections = len(article_data.get("headings", []))
        
        for i, h in enumerate(article_data.get("headings", [])):
            if h.lower() == heading.lower():
                section_number = i + 1
                break
        
        # Generate paragraphs with headings using a single API call
        paragraphs = []
        for i in range(self.config.paragraphs_per_section):
            current_paragraph = i + 1
            
            if self.config.enable_paragraph_headings:
                # Generate paragraph with heading in a single call
                formatted_paragraph = self.content_generator.generate_paragraph_with_heading(
                    keyword, 
                    heading, 
                    current_paragraph=current_paragraph,
                    paragraphs_per_section=self.config.paragraphs_per_section,
                    section_number=section_number,
                    total_sections=total_sections,
                    section_points=section_points,
                    web_context=web_context
                )
            else:
                # Use existing paragraph generation without headings
                paragraph_content = self.content_generator.generate_paragraph(
                    keyword, 
                    heading, 
                    current_paragraph=current_paragraph,
                    paragraphs_per_section=self.config.paragraphs_per_section,
                    section_number=section_number,
                    total_sections=total_sections,
                    section_points=section_points,
                    web_context=web_context
                )
                formatted_paragraph = f'<p>{paragraph_content}</p>'
            
            paragraphs.append(formatted_paragraph)
        
        # Join all paragraphs with double newlines
        section_content = "\n\n".join(paragraphs)
        
        # Return the generated section content
        return section_content
        
    except Exception as e:
        provider.error(f"Error generating section content: {str(e)}")
        provider.error(f"Stack trace:\n{traceback.format_exc()}")
        return None
```

### 4.5 Updating WordPress Formatting

#### Script1: `text_processor.py`

Update the WordPress formatting function to handle paragraph headings:

```python
def format_article_for_wordpress(
    article_dict: Dict[str, str],
    youtube_position: str = "after_introduction",
    body_images: Optional[List[Dict[str, str]]] = None,
    add_summary: bool = False  # Add parameter for summary control
) -> str:
    # ...existing code...
    
    # 3. Add sections (Body)
    for section in article_dict.get('sections', []):
        if section.strip():
            # Extract section heading
            lines = section.split('\n')
            heading = lines[0].strip()
            
            # Remove heading markers if present
            if heading.startswith('#'):
                heading = heading.lstrip('#').strip()
            
            # Add heading as h2
            content.append(f'<!-- wp:heading -->\n<h2>{heading}</h2>\n<!-- /wp:heading -->\n')
            
            # Process section content
            section_content = '\n'.join(lines[1:]).strip()
            
            # Parse section content looking for paragraph headings (<h4>) and paragraphs (<p>)
            pattern = r'<h4>(.*?)</h4>\s*<p>(.*?)</p>'
            matches = re.findall(pattern, section_content, re.DOTALL)
            
            if matches:
                # We found paragraph headings and paragraphs
                for para_heading, para_content in matches:
                    # Add paragraph heading as h4
                    content.append(f'<!-- wp:heading {{"level":4}} -->\n<h4>{para_heading}</h4>\n<!-- /wp:heading -->\n')
                    
                    # Add paragraph content
                    content.append(f'<!-- wp:paragraph -->\n<p>{para_content}</p>\n<!-- /wp:paragraph -->\n')
            else:
                # Fallback to old paragraph processing if no heading/paragraph pairs are found
                paragraphs = section_content.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        content.append(f'<!-- wp:paragraph -->\n<p>{paragraph.strip()}</p>\n<!-- /wp:paragraph -->\n')
    
    # ...rest of the function remains the same...
```

#### Script2: `text_processor.py`

Update the WordPress formatting function to handle paragraph headings:

```python
def format_article_for_wordpress(
    article_dict: Dict[str, str],
    youtube_position: str = "after_introduction",
    body_images: Optional[List[Dict[str, str]]] = None,
    add_summary: bool = False,  # Add parameter for summary control
    add_block_notes: bool = True  # Add parameter for block notes control (default True)
) -> str:
    # ...existing code...
    
    # Process main sections with headings and images (Body)
    if 'headings' in article_dict and 'sections' in article_dict:
        for i, (heading, section) in enumerate(zip(article_dict['headings'], article_dict['sections'])):
            # Add section heading
            content.append(f'<!-- wp:heading -->\n<h2>{heading}</h2>\n<!-- /wp:heading -->\n')
            
            # Add body image if available
            # ...existing image handling code...
            
            # Parse section content looking for paragraph headings (<h4>) and paragraphs (<p>)
            pattern = r'<h4>(.*?)</h4>\s*<p>(.*?)</p>'
            matches = re.findall(pattern, section, re.DOTALL)
            
            if matches:
                # We found paragraph headings and paragraphs
                for para_heading, para_content in matches:
                    # Add paragraph heading as h4
                    content.append(f'<!-- wp:heading {{"level":4}} -->\n<h4>{para_heading}</h4>\n<!-- /wp:heading -->\n')
                    
                    # Add paragraph content
                    content.append(f'<!-- wp:paragraph -->\n<p>{para_content}</p>\n<!-- /wp:paragraph -->\n')
            else:
                # Fallback to old paragraph processing if no heading/paragraph pairs are found
                paragraphs = section.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        # Check if paragraph starts with a subheading (###)
                        if paragraph.startswith('###'):
                            subheading = paragraph.replace('###', '').strip()
                            content.append(f'<!-- wp:heading {{"level":3}} -->\n<h3>{subheading}</h3>\n<!-- /wp:heading -->\n')
                        else:
                            content.append(f'<!-- wp:paragraph -->\n<p>{paragraph.strip()}</p>\n<!-- /wp:paragraph -->\n')
            
            # ...rest of the section processing code...
    
    # ...rest of the function remains the same...
```

### 4.6 Updating Markdown Formatting

For any functions that convert content to Markdown, update them to handle the paragraph headings:

```python
def convert_to_markdown(content: str) -> str:
    """Convert content with HTML paragraph headings to Markdown."""
    # Parse content looking for paragraph headings (<h4>) and paragraphs (<p>)
    pattern = r'<h4>(.*?)</h4>\s*<p>(.*?)</p>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    markdown_content = []
    for para_heading, para_content in matches:
        # Convert HTML heading to Markdown (#### for h4)
        markdown_content.append(f"#### {para_heading}")
        markdown_content.append("")  # Empty line for proper Markdown formatting
        
        # Clean HTML tags from paragraph content
        para_content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', para_content)
        para_content = re.sub(r'<em>(.*?)</em>', r'*\1*', para_content)
        para_content = re.sub(r'<[^>]+>', '', para_content)  # Remove any other HTML tags
        
        markdown_content.append(para_content)
        markdown_content.append("")  # Empty line between paragraph blocks
    
    return "\n".join(markdown_content)
```

## 5. Error Handling and Validation

Add error handling to properly parse paragraph content with headings:

```python
def _parse_paragraph_with_heading(content: str) -> Tuple[str, str]:
    """Parse content to extract paragraph heading and content."""
    pattern = r'<h4>(.*?)</h4>\s*<p>(.*?)</p>'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        heading = match.group(1).strip()
        paragraph = match.group(2).strip()
        return heading, paragraph
    
    # Fallback if the format is not correct
    return "Additional Information", content.strip()
```

## 6. Implementation To-Do List

### Configuration Settings
- [x] Add `enable_paragraph_headings` boolean flag (default: True)
- [x] Add `max_paragraph_headings_per_section` integer parameter (default: 5)
- [x] Add `refine_paragraph_headings` boolean flag (default: True)
- [x] Add `variable_paragraph_headings` boolean flag (default: False)

### Prompt Templates
- [x] Create `PARAGRAPH_WITH_HEADING_PROMPT` in Script1
- [x] Create `PARAGRAPH_WITH_HEADING_PROMPT` in Script2
- [x] Ensure prompts include clear format instructions for `[HEADING]` and `[CONTENT]` markers
- [x] Add HTML tag formatting instructions for `<h4>` and `<p>` tags

### Core Implementation
- [x] Implement `generate_paragraph_with_heading()` function in Script1
- [x] Implement `generate_paragraph_with_heading()` method in Script2
- [x] Update `generate_section()` in Script1 to use paragraph headings when enabled
- [x] Update `_generate_sections()` in Script2 to use paragraph headings when enabled
- [x] Implement regex parsing for extracting heading and content
- [x] Add fallback mechanisms for parsing errors

### Formatting Functions
- [x] Update WordPress formatting in Script1 to handle paragraph headings
- [x] Update WordPress formatting in Script2 to handle paragraph headings
- [x] Update Markdown conversion to handle paragraph headings
- [x] Create helper function for parsing paragraph with heading

### Testing
- [x] Create test cases for paragraph headings functionality
- [ ] Run tests with paragraph headings enabled
- [ ] Run tests with paragraph headings disabled
- [ ] Test error scenarios with malformed LLM responses
- [ ] Validate HTML formatting in WordPress output

### Documentation
- [x] Document new configuration parameters
- [x] Create usage examples
- [x] Document fallback mechanisms
- [x] Add sample output examples

## Progress Tracking

| Date | Task | Status | Notes |
|------|------|--------|-------|

<!-- Original content starts below -->

## 7. Improvement: Content-First Paragraph Headings

### 7.1 Problem Statement

The initial implementation instructed the language model to generate headings before content, which may lead to less accurate headings as the model must predict what it will write before writing it.

### 7.2 Solution: Reversed Generation Order

To improve the quality of paragraph headings, we've modified the approach to have the language model generate content first, then create a heading that accurately summarizes that content.

#### Changes Required:

1. **Prompt Template Modifications:**
   - Updated format instructions to request content before headings:
   ```
   [CONTENT] Your paragraph content here...
   [HEADING] Brief, descriptive heading that summarizes the above content
   ```

2. **Parsing Function Updates:**
   - Modified regex patterns to extract content first, then heading
   - Updated the order of extraction in both scripts

3. **Helper Function Enhancements:**
   - Added fallback for raw LLM output to the parsing helper
   - Modified Markdown conversion to preserve content-first order

#### Benefits:

- More accurate headings that better represent the content
- Better alignment with how LLMs generate text sequentially
- Improved reading experience with more representative headings

This change is a prompt engineering improvement that doesn't affect the overall architecture but should result in higher quality paragraph headings.

## 8. Implementation Notes and Bug Fixes

### 8.1 Error Fixes During Testing (June 15, 2025)

During testing of the paragraph headings implementation, two critical issues were encountered and resolved:

#### Issue 1: Format Variables Scope Problem

**Problem:** When paragraph headings were enabled, an `UnboundLocalError` occurred with the message "cannot access local variable 'format_kwargs' where it is not associated with a value". This happened in the `generate_section` function.

**Solution:** 
1. The structure of the `generate_section` function was modified to properly handle the conditional flow when paragraph headings are enabled.
2. The `format_kwargs` variable was moved outside the conditional block to ensure it's available regardless of whether paragraph headings are enabled.
3. A `continue` statement was added after appending the formatted paragraph to skip the rest of the loop for that paragraph when using paragraph headings.

**Code Change:**
```python
if context.config.enable_paragraph_headings:
    # Generate paragraph with heading in a single API call
    formatted_paragraph = generate_paragraph_with_heading(
        # ... parameters ...
    )
    paragraphs.append(formatted_paragraph)
    continue  # Skip the rest of the loop for this paragraph
        
# Create format kwargs for this specific paragraph (only used for regular paragraphs)
format_kwargs = {
    # ... format parameters ...
}
```

#### Issue 2: Response Parsing Edge Cases

**Problem:** The LLM sometimes did not follow the exact tag format requested in the prompt, leading to failures in parsing the content and heading from the response.

**Solution:**
1. Improved the parsing logic to handle various edge cases:
   - When standard format with tags is correctly used
   - When heading tag exists but content tag doesn't (content might be before the heading tag)
   - When no tags are used at all (extract a heading from the content)

2. Updated the regex patterns to be more flexible in parsing LLM responses.

3. Enhanced fallback mechanisms to ensure a properly formatted paragraph is always returned, even if parsing fails.

**Code Change:**
```python
# Parse the response to extract content and heading
content_match = re.search(r'\[CONTENT\](.*)', response, re.DOTALL)
heading_match = re.search(r'\[HEADING\](.*)', response, re.DOTALL)

if content_match and heading_match:
    # Standard format - tags are correctly used
    # ... process normally ...
else:
    # Check if there's a standalone [HEADING] tag but no [CONTENT] tag
    if heading_match and not content_match:
        # Content might be before the [HEADING] tag
        potential_content = response.split('[HEADING]')[0].strip()
        # ... use this content ...
    
    # If we still can't parse correctly, extract a heading from the content
    if len(response) > 20:
        # Use the first sentence or phrase as the heading
        # ... extract heading and use rest as content ...
```

### 8.2 Future Improvement Recommendations

For future iterations of the paragraph headings feature, consider the following improvements:

#### HTML Tag Format Instead of Square Brackets

**Recommendation:** Replace the current square bracket tags (`[CONTENT]`, `[HEADING]`) with HTML-style tags that would be more natural for the LLM to understand and less ambiguous.

**Proposed Implementation:**
1. Update the prompt to request content in HTML format directly:
```
<paragraph>Your paragraph content here with proper formatting as needed...</paragraph>
<heading>Brief, descriptive heading that summarizes the above content</heading>
```

2. Update the parsing regex to match HTML tags instead of square brackets:
```python
content_match = re.search(r'<paragraph>(.*?)</paragraph>', response, re.DOTALL)
heading_match = re.search(r'<heading>(.*?)</heading>', response, re.DOTALL)
```

**Benefits:**
- HTML tags are more natural for the LLM to understand since they're used extensively in web content
- The opening and closing tag structure is less ambiguous than square brackets
- The LLM is already instructed to use HTML formatting for emphasis, so extending this to structure is consistent
- Reduced likelihood of the LLM ignoring or misinterpreting the format instructions

This recommendation should be considered for the next iteration of the paragraph headings feature, as it would require updates to both the prompt templates and the parsing logic.

### 8.3 Additional Fixes (June 16, 2025)

#### Issue 3: Config Token Limits Attribute Error

**Problem:** When trying to generate a paragraph with heading, an `AttributeError` was thrown: "'Config' object has no attribute 'token_limits'". This occurred in the `generate_paragraph_with_heading` function when attempting to get the max tokens for paragraph generation.

**Solution:**
1. The error was fixed by replacing the non-existent `token_limits` dictionary lookup with the appropriate direct config attribute.
2. Changed `context.config.token_limits.get('paragraph', 700)` to `context.config.paragraph_max_tokens`.

**Code Change:**
```python
# Before:
response = gpt_completion(
    context=context,
    prompt=prompt,
    generation_type="paragraph_with_heading",
    temp=context.config.content_generation_temperature,
    max_tokens=context.config.token_limits.get('paragraph', 700),
    seed=seed
)

# After:
response = gpt_completion(
    context=context,
    prompt=prompt,
    generation_type="paragraph_with_heading",
    temp=context.config.content_generation_temperature,
    max_tokens=context.config.paragraph_max_tokens,
    seed=seed
)
```

This fix ensures the proper configuration value is used for token limits in paragraph heading generation. The `Config` class directly provides specific token limit attributes for different content types rather than storing them in a dictionary.
