# Prompt Refinement Plan: Natural & Efficient LLM Prompts

## Current Problems Identified

### 1. Excessive Strictness & Verbosity
- **OUTLINE_PROMPT**: 104 lines of rigid formatting rules
- **PARAGRAPH_PROMPT**: 234 lines of micro-management
- **TITLE_PROMPT**: 26 lines of constraints for a simple title
- Multiple "MUST", "NO", "EXACTLY" commands causing model stress

### 2. Token Inefficiency
- Redundant instructions across prompts
- Long negative constraint lists
- Over-explanation of simple concepts
- Estimated 70-80% token waste

### 3. Hallucination Triggers
- Impossible requirement combinations
- Contradictory formatting rules
- Over-specification leaving no room for natural variation

## Simplified Prompt Philosophy

### Core Principles
1. **Natural Language First**: Write like you're briefing a human writer
2. **Essential Constraints Only**: Keep only what's truly necessary
3. **XML Tag Format Control**: Use `<tag>content</tag>` for structure
4. **Progressive Enhancement**: Start simple, add refinement layers
5. **Graceful Degradation**: Work with any model capability

## New Prompt Structure

### 1. Outline Generation (Reduced from 104 to ~8 lines)
```
Create a {sizesections}-section outline about {keyword} for {articleaudience} readers.
Format: Use Roman numerals (I, II, III) for main sections and letters (A, B, C) for subsections.
Focus on practical, actionable content that could use tables or lists.
Return only the outline in <outline> tags.
```

### 2. Title Generation (Reduced from 26 to ~6 lines)
```
Write a compelling 50-60 character title about {keyword} in {articlelanguage}.
Tone: {voicetone} for {articleaudience} readers.
Include the exact keyword phrase naturally.
Return only the title in <title> tags.
```

### 3. Paragraph Generation (Reduced from 234 to ~12 lines)
```
Write paragraph {current_paragraph} of {paragraphs_per_section} for section "{heading}".
Cover: {current_points}
Audience: {articleaudience} | Tone: {voicetone} | Style: {articletype}
Use natural HTML formatting (<strong>, <em>, lists) for readability.
Return content in <paragraph> tags.
```

## XML Tag System Implementation

### Tag Structure
- `<title>` - Article titles
- `<outline>` - Structured outlines  
- `<paragraph>` - Individual paragraphs
- `<introduction>` - Article introductions
- `<conclusion>` - Article conclusions
- `<summary>` - Article summaries

### Regex Extraction Pattern
```python
import re

def extract_tag_content(response, tag_name):
    """Extract content from XML-style tags."""
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else response.strip()
```

## Implementation Plan

### Phase 1: Prompt Simplification
1. **Create new prompt file** with natural language versions
2. **Reduce token count** by 70-80%
3. **Remove negative constraints** and focus on positive guidance
4. **Add XML tag requirements** for format control

### Phase 2: Code Updates
1. **Update content_generator.py** to handle XML tag extraction
2. **Add fallback parsing** for models that ignore tags
3. **Implement progressive enhancement** based on model capability
4. **Add logging** for tag extraction success rates

### Phase 3: Testing & Validation
1. **A/B test** old vs new prompts across different models
2. **Measure token usage** reduction
3. **Evaluate output quality** improvements
4. **Test with various LLMs** (GPT-4, Claude, DeepSeek, etc.)

## Expected Benefits

### Quantitative Improvements
- **70-80% token reduction** in prompts
- **50-60% cost reduction** in API calls
- **30-40% faster** response times
- **Reduced hallucination rate**

### Qualitative Improvements
- **More natural output** from LLMs
- **Better model compatibility** across providers
- **Easier maintenance** and updates
- **Improved human readability** of prompts

## Risk Mitigation

### Potential Issues & Solutions
1. **Format inconsistency**: Use fallback regex patterns
2. **Model non-compliance**: Implement progressive enhancement
3. **Quality concerns**: A/B testing with rollback capability
4. **Edge cases**: Comprehensive test suite

## Next Steps

1. **Create simplified prompt file** (prompts_natural.py)
2. **Implement tag extraction** in content_generator.py
3. **Add configuration flag** for old vs new prompt system
4. **Run comparative tests** with sample articles
5. **Document migration guide** for existing users

Would you like me to proceed with implementing this plan? I can start by creating the simplified prompt file and then move to updating the content generator to handle the new tag-based system.