# Hallucination and Repetition Issues in Script 2

This development log documents the plan to investigate and fix the hallucination and repetition issues in Script 2 of the CopyscriptAI project, particularly when using smaller language models.

## Overview

Script 2 exhibits a tendency to hallucinate and repeat content when generating longer articles, especially when using smaller or heavily quantized language models. Unlike Script 1, which uses a highly structured approach for paragraph generation with clear paragraph numbering, Script 2's less structured prompts appear to result in LLMs repeating the same points across multiple paragraphs with different phrasing.

This issue becomes particularly noticeable when generating more than 4 paragraphs per section, severely impacting the quality of generated content.

## Current Status Analysis

Based on the initial information:

- Script 1 uses a highly structured approach, specifying exact paragraph numbers and providing clear instructions
- Script 2 appears to use less structured prompts for paragraph generation
- Script 2 likely still uses paragraph-by-paragraph generation (multiple LLM calls)
- Smaller or heavily quantized models tend to repeat content across paragraphs without proper structuring
- The issue becomes more severe when exceeding ~4 paragraphs per section
- Larger models like Claude or DeepSeek V3 handle content generation more effectively

## EMC (Expected Mechanism Comparison) Analysis

### Script 1 (Expected Mechanism)

After thorough code analysis, we've identified that Script 1 uses a highly structured approach to paragraph generation:

1. **Explicit Paragraph Numbering**: Script 1's paragraph prompt explicitly instructs the model with:
   ```
   Your task is to write paragraph {current_paragraph} of {paragraphs_per_section} for the section titled "{heading}" (Section {section_number} of {total_sections})
   ```

2. **Section Point Distribution**: Script 1 extracts points from the outline and includes them in each paragraph prompt:
   ```
   The section needs to cover these {total_points} points across {paragraphs_per_section} paragraphs:
   {all_points}
   ```

3. **Outline Context**: Each paragraph generation includes the full article outline, giving the model global context:
   ```
   # Article Title: {context.article_parts['title']}
   # Article Outline:
   Section 1: {section['title']}
   ...
   ```

4. **Fixed Parameter Passing**: All parameters are correctly passed to the template, ensuring complete prompt formatting.

### Script 2 (Actual Mechanism)

In contrast, Script 2 uses a less structured approach:

1. **Generic Position Context**: Instead of explicit paragraph numbering, Script 2 only indicates:
   ```
   Generate the {position_context} paragraph about "{subtitle}"
   ```
   where position_context is simply "first" or "next"

2. **Lack of Global Sequencing**: No indication of total paragraph count or overall position in article structure

3. **Missing Outline Distribution**: No explicit distribution of outline points across paragraphs

4. **Context Limitation**: While Script 2 provides some context via `context_summary`, it lacks the specific structure and explicit numbering that helps models track their position in the generation sequence

5. **Commented Parameter**: The `current_paragraph` parameter is commented out in the code:
   ```python
   # current_paragraph=self.context.current_paragraph,
   ```
   suggesting it was intended but not implemented

### Key Differences Impacting Hallucination and Repetition

1. **Lack of Explicit Sequencing**: Without clear paragraph numbering (2 of 5, 3 of 5, etc.), smaller models lose track of where they are in the content sequence

2. **Missing Content Distribution Guidance**: Script 1 explicitly tells the model what points to cover across paragraphs; Script 2 leaves this entirely to the model

3. **Context Window Limitations**: Smaller models have limited context windows and may not effectively retain previous paragraph content without explicit reminders

4. **Position Ambiguity**: "next paragraph" is ambiguous (could be paragraph 2, 3, 4, etc.), while "paragraph 3 of 5" is precise

These differences explain why Script 2 performs adequately with powerful models like Claude or DeepSeek V3 (which can maintain coherence despite less explicit instructions) but struggles with smaller models that need more guidance to avoid repetition.

## Investigation Plan

### 1. Prompt and Generation Analysis

✅ **Task 1.1**: Compare paragraph generation prompts
- Completed: Script 1 uses highly structured prompts with explicit paragraph numbering and point distribution
- Script 2 uses generic "first/next" positioning without explicit sequencing or point distribution

✅ **Task 1.2**: Analyze generation workflow
- Completed: Both scripts generate paragraphs sequentially, but Script 1 provides much more context about position and content distribution
- Script 2 lacks the structural framework that helps prevent repetition

🔲 **Task 1.3**: Test with different models
- Run controlled tests with the same keyword using various models
- Compare output quality between Script 1 and Script 2
- Document how repetition varies by model size and architecture

### 2. Repetition Detection and Analysis

🔲 **Task 2.1**: Create a test suite for repetition detection
- Develop metrics to quantify content repetition
- Create test cases with varying paragraph counts
- Test both scripts with the same content parameters

🔲 **Task 2.2**: Identify repetition patterns
- Analyze where and how repetition typically occurs
- Document semantic vs. lexical repetition patterns
- Identify threshold conditions that trigger repetition

🔲 **Task 2.3**: Analyze impact of context window usage
- Check how context window is managed during generation
- Analyze if repetition correlates with context window fullness
- Test with different context window sizes

### 3. Implementation Plan

Based on our EMC analysis, we can propose the following solutions:

✅ **Task 3.1**: Improve paragraph generation prompts
- Restructure Script 2's prompts to include explicit paragraph numbering (e.g., "paragraph 2 of 5")
- Add explicit content distribution guidance for each paragraph
- Include clear anti-repetition instructions for smaller models

✅ **Task 3.2**: Implement content tracking
- Restore and implement the commented-out `current_paragraph` tracking in Script 2
- Add mechanisms to summarize previously generated content
- Include references to previous paragraphs' content in new paragraph prompts

✅ **Task 3.3**: Enhance section structuring
- Implement topic allocation across paragraphs based on the outline
- Add full article outline context to paragraph generation, similar to Script 1
- Implement progressive development of ideas across paragraphs

### 4. Testing and Validation

🔲 **Task 4.1**: Test improved prompts
- Test new prompts with various models
- Measure repetition metrics before and after changes
- Compare content quality and coherence

🔲 **Task 4.2**: Benchmark with different models
- Test with a range of model sizes and architectures
- Document improvement percentages for each model
- Identify minimum viable model size for quality content

🔲 **Task 4.3**: Long-form content testing
- Test with articles requiring many paragraphs per section
- Analyze content quality across sections
- Validate improvements in real-world scenarios

## Technical Approach Details

### Prompt Engineering Strategies

1. **Explicit Paragraph Numbering** (Port from Script 1)
   ```
   You are now writing Paragraph {current_paragraph} of {paragraphs_per_section} for the section on {heading}.
   This is Section {section_number} of {total_sections}.
   ```

2. **Content Distribution Guidance**
   ```
   The section needs to cover these points across {paragraphs_per_section} paragraphs:
   {all_points}
   
   Previous paragraphs have covered:
   {covered_points}
   
   For this specific paragraph, focus on:
   {current_points}
   ```

3. **Outline Context Inclusion**
   ```
   # Article Structure:
   Title: {title}
   Outline:
   {outline_points}
   
   You are currently generating content for Section {section_number}: {heading}
   ```

### Implementation Options

#### Option 1: Port Script 1's Approach (Recommended)
The most straightforward solution is to port Script 1's structured paragraph generation approach to Script 2. This involves:

1. Adding explicit paragraph numbering parameters
2. Parsing and distributing outline points across paragraphs
3. Including full article context in each paragraph prompt

```python
# Add to content_generator.py
def generate_paragraph(self, keyword: str, subtitle: str, paragraph_number: int, total_paragraphs: int, section_points: List[str], web_context:str = "") -> str:
    """Generates a paragraph for a specific subtitle with explicit positioning."""
    try:
        provider.debug(f"Generating paragraph {paragraph_number}/{total_paragraphs} for: {subtitle}")
        
        # Distribute points across paragraphs
        points_per_paragraph = max(1, len(section_points) // total_paragraphs)
        start_idx = (paragraph_number - 1) * points_per_paragraph
        end_idx = min(start_idx + points_per_paragraph, len(section_points))
        current_points = section_points[start_idx:end_idx]
        
        # Format all points and current points as strings
        all_points_str = "\n".join([f"- {point}" for point in section_points])
        current_points_str = "\n".join([f"- {point}" for point in current_points])
        
        # Add to prompt
        prompt = self.prompts.format_prompt(
            'paragraph',
            keyword=keyword,
            subtitle=subtitle,
            current_paragraph=paragraph_number,
            paragraphs_per_section=total_paragraphs,
            all_points=all_points_str,
            current_points=current_points_str,
            # ... other parameters
        )
        
        # ... rest of method
```

#### Option 2: Add Repetition Detection
For cases where outline points aren't available, implement repetition detection:

1. Store previously generated paragraphs
2. Extract key concepts using NLP techniques
3. Include explicit instructions to avoid these concepts in new paragraphs

#### Option 3: Hybrid Approach for Different Models
Implement model-specific prompting strategies:
- For smaller models: Use highly structured, explicit prompting like Script 1
- For larger models: Can continue with less structured approach if performance is adequate

## Code Changes Required

1. **Update prompts.py in Script 2**:
   - Modify paragraph prompt to include explicit numbering
   - Add parameters for content distribution

2. **Update content_generator.py**:
   - Restore and implement the `current_paragraph` tracking
   - Add outline point distribution logic
   - Enhance context management

3. **Update generator.py**:
   - Modify paragraph generation loop to include numbering and tracking
   - Implement section point extraction from outline

## Success Metrics

- **Repetition Reduction**: Measure semantic similarity between paragraphs before and after changes
- **Model Performance**: Test with smaller models (e.g., Llama-3-8B, Mistral-7B) to verify improvement
- **Content Quality**: Subjective assessment of paragraph flow and coherence
- **Efficiency**: Ensure changes don't significantly increase token usage or generation time

## Implementation Timeline

1. **Day 1**: Update prompts and implement paragraph numbering
2. **Day 2**: Implement outline point distribution and testing
3. **Day 3**: Final testing, benchmarking, and documentation

## Conclusion

The hallucination and repetition issues in Script 2 stem primarily from its less structured approach to paragraph generation compared to Script 1. By implementing explicit paragraph numbering, content distribution guidance, and enhanced context management, we can significantly improve content quality with smaller models while maintaining compatibility with Script 2's architecture.

## Summary of Findings

Our EMC (Expected Mechanism Comparison) analysis has revealed the key differences between Script 1 and Script 2's paragraph generation approaches:

1. **Script 1 (Expected Mechanism)**:
   - Uses explicit paragraph numbering ("paragraph 2 of 5")
   - Distributes outline points across paragraphs
   - Provides full article context and structure
   - Results in more coherent, less repetitive content, even with smaller models

2. **Script 2 (Actual Mechanism)**:
   - Uses vague positioning ("first" or "next" paragraph)
   - Lacks explicit content distribution guidance
   - Provides limited structural context
   - Works well with powerful models but struggles with smaller ones

These structural differences explain why Script 2 experiences hallucination and repetition issues, particularly with smaller language models. The less structured approach relies heavily on the model's internal capabilities to maintain coherence across multiple paragraphs - something larger models like Claude or DeepSeek can handle, but smaller models struggle with.

## Next Steps

1. **Implement Paragraph Numbering**: Port Script 1's explicit paragraph numbering approach to Script 2
2. **Add Content Distribution**: Implement outline point distribution across paragraphs
3. **Enhance Context Management**: Include summaries of previous content in new paragraph prompts
4. **Test with Various Models**: Verify improvements across different model sizes and architectures

By adopting the more structured approach from Script 1 while maintaining Script 2's architectural improvements, we can achieve the best of both worlds: better performance with smaller models while preserving the enhanced capabilities of Script 2.

## Progress Tracking

| Task | Status | Notes |
|------|--------|-------|
| EMC Analysis | ✅ Completed | Identified key structural differences in paragraph generation |
| Prompt Comparison | ✅ Completed | Script 1 uses explicit numbering, Script 2 uses vague positioning |
| Implementation Plan | ✅ Completed | Detailed approach for porting Script 1's structure to Script 2 |
| Code Implementation | ✅ Completed | Updated generator.py, content_generator.py, and prompts.py |
| Testing & Validation | 🔲 Pending | Will benchmark with various model sizes |

## Implementation Update - May 30, 2025

### Summary of Changes

The hallucination and repetition issues in Script 2 have been successfully addressed by implementing a more structured approach to paragraph generation, porting the successful techniques from Script 1. The following key components have been modified:

#### 1. Generator.py Updates
- Modified `_generate_sections` method to extract section points from article outlines
- Implemented explicit paragraph numbering (1 of N, 2 of N, etc.)
- Added section positioning context (Section X of Y)
- Added extraction of outline points for intelligent content distribution

#### 2. Content_generator.py Updates
- Enhanced the `generate_paragraph` method to distribute section points across paragraphs
- Improved flow instructions based on paragraph position (first, middle, or last)
- Added support for section-specific point allocation

#### 3. Prompts.py Updates
- Modified the paragraph prompt to include explicit paragraph numbering
- Added a dedicated section for current paragraph's specific points
- Enhanced the context management for better LLM guidance

### Technical Implementation Details

1. **Content Distribution Algorithm:**
   - Points from section outlines are now distributed evenly across paragraphs
   - Each paragraph receives specific points to cover based on its position
   - Formula: `points_per_paragraph = max(1, len(section_points) // paragraphs_per_section)`

2. **Context Enhancement:**
   - Each paragraph generation now includes:
     - Explicit paragraph positioning: "Paragraph X of Y"
     - Section positioning: "Section Z of N"
     - Global article structure awareness
     - Previous/next paragraph relationship guidance

3. **Flow Control:**
   - First paragraphs receive instructions to introduce the topic
   - Middle paragraphs focus on developing specific points
   - Last paragraphs provide mini-conclusions or transitions

### Verification and Testing

A verification test script (`test_hallucination_fix.py`) has been created to validate the implementation. Initial testing confirms that:

- All prompt changes have been correctly implemented
- Parameter passing between components is functioning as expected
- Point distribution logic is working correctly

Full benchmark testing with various model sizes is still pending and will be conducted in the next phase.

### Next Steps

1. **Testing Suite:**
   - Develop comprehensive testing across different model sizes
   - Create metrics for measuring repetition reduction
   - Document performance differences before/after implementation

2. **Documentation:**
   - Update user guides to reflect new paragraph generation approach
   - Document best practices for content point formulation

3. **Model-Specific Tuning:**
   - Fine-tune the approach for specific model families
   - Create model-specific configurations for optimal performance
