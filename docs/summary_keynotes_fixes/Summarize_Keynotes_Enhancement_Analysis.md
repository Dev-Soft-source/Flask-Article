# Summarize and Keynotes Enhancement Analysis

## Current Implementation

### Overview
The article generation system currently has two separate features for summarizing content:

1. **Summarize Section**: A comprehensive summary of the article (200-300 words) that captures the main points and insights from each major section.
2. **Keynotes Section (Block Notes)**: A concise paragraph (5-6 sentences) highlighting the key takeaways from the article.

### Current Implementation Details

#### Summarize Section
- Implemented in both scripts as `generate_article_summary` (script1) and directly in the generator class (script2)
- Uses the same LLM model as the rest of the article generation
- Prompt focuses on creating a thorough overview of the main points (200-300 words)
- In script1, it's implemented in `content_generator.py`
- In script2, it's implemented directly in `generator.py`

#### Keynotes Section (Block Notes)
- Implemented in both scripts as `generate_block_notes` in the `text_processor.py` file
- Uses the same LLM model as the rest of the article generation
- Prompt focuses on creating a single cohesive paragraph with 5-6 key takeaways
- Limited to about 150 words

### OpenRouter Integration
- Both scripts have OpenRouter integration implemented
- Configuration allows specifying different models for different tasks
- In script1, there's a specific `rag_openrouter_model` parameter that could be leveraged
- In script2, there's a similar parameter structure

### Chunking Implementation
- Both scripts have chunking implementations in the RAG (Retrieval-Augmented Generation) components
- These implementations could be adapted for handling large articles when generating summaries and keynotes

## Requirements

The client wants to enhance the Summarize and Keynotes sections with the following improvements:

1. **Use a Large Context Window LLM**: Select a specific OpenRouter model with a large context window to process the entire article
2. **Maintain Different Prompts**: Keep the existing different prompts for Summarize and Keynotes sections
3. **Implement Chunking Strategy**: For very large articles that exceed the context window, implement a chunking strategy to process the article in parts

## Implementation Plan

### 1. Model Selection Configuration

Add configuration options to allow selecting a specific model for summarization and keynotes generation:

- Add `summary_keynotes_model` parameter to the Config class
- Default to a model with a large context window (e.g., Claude 3 Opus or GPT-4)
- Allow this to be configured separately from the main article generation model

### 2. Enhanced Summary Generation

Modify the summary generation function to:
- Use the specified large context window model
- Process the entire article at once
- Fall back to chunking for extremely large articles

### 3. Enhanced Keynotes Generation

Modify the keynotes generation function to:
- Use the specified large context window model
- Process the entire article at once
- Fall back to chunking for extremely large articles

### 4. Chunking Strategy Implementation

Implement a chunking strategy that:
- Splits the article into meaningful chunks (by sections)
- Processes each chunk separately
- Combines the results into a coherent summary/keynotes

## Technical Implementation Details

### Configuration Changes

Add the following parameters to the Config class:
- `summary_keynotes_model`: The OpenRouter model to use for summary and keynotes generation
- `enable_separate_summary_model`: Boolean flag to enable/disable using a separate model
- `summary_max_tokens`: Maximum tokens for the summary generation
- `keynotes_max_tokens`: Maximum tokens for the keynotes generation
- `summary_chunk_size`: Chunk size for summary generation when chunking is needed
- `keynotes_chunk_size`: Chunk size for keynotes generation when chunking is needed

### Function Modifications

1. Modify `generate_article_summary` to:
   - Check if a separate model is configured
   - Use the configured model if enabled
   - Implement chunking if the article is too large

2. Modify `generate_block_notes` to:
   - Check if a separate model is configured
   - Use the configured model if enabled
   - Implement chunking if the article is too large

### Chunking Implementation

Create a new utility function `chunk_article_for_processing` that:
- Takes an article dictionary as input
- Splits it into meaningful chunks based on sections
- Returns a list of chunks that can be processed separately

## Implementation Priority

1. Configuration changes
2. Model selection implementation
3. Summary generation enhancement
4. Keynotes generation enhancement
5. Chunking strategy implementation
