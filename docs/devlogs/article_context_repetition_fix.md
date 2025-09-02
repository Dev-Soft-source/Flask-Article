# Article Context Repetition Issue Fix - Development Log

## Date: June 10, 2025

## Issue Summary
Users have reported repetition problems in generated articles where content appears to be generated in isolation without considering the entire context. Investigation suggests that the ArticleContext class is not being properly utilized or passed during API calls to language models, particularly when using OpenRouter.

## Root Cause Analysis
After comparing both script implementations, the critical issue was identified:

1. In Script One (`script1`):
   - When using OpenRouter, the code is removing all previous user messages except the last one:
   ```python
   # remove all the user messages, except the last one, but keep all the system, and model messages
   messages = [msg for msg in messages if msg["role"] != "user" or msg == messages[-1]]
   ```
   - This means the language model doesn't have access to the full conversation history during content generation.
   - The ArticleContext object is not being fully utilized when making API calls.
   - RAG context is being added to individual prompts rather than integrated into the system message.

2. In Script Two (`script2`):
   - The ArticleContext object is properly passed to the `generate_completion` function.
   - Context is maintained through all API calls with proper message management.
   - The `set_rag_context` method integrates RAG context into the system message.
   - It has a more sophisticated pruning strategy for context management.

## Proposed Fix
1. Modify Script One's API call approach to:
   - Stop removing previous user messages when using OpenRouter
   - Properly utilize the ArticleContext during all API calls
   - Implement similar context management techniques as found in Script Two
   - Ensure RAG context is consistently available across article generation

## Task List
- [x] Create development log to track progress
- [x] Modify the OpenRouter API call in Script One to preserve message history
- [x] Ensure the ArticleContext is properly utilized in all API calls
- [x] Implement proper context pruning if token limits are reached
- [ ] Test article generation with the updated context handling
- [ ] Document the changes and update any relevant documentation
- [x] Identify and fix similar issues in Script Two
  - [x] Modify the `generate_completion` function to use ArticleContext's message history
  - [ ] Test the fix with article generation
  - [ ] Update documentation for Script Two

## Implementation Notes
The fix will focus on the `make_openrouter_api_call` function and surrounding code in Script One's content generator. We'll preserve the message history while ensuring that token limits are respected through smart context pruning when necessary.

## Script Two Context Repetition Issue
While implementing fixes for Script One, we identified that Script Two also has a similar issue, but in a different location. In Script Two, the ArticleContext class is properly implemented with message management and pruning, but the `generate_completion` function in `utils/ai_utils.py` doesn't use it correctly:

```python
# Current implementation in Script Two's generate_completion function:
# Prepare messages
messages = []

# Add system message if available
if config and hasattr(config, 'prompts'):
    system_message = config.prompts.system_message
    if system_message:
        messages.append({"role": "system", "content": system_message})

# Add user prompt
messages.append({"role": "user", "content": prompt})
```

Instead of using the full message history from the ArticleContext, it creates a new array with just the system message and current user prompt. This means each API call is made in isolation without the benefit of previous conversation context.

### Proposed Fix for Script Two
Modify the `generate_completion` function in `utils/ai_utils.py` to:
1. Use the full message history from ArticleContext when available
2. Only create a new message array when ArticleContext is not provided
3. Add the current prompt to the message history through ArticleContext before making API calls

This will ensure that the full conversation history is maintained between API calls, providing proper context for the language model and preventing repetition issues.

## Test Plan
- Generate articles with the same keyword using both the original and fixed versions
- Compare the output for repetition issues
- Validate that context is properly maintained throughout article generation
- Check token usage to ensure we're not exceeding limits

## Progress Updates
- 2025-06-10: Initial investigation complete, development log created
- 2025-06-10: Implemented fixes for Script One:
  - Modified the OpenRouter API call to preserve all user messages (conversation history)
  - Added smart token-aware context pruning that only removes messages when necessary
  - Added a `set_rag_context` method to properly integrate RAG context into the system message
  - Updated the Generator class to use the new `set_rag_context` method
- 2025-06-10: Identified a similar issue in Script Two:
  - Despite having a well-implemented ArticleContext class, the `generate_completion` function in `utils/ai_utils.py` creates new message arrays for each API call instead of using the full message history from ArticleContext
  - This causes the full conversation context to be lost between API calls, leading to repetition issues
- 2025-06-10: Implemented fix for Script Two:
  - Modified the `generate_completion` function in `utils/ai_utils.py` to use ArticleContext's message history instead of creating a new message array each time
  - Added proper logging for message history usage
  - Ensured the user prompt is properly added to ArticleContext before making API calls
  - The fix maintains backward compatibility for cases where ArticleContext is not provided
  - Created a test script (`tests/test_context_preservation.py`) to verify context preservation across multiple API calls
