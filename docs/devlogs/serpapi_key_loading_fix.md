# SerpAPI Key Loading Fix

## Issue
The article generator script in script1 was failing with an error message:
```
ValueError: SerpAPI key is required when add_PAA_paragraphs_into_article is True
```

The script was unable to load the SerpAPI key from the environment variables because there was a naming inconsistency between:
- How the keys were defined in the `.env` file (`SERP_API_KEY_1`, `SERP_API_KEY_2`)
- How the script was looking for them (`SERPER_API_KEY`, `SERPER_API_KEY_1`, etc.)

## Solution
1. Modified `script1/article_generator/serpapi.py` to check for both naming formats:
   - Now checks for both `SERPER_API_KEY` and `SERP_API_KEY` environment variables
   - Also checks for both `SERPER_API_KEY_n` and `SERP_API_KEY_n` formats for additional keys

2. Updated `script1/utils/config.py` to be more flexible in loading the SerpAPI key:
   - Changed the default factory to check both `SERPER_API_KEY` and `SERP_API_KEY` environment variables

## Testing
- The script should now properly load the SerpAPI keys from the `.env` file regardless of which naming format is used.
- This fix ensures compatibility with both naming conventions that appear in the codebase.

## Impact
This fix allows the PAA (People Also Ask) feature to work correctly when `add_PAA_paragraphs_into_article` is set to `True`.

Date: June 10, 2025
