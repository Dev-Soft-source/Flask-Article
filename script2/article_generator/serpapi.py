import itertools
import requests
import os
from dotenv import load_dotenv
from utils.error_utils import ErrorHandler, format_error_message
from article_generator.logger import logger

# Global error handler
error_handler = ErrorHandler(show_traceback=True)

# Load environment variables
load_dotenv()

# Get SERP API keys from environment variables
serpApiList = []

# Try to get the main SERP_API_KEY
main_key = os.getenv('SERP_API_KEY')
if main_key and main_key.strip():
    serpApiList.append(main_key.strip())

# Try to get additional keys if defined (SERP_API_KEY_1, SERP_API_KEY_2, etc.)
for i in range(1, 10):  # Check for up to 9 additional keys
    key = os.getenv(f'SERP_API_KEY_{i}')
    if key and key.strip():
        serpApiList.append(key.strip())

# Don't use hardcoded keys, just warn if no environment variables are set
if not serpApiList:
    error_handler.handle_error(
        Exception("No SERP_API_KEY found in environment variables. Please add your API key to the .env file."),
        context={"component": "SerpAPI", "configuration": "API Keys"},
        severity="warning"
    )

# from serpapi import GoogleSearch

infinite_cycle = itertools.cycle(serpApiList)


def validate_all_serpapi_keys():
    """
    Checks all SerpAPI keys and returns sorted list with available quotas
    Returns: (best_key, validated_keys)
    """
    validated_keys = []

    for api_key in serpApiList:
        key_data = {
            'full_key': api_key,
            'truncated': api_key[-6:],
            'valid': False,
            'type': 'unknown',
            'remaining': 0,
            'total_quota': 0,
            'status': 'unchecked'
        }

        try:
            response = requests.get(
                f"https://serpapi.com/account?api_key={api_key}",
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                key_data['type'] = data.get('plan_id', 'unknown')

                # Free plan check
                if 'total_searches_left' in data:
                    key_data['remaining'] = data['total_searches_left']
                    key_data['total_quota'] = data.get('searches_per_month', 0)
                    key_data['type'] = 'free'
                # Paid plan check
                else:
                    key_data['remaining'] = data.get('searches_remaining', 0)
                    key_data['total_quota'] = data.get('total_monthly_searches', 0)
                    key_data['type'] = 'paid'

                if key_data['remaining'] > 0:
                    key_data['valid'] = True
                    key_data['status'] = 'active'
                else:
                    key_data['status'] = 'exhausted'

            else:
                key_data['status'] = f'invalid (HTTP {response.status_code})'

        except Exception as e:
            key_data['status'] = f'error: {str(e)}'

        validated_keys.append(key_data)

    # Sort keys: free plans first, then by remaining quota descending
    sorted_keys = sorted(
        [k for k in validated_keys if k['valid']],
        key=lambda x: (
            -1 if x['type'] == 'free' else 1,  # Free plans first
            -x['remaining']  # Highest quota first
        )
    )

    # Generate report in logs
    logger.info("SERPAPI KEY VALIDATION REPORT")
    logger.debug(f"{'Key (last 6)':<8} | {'Type':<6} | {'Status':<18} | {'Remaining':<9} | {'Total Quota':<10}")
    
    for key in validated_keys:
        if key['valid']:
            status = "✅ Active"
        elif key['status'] == 'exhausted':
            status = "⚠️ Exhausted"
        else:
            status = f"❌ {key['status']}"
            
        logger.debug(f"{key['truncated']:<8} | {key['type'].capitalize():<6} | "
              f"{status:<18} | {key['remaining']:<9} | {key['total_quota']:<10}")

    # Select best available key
    best_key = sorted_keys[0]['full_key'] if sorted_keys else None

    if best_key:
        logger.success(f"Selected SerpAPI key: {best_key[-6:]}... (Type: {sorted_keys[0]['type'].capitalize()}, "
              f"Remaining: {sorted_keys[0]['remaining']}/{sorted_keys[0]['total_quota']})")
    else:
        error_handler.handle_error(
            Exception("No valid SerpAPI keys available for use"),
            context={"component": "SerpAPI", "validation": "API Keys", "keys_count": len(serpApiList)},
            severity="warning"
        )

    return best_key, validated_keys

# Example usage
if __name__ == "__main__":
    search_query = "latest AI developments 2024"
    logger.info(f"Performing search: '{search_query}'")
    
    best_key, validated_keys = validate_all_serpapi_keys()


