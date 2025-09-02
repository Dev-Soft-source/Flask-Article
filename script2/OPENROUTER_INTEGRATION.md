# OpenRouter Integration Guide for Script 2

This guide explains how to set up and use the OpenRouter integration in Script 2 of the AI Article Generator. It covers everything from initial setup to configuring OpenRouter parameters and using different language models.

## Table of Contents

1. [Introduction to OpenRouter](#introduction-to-openrouter)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Installing Dependencies](#installing-dependencies)
4. [OpenRouter Configuration](#openrouter-configuration)
5. [Configuration Parameters Explained](#configuration-parameters-explained)
6. [Using Different Models](#using-different-models)
7. [Command Line Options](#command-line-options)
8. [Troubleshooting](#troubleshooting)
9. [Frequently Asked Questions](#frequently-asked-questions)

## Introduction to OpenRouter

OpenRouter is a service that provides access to multiple AI language models through a single, unified API. Instead of signing up for multiple AI providers and managing different API formats, OpenRouter lets you use one API key to access models from Anthropic, Meta, Mistral, Google, and many other providers.

**Benefits of using OpenRouter:**

- Access to dozens of language models through a single API key
- Free tier with generous quotas for many powerful models
- Same API format as OpenAI, making it compatible with existing code
- Ability to switch between models without changing your code
- Can be more cost-effective than using OpenAI directly

## Setting Up Your Environment

Before using Script 2 with OpenRouter, you need to set up a Python virtual environment. This isolates the project's dependencies from your system Python installation.

### For Windows:

1. Open Command Prompt
2. Navigate to your project directory:
   ```
   cd path\to\your\script2\folder
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```

### For macOS and Linux:

1. Open Terminal
2. Navigate to your project directory:
   ```
   cd path/to/your/script2/folder
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

You'll know the virtual environment is activated when you see `(venv)` at the beginning of your command prompt.

## Installing Dependencies

After activating your virtual environment, install the required dependencies:

```
pip install -r requirements.txt
```

This command reads the `requirements.txt` file in the Script 2 folder and installs all the necessary packages, including the ones needed for OpenRouter integration.

## OpenRouter Configuration

### Step 1: Sign up for OpenRouter

1. Go to [OpenRouter.ai](https://openrouter.ai/) and sign up for an account
2. Generate an API key from your OpenRouter dashboard
3. Note down your API key - you'll need it for configuration

### Step 2: Configure the Script

Script 2 offers two ways to configure OpenRouter:

#### Option A: Edit the configuration in config.py

Open `config.py` in the Script 2 folder and modify the OpenRouter configuration parameters:

```python
# In the Config class definition
class Config:
    # ... other configuration parameters ...
    
    # OpenRouter Configuration
    openrouter_api_key: str = field(default_factory=lambda: os.getenv('OPENROUTER_API_KEY', ''))
    use_openrouter: bool = False
    openrouter_site_url: str = None
    openrouter_site_name: str = None
    openrouter_model: str = "anthropic/claude-3-opus-20240229"  # Default OpenRouter model
    openrouter_models: Dict[str, str] = None
```

#### Option B: Edit the main.py initialization

Open `main.py` and modify the Config initialization:

```python
config = Config(
    # Other settings...
    
    # OpenRouter Configuration
    openrouter_api_key="your_api_key_here",  # Replace with your actual API key
    use_openrouter=True,  # Set to True to enable OpenRouter
    openrouter_site_url="https://example.com",  # Your website URL for attribution
    openrouter_site_name="AI Article Generator",  # Your application name
    openrouter_model="anthropic/claude-3-opus-20240229",  # The model to use
)
```

#### Option C: Use environment variables

Create a `.env` file in the Script 2 directory with:

```
OPENROUTER_API_KEY=your_api_key_here
```

Then enable OpenRouter in your configuration.

#### Option D: Use command line arguments

Script 2 supports configuring OpenRouter via command line arguments:

```bash
python main.py \
  --input input.csv \
  --openrouter-key "your_api_key" \
  --use-openrouter \
  --openrouter-site-url "https://example.com" \
  --openrouter-site-name "My App" \
  --openrouter-model "anthropic/claude-3-opus-20240229"
```

Command line options take precedence over configuration in the code.

## Configuration Parameters Explained

Here's a detailed explanation of each OpenRouter configuration parameter:

| Parameter | Type | Description |
|-----------|------|-------------|
| `openrouter_api_key` | String | Your API key from OpenRouter. Required for authentication. |
| `use_openrouter` | Boolean | When set to `True`, the system will use OpenRouter instead of OpenAI's API. Set to `False` to use OpenAI directly. |
| `openrouter_site_url` | String | Your website URL, used for attribution by OpenRouter. Can be any valid URL. |
| `openrouter_site_name` | String | Your application name, used for attribution by OpenRouter. |
| `openrouter_model` | String | The specific model to use when OpenRouter is enabled. This parameter overrides the OpenAI model setting. Use the full model path (e.g., "anthropic/claude-3-opus-20240229") or a shortname that maps to a model in `openrouter_models`. |
| `openrouter_models` | Dictionary | A mapping of shortnames to full model IDs. This allows you to use convenient shortnames instead of the full model paths. |

## Using Different Models

OpenRouter gives you access to many different language models. Here are some popular options:

### Free Models (Pre-configured)

Script 2 comes pre-configured with these free models:

```python
openrouter_models = {
    "deepseek-coder": "deepseek/deepseek-coder",
    "deepseek-chat": "deepseek/deepseek-chat",
    "llama3-70b": "meta-llama/llama-3-70b-instruct",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
    "zephyr-chat": "huggingface/zephyr-7b-beta"
}
```

To use one of these, simply set:

```python
openrouter_model="llama3-70b"  # Will use meta-llama/llama-3-70b-instruct
```

### Premium Models

OpenRouter also provides access to premium models like Claude and GPT-4. These require credits:

- `anthropic/claude-3-opus-20240229` - Most powerful Claude model
- `anthropic/claude-3-sonnet-20240229` - Balanced Claude model
- `anthropic/claude-3-haiku-20240307` - Fastest Claude model
- `openai/gpt-4o` - OpenAI's latest GPT-4 model
- `openai/gpt-4-turbo` - OpenAI's GPT-4 Turbo model

To use a premium model:

```python
openrouter_model="anthropic/claude-3-opus-20240229"
```

## Command Line Options

Script 2 has full command-line support for OpenRouter configuration:

```bash
python main.py \
  --input input.csv \
  --openrouter-key "your_api_key" \
  --use-openrouter \
  --openrouter-site-url "https://example.com" \
  --openrouter-site-name "My App" \
  --openrouter-model "anthropic/claude-3-opus-20240229"
```

Command line options take precedence over configuration in the code.

## Script 2 Specific Integration Details

Script 2 integrates OpenRouter throughout all its article generation components:

1. **Content Generator**: The core content generation process uses the selected OpenRouter model when enabled
2. **Text Processor**: Humanization, grammar checks, and block note generation use OpenRouter
3. **Meta Handler**: SEO meta descriptions and WordPress excerpts use OpenRouter
4. **Generator**: All generation processes automatically use the correct API based on your configuration

The integration is seamless - the same functions that previously used OpenAI now detect whether to use OpenAI or OpenRouter based on the configuration.

## Troubleshooting

### Common Issues

1. **"Invalid API key" error**
   - Make sure you've copied your OpenRouter API key correctly
   - Verify that the API key is active in your OpenRouter dashboard

2. **"Model not available" error**
   - Check if the model is still available on OpenRouter
   - Some models may have usage limits or be temporarily unavailable

3. **Rate limit errors**
   - Different models have different rate limits
   - Try spacing out your requests or switching to a different model

4. **OpenRouter not working**
   - Ensure `use_openrouter` is set to `True`
   - Check your internet connection
   - Verify that OpenRouter's services are online

5. **Script 2 specific errors**
   - Look for error logs related to make_openrouter_api_call
   - Ensure that the ai_utils.py file is properly importing the necessary modules

## Frequently Asked Questions

**Q: Do I need to sign up for all these AI services like Anthropic and Meta?**
A: No! That's the beauty of OpenRouter - you only need an OpenRouter account, and they handle the connections to all the different providers.

**Q: Can I use OpenRouter for free with Script 2?**
A: Yes! OpenRouter provides free access to several powerful models. You get a monthly quota of free credits when you sign up, and some models like Llama 3 and Mistral have completely free tiers.

**Q: How do I know which model is best for my needs in Script 2?**
A: Start with a balanced model like "llama3-70b" or "mixtral-8x7b" for general use. If you need higher quality, try Claude models. For code generation, "deepseek-coder" is a good option.

**Q: Is there any difference in article quality between OpenAI and OpenRouter models?**
A: Quality varies by model. Claude models from Anthropic are comparable to GPT-4, while Llama 3 and Mixtral are comparable to GPT-3.5. You can experiment with different models to find the best fit for your content.

**Q: Will switching to OpenRouter affect my token usage or tracking in Script 2?**
A: No, the integration maintains all token tracking functionality. The system properly counts tokens and manages context windows regardless of which API is used.

**Q: Can I use environment variables for the API key?**
A: Yes! Script 2 is designed to use the `OPENROUTER_API_KEY` environment variable if present, making it easy to keep your API key out of the code.

**Q: How do I check which API is being used?**
A: The script logs which API provider is being used when generating content. Look for log messages like "Using OpenRouter with model: [model name]" or check the console output.

---

For more information and updates, visit [OpenRouter's official documentation](https://openrouter.ai/docs). 