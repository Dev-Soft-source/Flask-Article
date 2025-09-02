# Automated Article Generation & Publishing with Python and OpenAI

## Overview
This Python script automates the process of generating and publishing articles using OpenAI's GPT language model. It is designed to run in Google Colab, with dependencies installed in Google Drive.

The script reads a structured CSV file where each row represents an article to be generated. It then uses the OpenAI API to generate the article content, including the title, introduction, body paragraphs, and conclusion. The generated content is enhanced with additional elements like a summary, FAQ section, images, YouTube videos, and external links.

Finally, the generated article is published to a WordPress site using the WordPress REST API.

## Key Components
Here's a high-level overview of the main components and flow of the script:

```mermaid
graph LR
    A[Read Input CSV] --> B[Generate Article Content]
    B --> C[Enhance Article]
    C --> D[Publish to WordPress]
```

### 1. Setup and Configuration
The script starts by setting up the environment and loading configuration settings:

- Google Drive is mounted to access input files and store dependencies
- Required Python libraries are installed in a Google Drive directory
- API keys and other settings are loaded from configuration dictionaries

Key configuration options include:
- OpenAI settings: API key, engine, token limit
- Article generation settings: tone, structure, language, perspective
- Enhancement settings: enable/disable summary, FAQ, images, videos, links
- WordPress settings: site URL, authentication

### 2. Input Data Processing
The input CSV file is read using Python's built-in csv module. Each row of the CSV represents an article to be generated, with columns for the main keyword, featured image, and subtitles.

The CSV data is validated and processed into a `dicts` dictionary, where each key represents a row index and the value is a dictionary of the row data.

### 3. Article Generation with OpenAI
The core article generation is handled by the `gpt3Request` function, which makes a request to OpenAI's GPT-3 API.

Several helper functions are used to generate specific parts of the article:
- `generateTitle`: Generates an engaging, SEO-friendly title
- `generateIntroduction`: Generates an introduction paragraph with a hook and summary
- `generateParagraph`: Generates a body paragraph for a given subtitle and keyword
- `generateConclusion`: Generates a conclusion paragraph that summarizes the article

Each of these functions constructs a detailed prompt with instructions and guidelines for the AI model, including the desired tone, structure, language, and perspective.

The generated text is then post-processed with sentence splitting, paragraph distribution, and cleaning functions to format it for the final article.

### 4. Content Enhancement
The generated article is enhanced with several optional elements based on the configuration settings:

- `summary_text`: Generates a summary paragraph
- `youtubevid`: Fetches a relevant YouTube video to embed
- `paa_fun`: Generates an FAQ section using "People Also Ask" questions from the SerpAPI
- `serp`: Fetches relevant external links to include using the SerpAPI
- `blockquoteFunction`: Generates a "keynote" section with key takeaways

Images are fetched from Unsplash using the pyunsplash library and added to the article based on subtitle keywords.

### 5. Publishing to WordPress
The final article data, including the title, content, category, tags, and featured image, is constructed into a post dictionary.

This post is then submitted to the WordPress REST API `/posts` endpoint using an HTTP POST request. Authentication is handled using Basic Auth with an application password.

The response from the API is checked to determine if the post was successfully created, and the URL of the new post is printed.

## Error Handling and Retry Logic
The script includes extensive error handling and retry logic to gracefully handle issues like rate limiting or empty API responses.

Key examples include:
- `gpt3Request`: Implements exponential backoff retry logic if the OpenAI API returns a rate limit error
- `paa_fun`: Wraps the `get_paa_questions` and `paaBreak` calls in try/except blocks to handle errors fetching or generating the FAQ section
- SerpAPI functions: Rotate through multiple API keys to avoid hitting rate limits

Errors are logged to the console for debugging purposes.

## Conclusion
Putting it all together, this script provides an end-to-end solution for automating article generation and publishing using AI.

The modular design and separation of concerns make it easy to understand and maintain. The extensive configuration options and error handling make it flexible and robust.

While there are many moving pieces, the core flow is straightforward:
1. Read structured article data from a CSV file
2. For each article:
   - Generate the title, introduction, body, and conclusion using OpenAI
   - Enhance the content with a summary, FAQ, images, videos, and links
   - Publish the article to WordPress
3. Handle errors and retry as needed

With this script, you can quickly generate and publish high-quality, SEO-friendly articles at scale, saving time and effort compared to manual writing and publishing.
