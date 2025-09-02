"""Prompt templates for article generation."""

# Core article structure prompts
TITLE_PROMPT = """Please ignore all previous instructions and prompts. I want you to respond in {articlelanguage} only.
I want you to act as a blog post title writer who speaks and writes fluent {articlelanguage}.
I type a title, or keywords separated by a comma and you respond with blog post titles in {articlelanguage}.
They should all have a hook and high potential to go viral on social media. Write everything in
{articlelanguage} without change order of keywords. Write only "1" title: {keyword}"""

OUTLINE_PROMPT = """Based on the following context:
{context_summary}

Please ignore all previous instructions and prompts. Role: You are an expert article writer specializing in creating highly engaging content that resonates deeply with specific audiences.
Context:
The subject of the article is: {keyword}
The areas to be discussed in the article should include: {keyword}
The target audience for the article is: {articleaudience}
My Main SEO Keyword is: {keyword}

Objective:
Generate only the structure and headings of the article. Create up to "{sizeheadings}" headings for output.

Guidelines:
- Human-First Approach
- In-Depth Exploration
- Tailored for the Audience
- Substantiated with Evidence
- Compelling Storytelling
- Acknowledge Complexity
- Forward-Thinking"""

INTRODUCTION_PROMPT = """Based on the following context:
{context_summary}

Generate an engaging introduction for an article about {keyword} titled "{title}".
Language: {articlelanguage}
Audience: {articleaudience}
Voice Tone: {voicetone}
Point of View: {pointofview}
Style: {articletype}

Guidelines:
- Start with a compelling hook
- Establish context and relevance
- Preview main points
- Natural integration of keyword
- Length: {sizesections} paragraphs"""

PARAGRAPH_PROMPT = """Based on the following context:
{context_summary}

Generate the {position_context} paragraph about "{subtitle}" in the context of {keyword}.

Requirements:
1. Write exactly ONE well-structured paragraph
2. Focus on a single main point or aspect
3. Use {articlelanguage} language
4. Target the {articleaudience} audience
5. Maintain a {voicetone} tone
6. Use {pointofview} point of view
7. Include specific details and examples
8. Natural keyword integration
9. Clear topic sentence and conclusion
10. Smooth transitions between ideas

Additional Guidelines:
{flow_instruction}
- Keep the paragraph focused and concise
- Use evidence-based content
- Maintain consistency with the article's style
- Ensure readability and engagement"""

CONCLUSION_PROMPT = """Based on the following context:
{context_summary}

Craft a compelling conclusion for this article titled "{title}" about {keyword}.

Guidelines:
- Summarize key points from all sections
- Reinforce main message
- Call to action or final thoughts
- Natural keyword integration
- Length: 2-3 paragraphs
- Maintain consistency with the article's tone and style"""

# Enhancement prompts
FAQ_PROMPT = """Based on the following context:
{context_summary}

Generate {num_questions} frequently asked questions and answers about {keyword}.

Requirements:
1. Format the response EXACTLY as a Python list of dictionaries
2. Make answers concise but informative (2-3 sentences)
3. Ensure questions and answers align with the article's content
4. Use {articlelanguage} language
5. Target the {articleaudience} audience
6. Maintain a {voicetone} tone
7. Use {pointofview} point of view

IMPORTANT: Return ONLY the JSON array in this EXACT format:
[
    {{
        "question": "What is {keyword}?",
        "answer": "Clear, concise answer about {keyword}. Additional context or detail."
    }},
    {{
        "question": "Another relevant question?",
        "answer": "Clear, factual answer. Supporting information."
    }}
]

DO NOT include any other text, explanations, or formatting - ONLY the JSON array."""

# System prompts
SYSTEM_MESSAGE = """You are an expert content writer creating cohesive, engaging articles.
Maintain consistent tone, style, and narrative flow throughout the piece.
Each response should build upon previous content while adding new value.
Focus on clarity, accuracy, and engaging storytelling.""" 