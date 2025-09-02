"""Prompts for article generation."""

# Outline prompt for generating article structure
OUTLINE_PROMPT = """
===== CRITICAL FORMATTING REQUIREMENTS =====
You MUST create an outline that EXACTLY follows this format with NO DEVIATIONS:

{example_outline}

===== STRICT STRUCTURAL RULES =====
• EXACTLY {sizesections} main sections (numbered with Roman numerals I., II., III., etc.)
• EXACTLY {sizeheadings} subsections per main section (lettered A., B., C., etc.)
• EXACTLY one blank line between each main section
• NO Introduction or Conclusion sections in the outline
• NO extra text, explanations, or notes outside the outline format
• NO bullet points, numbered lists, or any other formatting

===== CONTENT GUIDELINES =====
• Main section titles: Clear, descriptive phrases about {keyword} (5-10 words)
• Subsection points: Specific aspects to cover (3-8 words each)
• Maximize subsections that trigger HTML table generation (e.g., "Compare key features," "Summarize benefits," "Analyze differences," "Overview of components," "Evaluate options," "Detailed specification matrix," "Side‑by‑side comparison," "Performance metrics table," "Feature vs. cost analysis," "Compatibility chart," "Pros and cons grid," "Benchmark results," "Model variations comparison," "Resource allocation overview," "Pricing tiers table")
• Maximize subsections that trigger HTML ordered list generation (e.g., "Steps for implementation," "Process overview," "Sequential strategies," "Action plan details," "Workflow breakdown," "Step‑by‑step tutorial," "Execution roadmap," "Rollout phases," "Migration plan," "Installation sequence," "Development milestones," "Onboarding procedure," "Upgrade path," "Checklist with priority order," "Quality assurance stages")
• Maximize subsections that trigger HTML unordered list generation (e.g., "Tips for success," "Key features list," "Best practices overview," "Practical suggestions," "Core attributes," "Advantages at a glance," "Important considerations," "Do’s and don’ts," "Essential requirements," "High‑level highlights," "Supportive resources," "Common pitfalls," "Design principles," "Functional capabilities," "Recommended tools")
• Ensure most subsections (at least 80%) are suitable for tables, ordered lists, or unordered lists to enhance scannability
• Content should be appropriate for {articleaudience} audience level
• Content should follow {articletype} style and approach
• All content must directly relate to {keyword}
• Use professional, engaging language appropriate for {articleaudience}

===== FORMATTING DETAILS =====
• Main sections: Roman numerals with period (I., II., III.)
• Subsections: Capital letters with period (A., B., C.)
• NO punctuation at the end of any line
• NO bold, italic, or other text formatting
• NO indentation variations - follow the example format exactly
• NO extra spaces or lines beyond the specified format
• NO HTML tags or any other formatting
• NO special characters or symbols
• NO emojis or informal language
• NO abbreviations or acronyms unless commonly known
• NO repetition of words or phrases
• NO TALKING, AS THIS WILL BE USED FOR AN API
• NO explanations or comments
• NO extra text or context
• NO unnecessary details or filler content
• NO comments such as here is your heading, or here is your outline etc.
• NO unnecessary words or phrases
• NO filler content or irrelevant information
• NO personal opinions or subjective statements
• NO assumptions about the reader's knowledge or background
• NO references to external sources or citations
• NO use of ** or any kind of markdown etc.
• NO quotes or paraphrasing from other sources
• NO use of "blog" or similar terms
• NO use of "article" or similar terms
• NO use of "content" or similar terms
• NO HTML or BODY Tag.

===== EXAMPLE OF CORRECT FORMAT =====
I. Understanding the Fundamentals of [Topic]
A. Core concept explanation
B. Compare foundational elements
C. Steps to grasp basics
D. Tips for effective learning
E. Analyze key differences

II. Practical Applications of [Topic]
A. Real-world use case
B. Summarize practical benefits
C. Action plan for implementation
D. Best practices overview
E. Evaluate application options

III. Advanced Strategies for [Topic] Success
A. Expert technique breakdown
B. Compare optimization approaches
C. Sequential strategies for success
D. Practical suggestions for experts
E. Overview of advanced components

IV. Future Trends in [Topic]
A. Emerging trend analysis
B. Summarize predicted outcomes
C. Workflow for future preparation
D. Core attributes of innovations
E. Evaluate future options

V. Measuring Success in [Topic]
A. Key performance metrics
B. Compare evaluation techniques
C. Steps for tracking progress
D. Practical improvement suggestions
E. Analyze success factors

After creating your outline, verify:
1. You have EXACTLY {sizesections} main sections
2. Each main section has EXACTLY {sizeheadings} subsections
3. The format matches the example precisely
4. No introduction or conclusion sections are included
5. No extra text appears outside the outline structure
6. At least 80% of subsections are suitable for triggering table, ordered list, or unordered list generation

Create a detailed outline for an article about {keyword}, targeting {articleaudience} readers:
"""

# Title generation prompt
TITLE_PROMPT = """Act as a blog post title writer who speaks and writes fluent {articlelanguage}. Create a blog post title for the keyword: {keyword}

The title should:
1. Have a hook and high potential to go viral on social media
2. Be 50-60 characters long
3. Include the full keyword without changing word order
4. Be written in {articlelanguage}
5. Match the tone: {voicetone}
6. Be suitable for {articletype} article type
7. Appeal to {articleaudience} level readers
8. Be raw text without any HTML tags or formatting
9. Avoid using any special characters, numbers, or punctuation
10. Be unique and not similar to existing titles
11. Avoid using the word "blog" in the title
12. Be catchy and engaging
13. Avoid using ** or any kind of markdown etc.
14. AVOID Wrapping the title in quotes, or **, or any other characters etc, at any cost.
15. NEVER RETURN AN EMPTY RESPONSE AT ANY COST.
16. DO NOT EVER WRAP THE TEXT IN QUOTES AT ANY COST.

Write only one title."""

# Title crafting prompt
TITLE_CRAFT_PROMPT = """You are an expert SEO copywriter specializing in optimizing article titles. Your task is to enhance this title while preserving the core keyword and meaning.

Original title: {title}
Main keyword: {keyword}

Requirements:
1. Keep the exact keyword phrase "{keyword}" intact and in the same order
2. Maintain a length between 50-60 characters
3. Add emotional appeal or curiosity factor
4. Ensure title remains clear and descriptive
5. Use power words appropriately
6. Keep it natural and not clickbait-y
7. Target {articleaudience} audience level
8. Match {voicetone} tone
9. Write in {articlelanguage}
10. Optimize for CTR and SEO

Rules:
- NO special characters or excessive punctuation
- NO ALL CAPS
- NO markdown or formatting
- NO quotation marks in output
- NO meta text or explanations
- Return only the enhanced title

RESPONSE FORMAT:
[Return only the enhanced title with no additional text]"""

# Introduction generation prompt
INTRODUCTION_PROMPT = """Assume the role of a seasoned writer specializing in captivating introductions for articles. Your task is to create engaging introductions for article {articletype} posts about {keyword} in {articlelanguage} language with {voicetone} voice tone, targeting {articleaudience} readers. Also use {pointofview}.

Each introduction should hook the reader immediately, setting the tone and context of the article while intriguing them to continue reading. Start with a compelling hook—this could be a provocative question, a surprising fact, a vivid anecdote, or a powerful quote. Briefly outline the main points that will be covered in the article, establishing your credibility on the subject.

Make sure the introduction aligns with the overall tone and style of the article, whether it's formal, conversational, or humorous. Include a transition at the end of the introduction that seamlessly leads into the main body of the article, ensuring a smooth reader experience. Adjust the complexity and terminology to match the {articleaudience} expertise level.

Create just 1 introduction and try to act as experienced human SEO copywriter. This human copywriter doesn't response MUST not include any of the following words and phrases: meticulous, meticulously, navigating, complexities, realm, understanding, dive in, shall, tailored, towards, underpins, everchanging, ever-evolving, the world of, not only, alright, embark, Journey, In today's digital age, hey, game changer, designed to enhance, it is advisable, daunting, in the realm of, amongst, unlock the secrets, unveil the secrets, and robust, diving, elevate, unleash, power, cutting-edge, rapidly, expanding, mastering, excels, harness, imagine, It's important to note, Delve into, Tapestry, Bustling, In summary, Remember that…, Take a dive into, Navigating, Landscape, Testament, In the world of, Realm, Embark, Analogies to being a conductor or to music, Vibrant, Metropolis, Firstly, Moreover, Crucial, To consider, There are a few considerations, Ensure, Furthermore, Fancy, As a professional, Therefore, Additionally, Specifically, Generally, Consequently, Importantly, nitty-gritty, Thus, Alternatively, Notably, As well as, Weave, Despite, Essentially, While, Also, Even though, Because, In contrast, Although, In order to, Due to, Even if, Arguably, On the other hand, It's worth noting that, To summarize, Ultimately, To put it simply, Promptly, Dive into, In today's digital era, Reverberate, Enhance, Emphasize, Revolutionize, Foster, Remnant, Subsequently, Nestled, Game changer, Labyrinth, Enigma, Whispering, Sights unseen, Sounds unheard, Indelible, My friend, Buzz, In conclusion.
MAKE SURE TO RETURN ONLY RAW HTML, NO MARKDOWN, NO **, NO EXTRA COMMENTS ETC.
I JUST WANT RAW HTML WITHOUT ANY MARKDOWN OR ANYTHING ELSE.
IT SHOULD INCORPORATE HTML FORMATTING TAGS SUCH AS STRONG, EM ETC. FOR BETTER, ENGAGING, AND SEO-FRIENDLY INTRODUCTION.
NEVER RETURN AN EMPTY RESPONSE AT ANY COST.
DO NOT EVER WRAP THE TEXT IN QUOTES AT ANY COST.
NEVER EVER RETURN THE TITLE, OUTLINE, OR ANY OTHER THING EXCEPT THE INTRODUCTION.
Make sure to make it look as human-like as possible, and please avoid any hyperbolic language.
Your content should be indistinguishable from human writing, with a natural flow and engaging style. Avoid using any phrases or words that would make it sound robotic or overly formal. The goal is to create content that resonates with the reader and provides real value.
Please ensure that the content is well-structured, easy to read, and provides clear information. Use appropriate HTML tags for emphasis and structure, such as <strong> for strong emphasis and <em> for italicized text. Avoid using any markdown or special characters that could disrupt the formatting.
NO HTML or BODY Tag.
"""

# Conclusion generation prompt
CONCLUSION_PROMPT = """Craft an attention-grabbing summarized conclusion part of blog post titled "{keyword}" for {articleaudience} readers.

Voice & Style: Consider infusing a distinctive voice and style that resonates with {articleaudience} expertise level. Is it authoritative and instructive? Conversational and friendly, Humorous and bit of witty style. Align the voice with the target audience and the overall article's objective.

Be strict in length because presentation of paragraph because your output should be - a paragraph up to 6 sentences for your output (please be random create 4 to 5 sentences) only. You're creating only conclusion section only. Don't use any emojis.

Revise the conclusion to strictly adhere to a maximum of 6 sentences while maintaining clarity, relevance, and engagement. Focus on emphasizing the key takeaway about {keyword}, while incorporating the main keyword naturally and without redundancy. Ensure the tone aligns with the intended audience, blending authority with an approachable style. Eliminate any unnecessary details or repetition to keep the conclusion concise and impactful.

MAKE SURE TO RETURN ONLY RAW HTML, NO MARKDOWN, NO **, NO EXTRA COMMENTS ETC.
I JUST WANT RAW HTML WITHOUT ANY MARKDOWN OR ANYTHING ELSE.
IT SHOULD INCORPORATE HTML FORMATTING TAGS SUCH AS STRONG, EM ETC. FOR BETTER, ENGAGING, AND SEO-FRIENDLY.
Make sure to make it look as human-like as possible, and please avoid any hyperbolic language.
Your content should be indistinguishable from human writing, with a natural flow and engaging style. Avoid using any phrases or words that would make it sound robotic or overly formal. The goal is to create content that resonates with the reader and provides real value.
Please ensure that the content is well-structured, easy to read, and provides clear information. Use appropriate HTML tags for emphasis and structure, such as <strong> for strong emphasis and <em> for italicized text. Avoid using any markdown or special characters that could disrupt the formatting.
NO HTML or BODY Tag.
"""


PARAGRAPH_GENERATE_PROMPT = """You are a seasoned SEO content writer with over a decade of experience crafting high-performing, keyword-optimized content for Fortune 500 companies and leading digital brands across diverse industries.

Your expertise lies in creating engaging, informative, and strategically structured content that consistently achieves top SERP rankings while maintaining exceptional readability and user engagement metrics.

Your task is to write paragraph {current_paragraph} of {paragraphs_per_section} for the section titled "{heading}" (Section {section_number} of {total_sections}). The content should be informative, engaging. Target {articleaudience} readers with appropriate complexity and examples.

The section needs to cover these {total_points} points across {paragraphs_per_section} paragraphs:
{all_points}

Guidelines:
1. Write approximately 150-200 words
2. This is paragraph {current_paragraph} of {paragraphs_per_section} - structure your content accordingly
3. Cover the points that flow naturally in this paragraph, considering what came before and what comes after
4. If this is the first paragraph ({current_paragraph} == 1), introduce the section's main topic
5. If this is the last paragraph ({current_paragraph} == {paragraphs_per_section}), ensure all points have been covered and provide a smooth conclusion
7. Include LSI (Latent Semantic Indexing) keywords where relevant
8. Ensure mobile-friendly readability with short sentences
9. Align with E-E-A-T principles
10. Add unique insights while maintaining topic coherence
11. Use smooth transitions between ideas
12. Adjust technical depth and terminology for {articleaudience} expertise level
13. MAKE SURE TO RETURN ONLY RAW HTML, NO MARKDOWN, NO **, NO EXTRA COMMENTS ETC.
14. I JUST WANT RAW HTML WITHOUT ANY MARKDOWN OR ANYTHING ELSE.
15. IT SHOULD INCORPORATE HTML FORMATTING TAGS SUCH AS STRONG, EM ETC. FOR BETTER, ENGAGING, AND SEO-FRIENDLY.

Parameters:
- Point of View: {pointofview}
- Voice Tone: {voicetone}
- Language: {articlelanguage}
- Article Type: {articletype}
- Target Audience: {articleaudience}

Write a cohesive paragraph that educates and engages the reader while maintaining SEO optimization.
Make sure to make it look as human-like as possible, and please avoid any hyperbolic language.
Your content should be indistinguishable from human writing, with a natural flow and engaging style. Avoid using any phrases or words that would make it sound robotic or overly formal. The goal is to create content that resonates with the reader and provides real value.
Please ensure that the content is well-structured, easy to read, and provides clear information. Use appropriate HTML tags for emphasis and structure, such as <strong> for strong emphasis and <em> for italicized text. Avoid using any markdown or special characters that could disrupt the formatting.
NO HTML or BODY Tag.
"""

# Paragraph generation prompt
# PARAGRAPH_GENERATE_PROMPT = """
# You are a veteran SEO content writer with 10+ years crafting high‑ranking, reader‑friendly copy for Fortune 500 brands.

# TASK  
# Write <strong>paragraph {current_paragraph}</strong> of <strong>{paragraphs_per_section}</strong> for the section <em>“{heading}”</em> (Section {section_number} of {total_sections}) in an article aimed at <strong>{articleaudience}</strong> readers.

# CONTENT TO COVER  
# Across the full section you must address <strong>{total_points}</strong> key points:  
# {all_points}

# FORMAT & LENGTH  
# • Choose the <em>best</em> HTML structure to convey this paragraph’s portion of the content—single &lt;p&gt;, bullet/numbered &lt;ul&gt;/&lt;ol&gt;, definition list &lt;dl&gt;, simple &lt;table&gt;, or a combination.<br>
# • Aim for ~150‑200 words if prose; if using list or table, provide 4‑6 concise items/rows of comparable information density.<br>
# • Return only raw HTML (no &lt;html&gt; or &lt;body&gt; tags, no Markdown, no extra commentary).

# FLOW RULES  
# 1. Respect {current_paragraph}/{paragraphs_per_section} placement:  
#    – If first, briefly introduce the section’s theme.  
#    – If last, finish remaining points and transition smoothly to the next section.  
# 2. Weave points logically with what precedes/follows.  
# 3. Use short sentences for mobile readability and smooth transitions.  
# 4. Sprinkle relevant LSI keywords naturally; follow E‑E‑A‑T principles.  
# 5. Tailor depth and terminology to <strong>{articleaudience}</strong> expertise.  
# 6. Emphasize clarity, authenticity, and unique insight—avoid hype or robotic tone.

# STYLE PARAMETERS  
# • Point of View: {pointofview}  
# • Voice & Tone: {voicetone}  
# • Language: {articlelanguage}  
# • Article Type: {articletype}

# OUTPUT  
# Produce human‑like, value‑driven content with appropriate HTML tags for emphasis (&lt;strong&gt;, &lt;em&gt;, etc.). Do not include any Markdown, asterisks, or comments—return <u>only</u> the final HTML snippet.
# """


# Text humanization prompt
HUMANIZE_PROMPT = """You are tasked with making the following text more human-like and natural. 

RULES:
1. Return ONLY the humanized text without any additional comments, explanations, or meta-text
2. Keep sentences brief and clear (10-20 words)
3. Use everyday words that are easy to understand
4. Avoid technical terms unless absolutely necessary
5. Write for an 8th-grade reading level
6. Skip overused business terms and jargon
7. Make direct statements without hedging
8. Use standard punctuation
9. Vary sentence structure naturally
10. PRESERVE ALL PARAGRAPH BREAKS exactly as in the original text
11. Maintain empty lines between paragraphs
12. CRITICAL: Do not combine separate paragraphs into a single paragraph
13. Never use: indeed, furthermore, thus, moreover, notwithstanding, ostensibly, consequently, specifically, notably, alternatively

TEXT TO HUMANIZE:
{humanize}

RESPONSE FORMAT:
[Start your response with the humanized text directly, no introduction or meta-text, preserving paragraph structure]"""

# FAQ generation prompt
FAQ_PROMPT = """{faqs} Develop an FAQ section for {keyword} that addresses common user questions. Provide valuable information to users and improve your site's search visibility. It Resolves the needs of target audience quickly. Be clear and concise.

I want you to act as an SEO blog post expert, who can guide me through the process of creating high-quality, search engine optimized content that attracts organic traffic and engages readers.

You can use this FAQ section to add content your readers are looking from top searches of search engines, with clear, concise answers.

Important please do not write bullets, numbers or Q+A characters. Just use bold style for question. Then write a short, precise, accurate, very specific answer in {articlanguage} to the question.

Write all output in 100% {articlanguage} and top grammar style. Generate a list of 5 frequently asked questions and their answers.
NO HTML or BODY Tag.
"""

# PAA answer generation prompt
PAA_ANSWER_PROMPT = """You are writing an answer for a "People Also Ask" section in an article about "{keyword}".
The question is: "{question}"

Write a concise, informative answer that directly addresses this question.
The answer should be helpful, accurate, and provide value to the reader.
Format your answer in {paragraphs} paragraph(s) using plain text paragraphs separated by double line breaks.
Each paragraph should be about 80-120 words for a total of approximately {paragraphs_word_count} words.

Remember to:
- Be direct and answer the question clearly
- Structure your answer logically across {paragraphs} paragraph(s) with CLEAR PARAGRAPH BREAKS between them
- Include relevant facts or details
- Write in a natural, engaging style
- Maintain the context of the main article topic: "{keyword}"
- Adjust tone to match: {voicetone}
- Use appropriate language: {articlelanguage}
- Target audience expertise level: {articleaudience}
- Use appropriate {pointofview} point of view
- Use only plain text paragraphs - NO HTML tags, NO special formatting
- PRESERVE PARAGRAPH STRUCTURE: Make sure to separate paragraphs with double line breaks

Structure guidelines for a {paragraphs}-paragraph answer:
- First paragraph: Directly address the question and provide the core answer
- Middle paragraph(s): Expand with details, examples, or supporting information
- Last paragraph: Summarize or provide final insights/recommendations

FORMAT YOUR RESPONSE WITH DISTINCT PARAGRAPHS SEPARATED BY BLANK LINES. Each paragraph should be its own distinct block of text.
"""

# Grammar check prompt
GRAMMAR_CHECK_PROMPT = """You are a professional proofreading expert with decades of experience in correcting grammatical errors.

SYSTEM ROLE:
You are a precise grammar correction system that focuses solely on fixing grammatical issues while maintaining the original meaning and style.

TASK:
Analyze and correct the following text for:
- Grammatical errors
- Sentence structure issues
- Punctuation mistakes
- Verb tense consistency
- Word usage accuracy
- Readability and flow

RULES:
1. Return ONLY the corrected text
2. Do not include any explanations or comments
3. Do not add suggestions or improvements
4. Do not include any meta-text or analysis
5. Preserve all original formatting including HTML, AND PLEASE NO MARKDOWN
6. Maintain the exact same meaning and intent
7. Keep the same tone and style
8. PRESERVE ALL PARAGRAPH BREAKS exactly as in the original text
9. Maintain empty lines between paragraphs
10. CRITICAL: Separate paragraphs with blank lines exactly as in the original
11. DO NOT merge multiple paragraphs into a single paragraph

TEXT TO PROOFREAD:
{text}

RESPONSE FORMAT:
[Provide only the corrected text with no additional commentary, preserving paragraph structure]"""

# Block note key takeaways prompt
BLOCKNOTE_KEY_TAKEAWAYS_PROMPT = """You are an SEO Specialist tasked with creating a concise summary of the article's key takeaways. Your goal is to create a single cohesive paragraph that captures the most important information readers should remember from the following article:

Article Content:
{article_content}

RULES:
1. Create a single paragraph (no bullet points or numbered lists)
2. Paragraph should be 5-6 sentences maximum
3. Use clear, concise language that flows naturally
4. Focus on actionable insights and practical information
5. Avoid technical jargon unless absolutely necessary
6. Highlight 5-6 most important points from the article
7. Connect ideas with smooth transitions
8. Total output should not exceed 150 words
9. Do not include any introductory text or conclusions
10. Write in an authoritative, professional tone
11. NO HTML or BODY Tag.

OUTPUT FORMAT:
[Single paragraph containing 5-6 key takeaways with smooth transitions between points]

RESPONSE FORMAT:
[Start directly with the paragraph, no introduction or meta-text]"""

# Summary generation prompt
SUMMARIZE_PROMPT = """Create a professional summary of the article content that strictly adheres to these requirements:

1. Length: 200-300 words
2. Content: Comprehensive overview of all key points
3. Structure: Single cohesive paragraph with smooth transitions
4. Style: Matches the article's original tone and voice
5. Audience: Tailored for {articleaudience} readers
6. Keyword: Naturally incorporates "{keyword}" with 1-2% density
7. Format: HTML only (no markdown, no **, no extra comments)
8. Tags: Use <strong> and <em> for emphasis where appropriate
9. Prohibited phrases:
   - "Here's a summary"
   - "In this article"
   - "We'll cover"
   - "Let's dive in"
   - Any other conversational phrases
10. Directives:
   - Begin immediately with the summary content
   - No introductory phrases
   - No meta-commentary
   - No explanations
   - No references to the summary itself
   - No HTML or BODY tags

The summary must be indistinguishable from professional human writing, with natural flow and precise language. Focus exclusively on conveying the article's key information in a clear, concise manner.

Article content:
{article_content}

RESPONSE FORMAT:
[Begin summary directly without any introductory text]"""

# Summary combination prompt
SUMMARY_COMBINE_PROMPT = """Combine multiple chunks of article summary into a single coherent result following these strict requirements:

1. Length: {num_paragraphs} paragraph(s) exactly
2. Content: All unique information from chunks preserved
3. Structure: Well-organized and logically structured
4. Style: Matches original chunks' tone and voice
5. Format: HTML only (no markdown, no **, no extra comments)
6. Tags: Use <strong> and <em> for emphasis where appropriate
7. Line breaks: Add <br><br> between paragraphs for clear separation
8. Prohibited phrases:
   - "Here's the combined summary"
   - "I'll now combine these chunks"
   - Any other conversational phrases
9. Directives:
   - Begin immediately with the combined content
   - No introductory phrases
   - No meta-commentary
   - No explanations
   - No references to the combination process
   - No HTML or BODY tags

Chunks to combine:
{chunks_text}

RESPONSE FORMAT:
[Begin combined summary directly without any introductory text]"""

# Meta description prompt
META_DESCRIPTION_PROMPT = """You are a world-class SEO content writer specializing in generating content that is indistinguishable from human authorship. Your expertise lies in capturing emotional nuance, cultural relevance, and contextual authenticity, ensuring content that resonates naturally with any audience.

Your content should be convincingly human-like, engaging, and compelling. Create a meta description that strategically incorporates relevant keywords for {keyword}, enhancing our site's visibility.

When it comes to writing meta descriptions for SEO, you should do the following:
- Include Your Focus Keyword
- Describe Your Page's Content
- Include a Call to Action
- Make Them Unique and Interesting in {articlelanguage} language

The strict and important character limit is 155 only characters NOT more in your output. Please re-check again your output with character limit before generate and do not include anything else just meta description.

Combine stylistic points about rhetorical questions, analogies, and emotional cues into a streamlined guideline to reduce overlap. Adjust tone dynamically: keep it conversational and engaging for {articleaudience} audiences, and more formal or precise for professional topics. Use emotional cues sparingly for technical content."""

# WordPress meta description prompt
WORDPRESS_META_DESCRIPTION_PROMPT = """You are a world-class SEO content writer specializing in generating content that is indistinguishable from human authorship. Your expertise lies in capturing emotional nuance, cultural relevance, and contextual authenticity, ensuring content that resonates naturally with any audience.

Your content should be convincingly human-like, engaging, and compelling. Create a meta description that strategically incorporates relevant keywords for {keyword}, enhancing our site's visibility.

When it comes to writing meta descriptions for SEO, you should do the following:
- Include Your Focus Keyword
- Describe Your Page's Content
- Include a Call to Action
- Make Them Unique and Interesting in {articlelanguage} language

The strict and important character limit is 155 only characters NOT more in your output. Please re-check again your output with character limit before generate and do not include anything else just meta description.

Combine stylistic points about rhetorical questions, analogies, and emotional cues into a streamlined guideline to reduce overlap. Adjust tone dynamically: keep it conversational and engaging for {articleaudience} audiences, and more formal or precise for professional topics. Use emotional cues sparingly for technical content."""

# Blocknotes combination prompt
BLOCKNOTES_COMBINE_PROMPT = """Combine multiple chunks of key takeaways into a single coherent paragraph following these strict requirements:

1. Length: {num_paragraphs} paragraph(s) exactly
2. Content: Only the most critical and actionable insights preserved
3. Structure: Logical flow with smooth transitions between points
4. Style: Authoritative and directly useful tone
5. Format: HTML only (no markdown, no **, no extra comments)
6. Tags: Use <strong> and <em> for emphasis on key concepts
7. Include: Only 5-6 most essential points from all chunks
8. Line breaks: Add <br><br> between paragraphs for clear separation
9. Prohibited phrases:
   - "Here are the key takeaways"
   - "Based on the article"
   - "In conclusion"  
   - Any other meta-commentary
10. Directives:
    - Begin with the most important insights
    - Focus on practical value
    - Use clear, precise language
    - Maintain professional tone
    - Make each point actionable
    - Add line breaks between paragraphs
    - NO HTML or BODY tags


Key takeaways to combine:
{chunks_text}

RESPONSE FORMAT:
[Begin with the combined key takeaways directly, no introduction]"""


PARAGRAPH_WITH_HEADING_PROMPT = """You are an expert SEO content writer with over 10 years of experience creating high-ranking, reader-focused content for Fortune 500 companies and top digital brands. Your expertise ensures content is engaging, informative, and structured for top SERP performance while maintaining excellent readability.

Your task is to write paragraph {current_paragraph} of {paragraphs_per_section} for the section titled "{heading}" (Section {section_number} of {total_sections}). Target {articleaudience} readers with appropriate complexity and engaging examples, optimizing for SEO and WordPress compatibility.

Requirements:
1. Write ONE cohesive paragraph (3-5 sentences, 80-120 words) focusing on a single main point.
2. Use {articlelanguage} language, {voicetone} tone, and {pointofview} point of view.
3. Include specific details or actionable examples to engage {articleaudience}.
4. Integrate the primary keyword ({primary_keyword}) and LSI keywords ({lsi_keywords}) naturally, targeting 1-2% keyword density.
   - Example LSI keywords for "SEO": "search engine ranking," "keyword research," "on-page optimization."
5. Use HTML tags for scannability and emphasis (no Markdown), selecting the most effective structure based on the content and trigger words (or their synonyms) in {current_points} or {flow_instruction}:
   - `<p>`: Use for narrative or descriptive content. Trigger words: explain or describe or discuss or explore or introduce or define or elaborate or clarify or highlight or summarize or overview or provide context or detail or narrate or analyze or reflect or emphasize or transition or conclude or illustrate or outline (non-sequential) or review or address or present or articulate or convey or examine or delve or expand or interpret.
     - Example:
       <h3>SEO Basics</h3>
       <p>Exploring <strong>keyword research</strong> helps small businesses improve their <strong>search engine ranking</strong>. By identifying high-traffic terms, businesses can craft content that attracts the right audience. For instance, a coffee shop might target "best coffee near me" to draw local customers. This approach ensures content aligns with user intent. Strong keyword strategies boost online visibility.</p>
   - `<strong>`: Emphasize 1-2 key phrases/keywords per paragraph.
   - `<em>`: Use 0-1 times for subtle emphasis.
     - Example:
       <h3>SEO Basics</h3>
       <p>Keyword research is vital for <em>targeted content creation</em>, ensuring your site ranks for relevant terms. It involves analyzing search trends to optimize content effectively. Small businesses benefit from this <strong>SEO</strong> practice. Tools like Google Keyword Planner simplify the process. Higher rankings follow consistent efforts.</p>
   - `<ul><li>`: Include 2-4 item unordered list for tips, benefits, or non-sequential items. Trigger words: tips or benefits or features or advantages or reasons or examples or ideas or strategies or techniques or factors or elements or components or options or suggestions or best practices or guidelines (non-sequential) or key points or considerations or tools or resources or attributes or aspects or items or characteristics or approaches or methods or tactics or insights or pointers.
     - Example:
       <h3>SEO Basics</h3>
       <p>Effective <strong>on-page optimization</strong> offers key benefits for small businesses: <ul><li>Optimized title tags improve click-through rates.</li><li>Clear meta descriptions enhance <strong>search engine ranking</strong>.</li><li>Fast site speed boosts user experience.</li><li>Keyword-rich content attracts targeted traffic.</li></ul> These techniques drive better visibility.</p>
   - `<ol><li>`: Include 2-4 item ordered list for steps, processes, or sequences. Trigger words: steps or process or procedure or sequence or instructions or stages or phases or order or workflow or method or approach or plan or roadmap or guidelines (sequential) or priorities or hierarchy or chronology or timeline or progression or directions or protocol or system or series or steps to follow or blueprint or itinerary or schedule.
     - Example:
       <h3>SEO Basics</h3>
       <p>To enhance <strong>SEO</strong>, follow this process for <strong>keyword research</strong>: <ol><li>Use tools like Google Keyword Planner to find relevant terms.</li><li>Analyze competitors’ keywords for opportunities.</li><li>Prioritize long-tail keywords for niche markets.</li><li>Integrate keywords naturally into content.</li></ol> This method improves rankings.</p>
   - `<table>` with `<thead>`, `<tr>`, `<th>`, `<td>`: Use for comparisons, summaries, or structured data. Trigger words: compare or comparison or contrast or summarize or summary or overview (data-focused) or breakdown or analysis or metrics or data or statistics or results or features (comparative) or differences or similarities or specifications or criteria or evaluation or side-by-side or chart (data-focused) or table or assessment or review (data-focused) or metrics comparison or data points or distinctions or parallels or benchmarks.
     - Example:
       <h3>SEO Basics</h3>
       <p>A breakdown of <strong>SEO</strong> strategies clarifies their roles: <table><thead><tr><th>Strategy</th><th>Focus Area</th></tr></thead><tbody><tr><td>On-page SEO</td><td>Content and <strong>keyword research</strong> optimization.</td></tr><tr><td>Off-page SEO</td><td>Backlinks and social signals for authority.</td></tr></tbody></table> Both improve <strong>search engine ranking</strong>.</p>
6. Start with a clear topic sentence and end with a conclusion or transition.
7. Ensure cohesion and alignment with {flow_instruction}.
8. Use `<h3>` for sections 1-2 or `<h4>` for deeper sections in the heading.
9. Maintain a professional, human-like tone, avoiding exaggeration.
10. Select the HTML structure based on trigger words or their synonyms in {current_points} or {flow_instruction}. If no trigger words or synonyms are present, default to `<p>`.

Section points to cover across {paragraphs_per_section} paragraphs:
{all_points}

Focus for this paragraph ({current_paragraph} of {paragraphs_per_section}):
{current_points}

Structure content to fit the section’s flow: {flow_instruction}.

CRITICAL: YOU MUST USE THE EXACT FORMAT TAGS AS SHOWN BELOW!
<paragraph>Your paragraph content here with proper formatting as needed...</paragraph>
<heading>Brief, descriptive heading that summarizes the above content</heading>

IMPORTANT: Start your response with <paragraph> tag, followed by your paragraph text, then the <heading> tag followed by the heading text. Do not deviate from this exact format or your response will be unusable.

"""


# System prompts
SYSTEM_MESSAGE = """You are an expert content writer creating cohesive, engaging articles.
Maintain consistent tone, style, and narrative flow throughout the piece.
Each response should build upon previous content while adding new value.
Focus on clarity, accuracy, and engaging storytelling."""