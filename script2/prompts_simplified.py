"""Simplified natural prompts for article generation using XML tags."""

# Simplified outline prompt - From 104 lines to 8 lines
OUTLINE_PROMPT = """Create a {sizesections}-section outline about {keyword} for {articleaudience} readers.
Focus on practical, actionable content. Use Roman numerals (I, II, III) for main sections and letters (A, B, C) for {sizeheadings} subsections each.
Aim for subsections that work well as tables or lists.
Return only the outline in <outline> tags."""

# Simplified title prompt - From 26 lines to 6 lines
TITLE_PROMPT = """Write a compelling title about {keyword} in {articlelanguage}.
Tone: {voicetone} for {articleaudience} readers.
Return only the title in <title> tags."""

# Simplified title crafting - From 27 lines to 7 lines
TITLE_CRAFT_PROMPT = """Enhance this title while keeping "{keyword}" intact: {title}
Make it more engaging for {articleaudience} readers. Keep 50-60 characters.
Return only the improved title in <title> tags."""

# Simplified introduction - From 18 lines to 8 lines
INTRODUCTION_PROMPT = """Write an engaging introduction for an article about {keyword} in {articlelanguage}.
Target {articleaudience} readers with {voicetone} tone. Use {pointofview} perspective.
Hook readers immediately, briefly outline what's coming, and transition smoothly to the main content.
Return only the introduction HTML in <intro> tags."""

# Simplified conclusion - From 16 lines to 7 lines
CONCLUSION_PROMPT = """Write a concise conclusion for an article about {keyword}.
Summarize key takeaways in 4-5 sentences for {articleaudience} readers.
Use {voicetone} tone and {pointofview} perspective.
Return only the conclusion HTML in <conclusion> tags."""

# Simplified paragraph generation - From 37 lines to 10 lines
PARAGRAPH_GENERATE_PROMPT = """Write paragraph {current_paragraph} of {paragraphs_per_section} for section "{subtitle}".
Target {articleaudience} readers in {articlelanguage} with {voicetone} tone.
Cover relevant points naturally from: {current_points}
Use appropriate HTML tags for structure and emphasis.
Return only the paragraph HTML in <paragraph> tags."""

# Simplified humanization - From 22 lines to 8 lines
HUMANIZE_PROMPT = """Make this text more natural and human-like: {humanize}
Use simple, everyday language. Keep sentences short and clear.
Maintain all paragraph breaks exactly as they appear.
Return only the humanized text in <humanized> tags."""

# Simplified FAQ generation - From 10 lines to 6 lines
FAQ_PROMPT = """Create 5 FAQs about {keyword} for {articleaudience} readers.
Write in {articlanguage}. Use bold for questions, concise answers.
Return only the FAQ HTML in <faq> tags."""

# Simplified PAA answer - From 21 lines to 8 lines
PAA_ANSWER_PROMPT = """Answer this question about {keyword}: "{question}"
Write {paragraphs} paragraphs for {articleaudience} readers in {articlelanguage}.
Be direct, informative, and engaging. Use {voicetone} tone.
Return only the answer text in <answer> tags."""

# Simplified grammar check - From 29 lines to 7 lines
GRAMMAR_CHECK_PROMPT = """Fix grammar, punctuation, and sentence structure in: {text}
Keep the original meaning and formatting. Maintain all paragraph breaks.
Return only the corrected text in <corrected> tags."""

# Simplified key takeaways - From 23 lines to 7 lines
BLOCKNOTE_KEY_TAKEAWAYS_PROMPT = """Extract 5-6 key takeaways from this article: {article_content}
Create a single cohesive paragraph with smooth transitions.
Focus on actionable insights for {articleaudience} readers.
Return only the takeaways in <takeaways> tags."""

# Simplified summary - From 28 lines to 7 lines
SUMMARIZE_PROMPT = """Summarize this article in 200-300 words: {article_content}
Create a single paragraph covering all key points for {articleaudience} readers.
Naturally include "{keyword}" where relevant.
Return only the summary in <summary> tags."""

# Simplified summary combination - From 25 lines to 7 lines
SUMMARY_COMBINE_PROMPT = """Combine these summary chunks into {num_paragraphs} cohesive paragraphs: {chunks_text}
Preserve all unique information with smooth transitions.
Return only the combined summary in <combined> tags."""

# Simplified meta description - From 27 lines to 6 lines
META_DESCRIPTION_PROMPT = """Write a 155-character meta description for {keyword} in {articlelanguage}.
Include the keyword naturally, describe the content, and add a call to action.
Target {articleaudience} readers. Return only the description in <meta> tags."""

# Simplified WordPress meta description - From 27 lines to 6 lines
WORDPRESS_META_DESCRIPTION_PROMPT = """Write a 155-character WordPress meta description for {keyword} in {articlelanguage}.
Include the keyword naturally, describe the content, and add a call to action.
Target {articleaudience} readers. Return only the description in <meta> tags."""

# Simplified blocknotes combination - From 23 lines to 7 lines
BLOCKNOTES_COMBINE_PROMPT = """Combine these key takeaways into {num_paragraphs} paragraphs: {chunks_text}
Focus on the 5-6 most critical insights with smooth transitions.
Return only the combined takeaways in <takeaways> tags."""

# Simplified paragraph - From 51 lines to 10 lines
PARAGRAPH_PROMPT = """Write paragraph {current_paragraph} of {paragraphs_per_section} for section "{subtitle}".
Target {articleaudience} readers in {articlelanguage} with {voicetone} tone.
Focus on: {current_points}
Use appropriate HTML structure based on the content type.
Return in format: <paragraph>content</paragraph>"""

# Simplified system message - From 4 lines to 3 lines
SYSTEM_MESSAGE = """You are an expert content writer creating engaging, natural articles.
Maintain consistent tone and flow. Focus on clarity and value for readers."""