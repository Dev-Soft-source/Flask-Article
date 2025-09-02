# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import sys
from config import Config
from utils.prompts_config import Prompts
from article_generator.article_context import ArticleContext
from utils.rich_provider import provider

def main():
    """Test article context functionality."""
    try:
        provider.info("Testing article context functionality...")

        # Create a test config with context saving enabled
        config = Config(
            # Enable context saving
            enable_context_save=True,
            context_save_dir="article_contexts_test",

            # API settings
            openai_model="gpt-4o-mini-2024-07-18",
            use_openrouter=True,
            openrouter_model="meta-llama/llama-3.3-70b-instruct:free",

            # Other required settings
            articlelanguage="English",
            articleaudience="General",
            articletype="Default",
            voicetone="friendly",
            pointofview="Third Person",
            sizesections=1,
            sizeheadings=1,
            paragraphs_per_section=1,

            # Token settings
            max_context_window_tokens=128000,
            token_padding=1000,
            warn_token_threshold=0.9,
            enable_token_tracking=True,
        )

        # Create prompts
        prompts = Prompts(
            system_message="You are an expert content writer creating cohesive, engaging articles.",
            title="Generate a title for an article about {keyword}.",
            outline="Create an outline for an article about {keyword}.",
            introduction="Write an introduction for an article about {keyword}.",
            paragraph="Write a paragraph about {subtitle} related to {keyword}.",
            conclusion="Write a conclusion for an article about {keyword}.",
            faq="Generate FAQ questions and answers about {keyword}.",
            meta_description="Write a meta description for an article about {keyword}.",
            wordpress_excerpt="Write a WordPress excerpt for an article about {keyword}.",
            grammar="Improve the grammar of this text: {text}",
            humanize="Make this text more engaging and human-like: {text}",
            blocknote="Create block notes (key takeaways) for this article about {keyword}.",
            summarize="Summarize this article about {keyword}.",
            paa_answer="Answer this question about {keyword}: {question}"
        )

        # Create article context
        context = ArticleContext(config, prompts)

        # The ArticleContext already adds a system message during initialization,
        # so we don't need to add it again
        provider.info("System message already added during initialization...")

        # Add user messages
        provider.info("Adding user messages...")
        context.add_message("user", "Generate a title for an article about artificial intelligence.")
        context.add_message("assistant", "Understanding Artificial Intelligence: A Beginner's Guide")

        context.add_message("user", "Create an outline for this article.")
        context.add_message("assistant", "1. Introduction to AI\n2. History of AI\n3. Types of AI\n4. Applications of AI\n5. Future of AI")

        context.add_message("user", "Write an introduction for this article.")
        context.add_message("assistant", "Artificial Intelligence (AI) has become an integral part of our daily lives, from voice assistants like Siri and Alexa to recommendation systems on streaming platforms. This article aims to provide a comprehensive overview of AI, its history, types, applications, and future prospects.")

        context.add_message("user", "Write a paragraph about the history of AI.")
        context.add_message("assistant", "The history of AI dates back to the 1950s when the term was first coined by John McCarthy. The field has seen periods of rapid advancement followed by 'AI winters' where progress slowed. Early AI research focused on symbolic reasoning and rule-based systems, while modern AI has shifted towards machine learning and neural networks.")

        context.add_message("user", "Write a paragraph about types of AI.")
        context.add_message("assistant", "AI can be categorized into several types based on capabilities and design. Narrow or Weak AI is designed for specific tasks like facial recognition or language translation. General or Strong AI would possess human-like intelligence across various domains, though this remains theoretical. Machine Learning, Deep Learning, and Neural Networks are subfields that focus on systems that learn from data.")

        context.add_message("user", "Write a paragraph about applications of AI.")
        context.add_message("assistant", "AI applications span numerous industries and domains. In healthcare, AI assists in diagnosis, drug discovery, and personalized treatment plans. In finance, it powers fraud detection, algorithmic trading, and risk assessment. Transportation benefits from autonomous vehicles and traffic optimization. AI also enhances customer service through chatbots, content creation through generative models, and manufacturing through predictive maintenance and quality control.")

        context.add_message("user", "Write a conclusion for this article.")
        context.add_message("assistant", "As we continue to advance in the field of Artificial Intelligence, the possibilities seem endless. While concerns about ethics, job displacement, and privacy exist, the potential benefits of AI in solving complex problems, enhancing human capabilities, and improving quality of life are substantial. Understanding the fundamentals of AI is crucial for navigating our increasingly AI-driven world and participating in important discussions about its responsible development and deployment.")

        # Add article parts
        context.article_parts["title"] = "Understanding Artificial Intelligence: A Beginner's Guide"
        context.article_parts["outline"] = "1. Introduction to AI\n2. History of AI\n3. Types of AI\n4. Applications of AI\n5. Future of AI"
        context.article_parts["introduction"] = "Artificial Intelligence (AI) has become an integral part of our daily lives, from voice assistants like Siri and Alexa to recommendation systems on streaming platforms. This article aims to provide a comprehensive overview of AI, its history, types, applications, and future prospects."
        context.article_parts["sections"] = [
            "The history of AI dates back to the 1950s when the term was first coined by John McCarthy. The field has seen periods of rapid advancement followed by 'AI winters' where progress slowed. Early AI research focused on symbolic reasoning and rule-based systems, while modern AI has shifted towards machine learning and neural networks.",
            "AI can be categorized into several types based on capabilities and design. Narrow or Weak AI is designed for specific tasks like facial recognition or language translation. General or Strong AI would possess human-like intelligence across various domains, though this remains theoretical. Machine Learning, Deep Learning, and Neural Networks are subfields that focus on systems that learn from data.",
            "AI applications span numerous industries and domains. In healthcare, AI assists in diagnosis, drug discovery, and personalized treatment plans. In finance, it powers fraud detection, algorithmic trading, and risk assessment. Transportation benefits from autonomous vehicles and traffic optimization. AI also enhances customer service through chatbots, content creation through generative models, and manufacturing through predictive maintenance and quality control."
        ]
        context.article_parts["conclusion"] = "As we continue to advance in the field of Artificial Intelligence, the possibilities seem endless. While concerns about ethics, job displacement, and privacy exist, the potential benefits of AI in solving complex problems, enhancing human capabilities, and improving quality of life are substantial. Understanding the fundamentals of AI is crucial for navigating our increasingly AI-driven world and participating in important discussions about its responsible development and deployment."

        # Save context to file
        provider.info("Saving context to file...")
        filepath = context.save_to_file("test_context_only.md")

        provider.success("Context saved successfully!")
        provider.info(f"Context saved to: {filepath}")
        provider.info(f"Number of messages in context: {len(context.messages)}")

        # Print message roles to verify
        message_roles = [msg["role"] for msg in context.messages]
        provider.info(f"Message roles: {message_roles}")

    except Exception as e:
        provider.error(f"Error in test: {str(e)}")
        import traceback
        provider.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
