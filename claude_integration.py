import os
import anthropic
from typing import List
from datetime import datetime
from dotenv import load_dotenv
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
import re

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def get_search_response(query: str, entries: List[dict]) -> str:
    """Generate a response using Claude API for search results"""
    try:
        if not entries:
            return "I couldn't find any relevant entries matching your query."
        
        # Format entries for Claude
        entries_text = "\n\n".join([
            f"Entry {i+1}:\nURL: {entry['url']}\nContent: {entry['content']}\n"
            f"Thoughts: {entry['thoughts']}\nDate: {entry['created_at']}\n"
            f"Similarity: {entry['similarity']:.2f}"
            for i, entry in enumerate(entries)
        ])
        
        prompt = f"""
        The user searched for: "{query}"
        
        Here are the top relevant entries found in their database:
        
        {entries_text}
        
        Please provide a helpful response summarizing these entries and explaining 
        why they're relevant to the search query. If there are common themes, 
        highlight them. If some entries are more relevant than others, explain why.
        """
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    except anthropic.APIError as e:
        logging.error(f"Claude API error: {e}")
        return "I encountered an issue while generating a response. Please try again later."
    except Exception as e:
        logging.error(f"Unexpected error in Claude integration: {e}")
        return "An unexpected error occurred while processing your search results."

def get_weekly_rollup_response(entries: List, start_date: datetime, end_date: datetime) -> str:
    """Generate a weekly rollup response using Claude API"""
    try:
        if not entries:
            return f"No entries were found for the period from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}."
        
        # Format entries for Claude
        entries_text = "\n\n".join([
            f"Entry {i+1}:\nURL: {entry.url}\nContent: {entry.content}\n"
            f"Thoughts: {entry.thoughts}\nDate: {entry.created_at}"
            for i, entry in enumerate(entries)
        ])
        
        prompt = f"""
        Here's a weekly rollup of content saved from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}:
        
        {entries_text}
        
        Please provide a comprehensive weekly summary of these entries. The summary should:
        1. Identify major themes and topics across the saved content
        2. Highlight the most interesting or important entries
        3. Make connections between related entries where possible
        4. Be structured in a clear, readable format
        5. End with potential action items or follow-up ideas based on the content

        Use the following format as an example:
        # 🗓️ Weekly Resource Collection
            February 15, 2025

            ## 🤖 AI Development & Tools

            ### 🤝 AI Agents & Architecture
            - [2025 AI Agent Market Map](https://x.com/atomsilverman/status/1890534522560663806?s=46) - Comprehensive overview of AI agent landscape
            - [Proxy 1.0 Release](https://x.com/ai_for_success/status/1892137785420833273?s=46) - Convergence's new web agent
            - [Multi-agent AI Framework](https://x.com/sumanth_077/status/1891491497121354128?s=46) - Framework for multi-agent systems
            - [AI Agent Memory Systems](https://x.com/aurimas_gr/status/1892196166973977034?s=46) - Discussion of agent memory capabilities
            - [Native Sparse Attention Research](https://x.com/deepseek_ai/status/1891745487071609327?s=46) - New LLM architecture with performance improvements
            - [MUSE by Microsoft](https://x.com/_akhaliq/status/1892263116496400859?s=46) - Generative AI for gameplay

            ### 🛠️ Development Tools & Resources
            - [VSCode Sidebar Configuration](https://x.com/aidenybai/status/1890457226340700361?s=46) - Right-sided folder sidebar optimization
            - [Cursor/Windsurf Context Tip](https://x.com/cj_zzzz/status/1892615770040877494?s=46) - Reducing AI coding errors
            - [Cloud Computing Credits](https://x.com/brennanwoodruff/status/1892222642326839400?s=46) - Free credits from major providers

            ## 🔍 Research & Knowledge Management

            ### 📚 Research Tools
            - [Rabbit Hole App](https://x.com/mohams2001/status/1890103682630406403?s=46) - Tool for exploring follow-up questions
            - [Deep Research Prompting](https://x.com/buccocapital/status/1890745551995424987?s=46) - Meta-approach using O1 Pro for research
            - [Knowledge Graphs](https://x.com/svpino/status/1891488282040750344?s=46) - Knowledge organization systems
            - [Open Source Knowledge Management](https://x.com/tom_doerr/status/1891966398588387598?s=46) - Software solutions

            ### 📄 Document Processing
            - [PDF Content Extraction](https://x.com/rohanpaul_ai/status/1890552874155090388?s=46) - PDF processing tools
            - [PDF Analyzer](https://x.com/tom_doerr/status/1890346811350470989?s=46) - PDF analysis capabilities

            ## 📖 Learning & Education

            ### 💻 Technical Education
            - [Compiler Learning Resource](https://ssloy.github.io/tinycompiler/) - "A compiler in a week-end"
            - [Twitter Discussion](https://x.com/ludwigabap/status/1892500346833936779?s=46) - Compiler learning overview
            - [LLM Learning Roadmap](https://x.com/jxmnop/status/1890826135203717427?s=46) - Comprehensive path from basics to advanced AI

            ### 🧠 Personal Development
            - [Self-Discipline vs. Psychological Needs](https://youtu.be/V6hN8raThYk?si=su3gJLgRBzFt5Rwa) - Video on behavioral patterns
            - [Time Management System](https://youtu.be/VpN78TXMSUM?si=8FbaHQSD1ItfxfHb) - Ali Abdaal's productivity approach

            ## 📝 Content Creation & Communication
            - [Social Media Post Checklist](https://x.com/thepatwalls/status/1890798723342426144?s=46) - Framework for engaging content
            - [AI Content Automation](https://x.com/juliangoldieseo/status/1892439039627972640?s=46) - Multi-agent content system
            - [Effective Prompt Design](https://x.com/aakashg0/status/1890492955842007087?s=46) - Anatomy of an o1 prompt
            - [Prompting as Programming](https://x.com/mckaywrigley/status/1891925465062887524?s=46) - Conceptual framework for prompting

            ## 🧪 AI Theory & Research
            - [LLM Architecture Innovation](https://x.com/matthewberman/status/1890081482104008920?s=46) - Reasoning in latent space
            - [Grokking Research Paper](http://arxiv.org/abs/2412.18624) - Thermodynamic approach to understanding AI learning
            - [AI Implementation Insights](https://x.com/ztc1/status/1890888408047984926?s=46) - Practical AI integration strategies

            ## ✅ Personal ToDos
            - 
        """
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    except anthropic.APIError as e:
        logging.error(f"Claude API error: {e}")
        return "I encountered an issue while generating your weekly rollup. Please try again later."
    except Exception as e:
        logging.error(f"Unexpected error in Claude integration: {e}")
        return "An unexpected error occurred while processing your weekly rollup."

def is_youtube_url(url: str) -> bool:
    """Check if a URL is a YouTube video."""
    youtube_regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.match(youtube_regex, url)
    return bool(match)

def extract_youtube_id(url: str) -> str:
    """Extract the YouTube video ID from a URL."""
    youtube_regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(youtube_regex, url)
    if match:
        return match.group(1)
    return None

def get_youtube_transcript_summary(url: str) -> str:
    """
    Get a transcript for a YouTube video and summarize the top 3 takeaways using Claude.
    Returns the summary as a string.
    """
    try:
        # Extract the video ID from the URL
        video_id = extract_youtube_id(url)
        if not video_id:
            return f"Could not extract YouTube video ID from URL: {url}"
        
        # Get the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all text
        full_transcript = " ".join([item['text'] for item in transcript_list])
        
        # Truncate if very long (Claude has context limits)
        if len(full_transcript) > 25000:
            full_transcript = full_transcript[:25000] + "... [transcript truncated due to length]"
        
        # Create prompt for Claude
        prompt = f"""
        I need you to summarize a YouTube video transcript into the 3 most essential takeaways.
        Focus on the key insights, main arguments, or most important points.
        Format the response as 3 concise bullet points that capture the core value of the content.
        
        Here's the transcript:
        
        {full_transcript}
        """
        
        # Get Claude's summary
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        summary = response.content[0].text
        
        # Add a header to clarify this is an AI summary
        return f"## AI-Generated Summary of YouTube Video\n\n{summary}"
        
    except Exception as e:
        # Generic exception handling instead of specific types
        logging.error(f"Error extracting YouTube transcript: {e}")
        return f"Failed to extract and summarize YouTube transcript: {str(e)}" 