"""
Claude AI integration module.
"""

import os
import anthropic
from typing import List
from datetime import datetime
from dotenv import load_dotenv
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
import re

# Import from config
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_CONTENT_LENGTH, LOG_LEVEL

load_dotenv()

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("claude_integration")

# Initialize the Anthropic client
try:
    # Initialize with the latest client format
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    logger.info(f"Initialized Anthropic client with API key: {ANTHROPIC_API_KEY[:4]}...")
    logger.info(f"Using Claude model: {CLAUDE_MODEL}")
except Exception as e:
    client = None
    logger.error(f"Failed to initialize Anthropic client: {e}")
    logger.error("Claude integration will not be available")

def get_search_response(query: str, entries: List[dict]) -> str:
    """Generate a response from Claude for a search query"""
    try:
        # Format entries for prompt
        entries_text = ""
        for i, entry in enumerate(entries, 1):
            entries_text += f"\nEntry {i}:\nURL: {entry['url']}\nContent: {entry['content']}\nThoughts: {entry['thoughts']}\n"
        
        # Create prompt
        prompt = f"""
        I'd like you to help me understand the following content related to my search query.
        
        My search query: {query}
        
        Here are the top search results:
        {entries_text}
        
        Please provide:
        1. A concise summary of the relevant information from these results
        2. Connections between the different pieces of content
        3. Key insights these sources highlight about my query
        
        If the search results don't seem relevant to my query, please let me know.
        """
        
        if client is None:
            return "Claude API is not available. Please check your API key and SDK installation."
            
        # Call Claude with the latest API format
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Error calling Claude API: {e}")
            return f"Error processing with Claude API: {str(e)}"
            
    except Exception as e:
        logging.error(f"Unexpected error in Claude integration: {e}")
        return f"Sorry, I encountered an error while processing your request: {str(e)}"

def get_weekly_rollup_response(entries: List[dict]) -> str:
    """Generate a weekly rollup response using Claude API"""
    try:
        if not entries:
            return "No entries found for the specified period."
        
        # Format entries for Claude prompt
        entries_text = ""
        for i, entry in enumerate(entries, 1):
            entry_date = entry['created_at'].strftime("%Y-%m-%d %H:%M")
            entries_text += f"\nEntry {i}:\nDate: {entry_date}\nURL: {entry['url']}\nContent: {entry['content']}\nThoughts: {entry['thoughts']}\n"
        
        prompt = f"""
        I'd like you to analyze my saved content from the past week and provide insights.
        
        Here are the entries:
        {entries_text}
        
        Please reference the format below when constructing your response:

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
            - Try VSCode with right-sided folder sidebar at Wendy's/Chase
            - Explore Cursor as alternative IDE
            - Evaluate IDE setup feasibility at work
            - Check out Rabbit Hole app functionality
            - Consider how Rabbit Hole might fit into current research workflow
        
        Focus on being insightful rather than just summarizing each entry individually. Ensure that all links are provided. Use emojis in your headings!
        """
        
        if client is None:
            return "Claude API is not available. Please check your API key and SDK installation."
            
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Error calling Claude API: {e}")
            return f"Error processing with Claude API: {str(e)}"
            
    except Exception as e:
        logging.error(f"Error generating weekly rollup: {e}")
        return f"I encountered an error while generating your weekly content rollup: {str(e)}"

def get_youtube_transcript_summary(url: str) -> str:
    """Get a transcript for a YouTube video and summarize using Claude."""
    try:
        # Extract the video ID from the URL
        video_id = extract_youtube_id(url)
        if not video_id:
            return "Could not extract YouTube video ID from URL."
        
        # Get the transcript
        try:
            # Try to get English transcript first
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_transcript = " ".join([item['text'] for item in transcript_list])
        except Exception as e:
            # Try to get any available language if English fails
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Try manually created transcripts first
                try:
                    first_transcript = list(transcript_list._manually_created_transcripts.values())[0]
                except (IndexError, AttributeError):
                    # Then try auto-generated transcripts
                    try:
                        first_transcript = list(transcript_list._generated_transcripts.values())[0]
                    except (IndexError, AttributeError):
                        return "No transcript available for this video."
                
                transcript = first_transcript.fetch()
                full_transcript = " ".join([item['text'] for item in transcript])
            except Exception as e2:
                logging.error(f"Failed to get any transcript: {e2}")
                return "No transcript available for this video."
        
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
        
        if client is None:
            return "Claude API is not available. Please check your API key and SDK installation."
            
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.content[0].text
            
            # Add a header to clarify this is an AI summary
            return f"## AI-Generated Summary of YouTube Video\n\n{summary}"
        except Exception as e:
            logging.error(f"Error calling Claude API: {e}")
            return f"Error processing with Claude API: {str(e)}"
        
    except Exception as e:
        # Generic exception handling instead of specific types
        logging.error(f"Error extracting YouTube transcript: {e}")
        return f"Failed to extract and summarize YouTube transcript: {str(e)}"

def format_and_summarize_content(content, url, title):
    """Use Claude to format content and extract key points"""
    try:
        # Truncate if very long
        if len(content) > 30000:
            content = content[:30000] + "... [content truncated due to length]"
        
        prompt = f"""
        I need you to process this extracted web content from "{title}" ({url}).
        
        First, reformat the text to be more readable, fixing any extraction artifacts or formatting issues.
        
        Then, summarize the 3 most important points from this content in exactly 3 sentences.
        
        Format your response like this:
        
        ## Formatted Content
        
        [Properly formatted content here]
        
        ## Key Points
        
        1. [First key point in one sentence]
        2. [Second key point in one sentence]
        3. [Third key point in one sentence]
        
        Here's the extracted content:
        
        {content}
        """
        
        if client is None:
            return f"""
            {content}
            
            [Claude API is not available. Please check your API key and SDK installation.]
            """
            
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=35000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Error calling Claude API: {e}")
            return f"""
            {content}
            
            [Error processing with Claude API: {str(e)}]
            """
            
    except Exception as e:
        logging.error(f"Error processing content with Claude: {e}")
        # Return original content if Claude processing fails
        return f"""
        {content}
        
        [Claude processing failed: {str(e)}]
        """

def is_youtube_url(url: str) -> bool:
    """Check if a URL is a YouTube video."""
    youtube_regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(youtube_regex, url)
    return bool(match)

def extract_youtube_id(url: str) -> str:
    """Extract the YouTube video ID from a URL."""
    youtube_regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(youtube_regex, url)
    if match:
        return match.group(1)
    return None 