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
client = None
USING_NEW_CLIENT = False

# Better detection of Anthropic client version
try:
    # Try to initialize the newer client format
    temp_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Test if it has the messages attribute
    if hasattr(temp_client, 'messages'):
        client = temp_client
        USING_NEW_CLIENT = True
        logging.info("Using new Anthropic client with messages API")
    else:
        # It has the new class name but old API
        client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
        USING_NEW_CLIENT = False
        logging.info("Using older Anthropic client API")
except Exception as e:
    # Fall back to older client format
    logging.warning(f"Error initializing new Anthropic client: {e}")
    try:
        client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
        USING_NEW_CLIENT = False
        logging.info("Fallback to older Anthropic client")
    except Exception as e2:
        logging.error(f"Failed to initialize any Anthropic client: {e2}")

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
        
        # Call Claude with appropriate API version
        if USING_NEW_CLIENT:
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            response = client.completion(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
                max_tokens_to_sample=1000,
                model="claude-3-sonnet-20240229"
            )
            return response.completion
            
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
        
        Please provide:
        1. A summary of key themes and topics across these entries
        2. Interesting connections or patterns between different pieces of content
        3. Questions these entries raise that might be worth exploring further
        
        Focus on being insightful rather than just summarizing each entry individually.
        """
        
        if USING_NEW_CLIENT and hasattr(client, 'messages'):
            # Double-check the client has messages attribute
            response = client.messages.create(
                model="claude-3-sonnet-20240229",  # Use a known stable model
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            # Use the older client format
            response = client.completion(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
                max_tokens_to_sample=1500,
                model="claude-3-sonnet-20240229"
            )
            return response.completion
            
    except Exception as e:
        logging.error(f"Error generating weekly rollup: {e}")
        return f"I encountered an error while generating your weekly content rollup: {str(e)}"

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
        
        # Get Claude's summary
        if USING_NEW_CLIENT and hasattr(client, 'messages'):
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.content[0].text
        else:
            response = client.completion(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
                max_tokens_to_sample=500,
                model="claude-3-sonnet-20240229"
            )
            summary = response.completion
        
        # Add a header to clarify this is an AI summary
        return f"## AI-Generated Summary of YouTube Video\n\n{summary}"
        
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
        
        # Call Claude with appropriate API version
        if USING_NEW_CLIENT:
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=35000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            response = client.completion(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
                max_tokens_to_sample=35000,
                model="claude-3-sonnet-20240229"
            )
            return response.completion
            
    except Exception as e:
        logging.error(f"Error processing content with Claude: {e}")
        # Return original content if Claude processing fails
        return f"""
        {content}
        
        [Claude processing failed: {str(e)}]
        """ 