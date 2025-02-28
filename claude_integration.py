import os
import anthropic
from typing import List
from datetime import datetime
from dotenv import load_dotenv
import logging

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