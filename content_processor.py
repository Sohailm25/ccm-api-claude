def format_and_summarize_content(content, url=None):
    """
    Format and summarize the extracted content from a URL.
    
    Parameters:
    content (str): The content to summarize
    url (str, optional): The source URL
    
    Returns:
    dict: Formatted and summarized content
    """
    if not content:
        return {"content": "", "summary": "", "source": url}
    
    # Basic formatting - trim whitespace, normalize line breaks, etc.
    formatted_content = " ".join(content.split())
    
    # Return the formatted content
    # (You'd implement actual summarization here if needed)
    return {
        "content": formatted_content,
        "summary": formatted_content[:200] + ("..." if len(formatted_content) > 200 else ""),
        "source": url
    } 