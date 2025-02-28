from fastapi import FastAPI, Depends, HTTPException, Security, status, Request, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
import os
import logging
import time
from dotenv import load_dotenv
import sqlalchemy
import requests
from bs4 import BeautifulSoup
from newspaper3k import Article

from database import get_db, Entry, Embedding, SessionLocal
from embeddings import get_embedding, search_entries
from claude_integration import get_search_response, get_weekly_rollup_response, get_youtube_transcript_summary, is_youtube_url

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api")

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI(
    title="Content API",
    description="API for storing and retrieving content with semantic search and Claude integration. "
                "Automatically extracts and summarizes YouTube transcripts when a YouTube URL is provided.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication dependency
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key"
    )

# Pydantic models
class EntryCreate(BaseModel):
    url: HttpUrl
    content: str
    thoughts: str

class EntryResponse(BaseModel):
    id: int
    url: str
    content: str
    thoughts: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class SearchEntryResponse(EntryResponse):
    similarity: float

class SearchResponse(BaseModel):
    claude_response: str
    entries: List[SearchEntryResponse]

class WeeklyRollupResponse(BaseModel):
    claude_response: str
    start_date: datetime
    end_date: datetime
    entries: List[EntryResponse]

# Rate limiting implementation
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.request_history = {}
        
    async def __call__(self, request: Request):
        client_ip = request.client.host
        now = time.time()
        
        # Clean up old requests
        self.request_history = {ip: times for ip, times in self.request_history.items() 
                               if times and times[-1] > now - 60}
        
        # Get or create request history for this client
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
        
        # Add current request timestamp
        self.request_history[client_ip].append(now)
        
        # Check rate limit
        if len(self.request_history[client_ip]) > self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later."
            )
        
        return True

# Create limiter instance
rate_limiter = RateLimiter(requests_per_minute=60)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Error handling
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Value error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

# Routes
@app.post("/entries", response_model=EntryResponse)
def create_entry(
    entry: EntryCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
    _: bool = Depends(rate_limiter)
):
    """
    Store a new content entry with URL, content text, and thoughts.
    Also generates and stores a vector embedding for semantic search.
    """
    url = str(entry.url)
    content = entry.content
    
    # Check if this is a YouTube URL and content is empty or user wants auto-transcription
    if (not content or "[auto-transcribe]" in content.lower()) and is_youtube_url(url):
        try:
            # Get transcript and summarize
            logger.info(f"Extracting and summarizing YouTube transcript for {url}")
            youtube_summary = get_youtube_transcript_summary(url)
            
            # Use the summary as content if we got one
            if youtube_summary and not youtube_summary.startswith("Failed"):
                content = youtube_summary
                logger.info(f"Successfully extracted YouTube transcript for {url}")
        except Exception as e:
            logger.error(f"Error processing YouTube video: {e}")
            # Continue with user-provided content if extraction fails
    
    # Create entry
    db_entry = Entry(url=url, content=content, thoughts=entry.thoughts)
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    
    # Move embedding generation to background task
    background_tasks.add_task(
        generate_and_store_embedding, 
        db_entry.id, 
        content + " " + entry.thoughts
    )
    
    return db_entry

@app.get("/search", response_model=SearchResponse)
def search_content(
    query: str,
    limit: int = 5,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
    _: bool = Depends(rate_limiter)
):
    """
    Search for entries using semantic similarity.
    Returns Claude's interpretation of the results.
    """
    # Get semantically similar entries
    results = search_entries(db, query, limit)
    
    # Get Claude's response about the search results
    claude_response = get_search_response(query, results)
    
    return {
        "claude_response": claude_response,
        "entries": results
    }

@app.get("/weekly-rollup", response_model=WeeklyRollupResponse)
def get_weekly_rollup(
    date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
    _: bool = Depends(rate_limiter)
):
    """
    Get a weekly rollup of entries from the past 7 days.
    Returns Claude's summary of the week's content.
    """
    # Use provided date or current date
    end_date = date or datetime.now()
    
    # Set end_date to end of day
    end_date = end_date.replace(hour=23, minute=59, second=59)
    
    # Calculate start date (7 days before)
    start_date = end_date - timedelta(days=7)
    start_date = start_date.replace(hour=0, minute=0, second=0)
    
    # Query entries from the last 7 days
    entries = db.query(Entry).filter(
        Entry.created_at >= start_date,
        Entry.created_at <= end_date
    ).order_by(Entry.created_at.desc()).all()
    
    # Get Claude's weekly rollup response
    claude_response = get_weekly_rollup_response(entries, start_date, end_date)
    
    return {
        "claude_response": claude_response,
        "start_date": start_date,
        "end_date": end_date,
        "entries": entries
    }

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

def generate_and_store_embedding(entry_id: int, text: str):
    try:
        # Create a new session for background task
        db = SessionLocal()
        embedding_vector = get_embedding(text)
        db_embedding = Embedding(entry_id=entry_id, embedding=embedding_vector)
        db.add(db_embedding)
        db.commit()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
    finally:
        db.close()

@app.on_event("startup")
async def startup_db_client():
    try:
        from database import engine
        # Test the connection
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        logger.info("Successfully connected to the database")
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        # Don't raise here - let the app start and fail gracefully

def extract_url_content(url: str) -> dict:
    """
    Extract content from a URL and return a structured dictionary.
    Works for general websites, news articles, and falls back to YouTube extraction when applicable.
    """
    try:
        # First check if it's a YouTube URL and handle accordingly
        if is_youtube_url(url):
            summary = get_youtube_transcript_summary(url)
            return {
                "content": summary,
                "title": "YouTube Video Summary",
                "success": True
            }
        
        # For general websites, use newspaper3k for extraction
        article = Article(url)
        article.download()
        article.parse()
        
        # If it's an article, it might have a title and text
        if article.text and len(article.text.strip()) > 100:
            # Get title if available, otherwise use URL
            title = article.title if article.title else url
            return {
                "content": article.text,
                "title": title,
                "success": True
            }
            
        # Fallback to basic HTML scraping if newspaper extraction isn't satisfactory
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the main text content, prioritizing article content
        content = ""
        
        # Try to find the main content
        for tag in ['article', 'main', 'div[role="main"]', '.content', '#content']:
            main_content = soup.select(tag)
            if main_content:
                content = main_content[0].get_text(separator='\n', strip=True)
                break
        
        # If no specific content container found, get the body content
        if not content:
            content = soup.body.get_text(separator='\n', strip=True)
            
        # Get title
        title = soup.title.string if soup.title else url
            
        # Use Claude to summarize if content is too long
        if len(content) > 5000:
            summary_prompt = f"""
            Please summarize the following content from {url} in a concise way, 
            preserving the key information and main points:
            
            {content[:10000]}  # Truncate if extremely long
            """
            
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            
            content = f"## AI-Generated Summary from {title}\n\n{response.content[0].text}"
            
        return {
            "content": content,
            "title": title,
            "success": True
        }
    
    except Exception as e:
        logging.error(f"Error extracting content from URL {url}: {e}")
        return {
            "content": f"Failed to extract content from {url}: {str(e)}",
            "title": url,
            "success": False
        }

@app.post("/extract", response_model=EntryResponse)
def create_from_url(
    url: str,
    thoughts: str = "",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
    _: bool = Depends(rate_limiter)
):
    """
    Create a new entry by automatically extracting content from a URL.
    Only requires the URL and optional thoughts.
    """
    # Extract content from URL
    extraction = extract_url_content(url)
    
    # Create entry with extracted content
    db_entry = Entry(
        url=url, 
        content=extraction["content"],
        thoughts=thoughts if thoughts else f"Auto-extracted from {extraction['title']}"
    )
    
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    
    # Generate embedding in background
    background_tasks.add_task(
        generate_and_store_embedding, 
        db_entry.id, 
        extraction["content"] + " " + db_entry.thoughts
    )
    
    return db_entry

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)