import logging
try:
    import fix_imports  # Apply import workaround
except ImportError:
    # Create the fix on the fly if the file doesn't exist
    import sys
    import types
    
    # Create dummy lxml.html.clean module
    dummy_module = types.ModuleType("lxml.html.clean")
    
    # Add a dummy Cleaner class
    class Cleaner:
        def __init__(self, **kwargs):
            pass
        def clean_html(self, html):
            return html
    
    dummy_module.Cleaner = Cleaner
    
    # Insert the dummy module
    sys.modules["lxml.html.clean"] = dummy_module
    logging.warning("Created inline lxml.html.clean workaround")

from fastapi import FastAPI, Depends, HTTPException, Security, status, Request, BackgroundTasks, Body
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
import os
import time
from dotenv import load_dotenv
import sqlalchemy
import requests
from bs4 import BeautifulSoup
import traceback

from database import get_db, Entry, Embedding, SessionLocal, reset_db_connection
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

# Add this new model for extract endpoint
class ExtractRequest(BaseModel):
    url: str
    thoughts: str = ""

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
    try:
        # Get semantically similar entries
        results = search_entries(db, query, limit)
        
        # Get Claude's response about the search results
        claude_response = get_search_response(query, results)
        
        return {
            "claude_response": claude_response,
            "entries": results
        }
    except Exception as e:
        # Log the error and return a fallback response
        logger.error(f"Search error: {e}")
        
        # Always roll back on error to prevent transaction issues
        db.rollback()
        
        # Get recent entries as fallback
        fallback_entries = db.query(Entry).order_by(Entry.created_at.desc()).limit(limit).all()
        fallback_results = [
            {
                "id": entry.id,
                "url": entry.url,
                "content": entry.content,
                "thoughts": entry.thoughts,
                "created_at": entry.created_at,
                "similarity": 0.0
            } for entry in fallback_entries
        ]
        
        return {
            "claude_response": f"I encountered an error while searching: {str(e)}. Here are your most recent entries instead.",
            "entries": fallback_results
        }

@app.get("/weekly-rollup", response_model=WeeklyRollupResponse)
def get_weekly_rollup(
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
    _: bool = Depends(rate_limiter)
):
    """
    Return a weekly rollup of content with insights from Claude
    """
    # Get current time and 7 days ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Query entries from the past week
    try:
        # Get entries from the past week
        entries = db.query(Entry).filter(
            Entry.created_at >= start_date,
            Entry.created_at <= end_date
        ).all()
        
        # If no entries, return a message
        if not entries:
            return WeeklyRollupResponse(
                claude_response="No entries found for the past week.",
                start_date=start_date,
                end_date=end_date,
                entries=[]
            )
        
        # Convert entries to list of dicts for Claude
        entries_for_claude = [
            {
                "id": entry.id,
                "url": entry.url,
                "content": entry.content,
                "thoughts": entry.thoughts,
                "created_at": entry.created_at
            }
            for entry in entries
        ]
        
        # Get Claude's insights
        claude_response = get_weekly_rollup_response(entries_for_claude)
        
        return WeeklyRollupResponse(
            claude_response=claude_response,
            start_date=start_date,
            end_date=end_date,
            entries=entries
        )
    
    except Exception as e:
        logging.error(f"Error in weekly rollup: {e}")
        return WeeklyRollupResponse(
            claude_response=f"Error generating weekly rollup: {str(e)}",
            start_date=start_date,
            end_date=end_date,
            entries=[]
        )

@app.get("/health")
def health_check():
    """Check if the API is running and connections are working"""
    try:
        # Test database connection
        with SessionLocal() as db:
            db.execute(sqlalchemy.text("SELECT 1"))
        
        # Test vector functionality
        vector_working = reset_db_connection()
        
        return {
            "status": "healthy",
            "database": "connected",
            "vector_extension": "working" if vector_working else "not working"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

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

def extract_with_bs4(url):
    """Extract content without using newspaper3k"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    response = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get title
    title = soup.title.string if soup.title else url
    
    # Extract content
    content = ""
    for tag in ['article', 'main', '.content', '#content', 'div.post', 'div.entry']:
        elements = soup.select(tag)
        if elements:
            content = elements[0].get_text(separator='\n', strip=True)
            break
    
    # Fallback to body if no specific content container found
    if not content and soup.body:
        content = soup.body.get_text(separator='\n', strip=True)
    
    return {
        "content": content,
        "title": title,
        "success": True if content else False
    }

@app.post("/extract", response_model=EntryResponse)
def create_from_url(
    url: str = None,
    thoughts: str = "",
    request_data: ExtractRequest = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
    _: bool = Depends(rate_limiter)
):
    """
    Create a new entry by automatically extracting content from a URL.
    Accepts parameters either as query parameters or in the request body.
    """
    # Use request_data if provided, otherwise use query parameters
    if request_data:
        url = request_data.url
        thoughts = request_data.thoughts or thoughts
    
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    # Extract content from URL
    extraction = extract_with_bs4(url)
    
    # Create entry with extracted content or a placeholder if extraction failed
    content = extraction["content"]
    auto_thoughts = f"Auto-extracted from {extraction['title']}"
    
    if not extraction["success"]:
        content = "[Content extraction failed]"
        auto_thoughts = f"Failed to extract content from {url}"
    
    db_entry = Entry(
        url=url, 
        content=content,
        thoughts=thoughts if thoughts else auto_thoughts
    )
    
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    
    # Generate embedding in background (even for failed extractions)
    # This will use whatever content we have plus thoughts for search
    background_tasks.add_task(
        generate_and_store_embedding, 
        db_entry.id, 
        content + " " + db_entry.thoughts
    )
    
    return db_entry

@app.get("/test-extract")
def test_extract(
    url: str,
    api_key: APIKey = Depends(get_api_key),
):
    """
    Test URL extraction without saving to database.
    Returns detailed diagnostics about the extraction process.
    """
    start_time = time.time()
    
    try:
        # Skip newspaper3k test since we're removing it
        newspaper_result = {
            "success": False, 
            "error": "Newspaper3k disabled due to compatibility issues", 
            "title": None, 
            "content_length": 0
        }
        
        # Try with BeautifulSoup (keep this part)
        bs_result = {"success": False, "error": None, "title": None, "content_length": 0}
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            content = ""
            title = soup.title.string if soup.title else url
            
            for tag in ['article', 'main', 'div[role="main"]', '.content', '#content']:
                main_content = soup.select(tag)
                if main_content:
                    content = main_content[0].get_text(separator='\n', strip=True)
                    break
            
            if not content and soup.body:
                content = soup.body.get_text(separator='\n', strip=True)
                
            bs_result = {
                "success": len(content) > 100,
                "title": title,
                "content_length": len(content),
                "error": None if len(content) > 100 else "Insufficient content extracted"
            }
        except Exception as e:
            bs_result["error"] = str(e)
            
        # Complete result using our custom function
        extraction = extract_with_bs4(url)
        
        # Check if it's a YouTube URL
        if is_youtube_url(url):
            try:
                youtube_result = {
                    "success": True,
                    "content": get_youtube_transcript_summary(url),
                    "title": "YouTube Video"
                }
                extraction = youtube_result
            except Exception as e:
                logger.error(f"YouTube extraction failed: {e}")
        
        # Return results
        return {
            "url": url,
            "newspaper3k_test": newspaper_result,  # Just return the disabled message
            "beautifulsoup_test": bs_result,
            "combined_result": extraction,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "is_youtube": is_youtube_url(url)
        }
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)