"""
Centralized configuration for the API.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"
PORT = int(os.getenv("PORT", "8000"))

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10")) 
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # 30 minutes

# Claude AI Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"

# Rate Limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# Content Extraction
EXTRACTION_TIMEOUT = int(os.getenv("EXTRACTION_TIMEOUT", "15"))  # seconds
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "30000"))  # characters

# Setup logger configuration parameters
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") 