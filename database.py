"""
Database module for storing and retrieving content entries.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import sqlalchemy
import logging
import contextlib

# Import from config
from config import DATABASE_URL, DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_TIMEOUT, DB_POOL_RECYCLE, LOG_LEVEL

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("database")

# Configure engine with explicit pool settings
engine = create_engine(
    DATABASE_URL,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT,
    pool_recycle=DB_POOL_RECYCLE,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Entry(Base):
    __tablename__ = "entries"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, index=True)
    content = Column(Text)
    thoughts = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    embedding = relationship("Embedding", back_populates="entry", uselist=False, cascade="all, delete-orphan")

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    entry_id = Column(Integer, ForeignKey("entries.id", ondelete="CASCADE"), unique=True)
    vector = Column(Vector(384))  # Matches the dimensions of the sentence transformer model
    
    entry = relationship("Entry", back_populates="embedding")

def create_tables():
    """Create database tables and vector extension"""
    logger.info("Creating database tables and vector extension")
    global engine
    
    with engine.connect() as conn:
        # Begin a transaction for vector extension
        trans = conn.begin()
        try:
            # Create vector extension
            conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
            # Commit explicitly
            trans.commit()
            logger.info("Vector extension created successfully")
        except Exception as e:
            # Roll back on error
            trans.rollback()
            logger.error(f"Error creating vector extension: {e}")
            # Continue anyway - tables might still work
        
        # Start a new transaction for table creation
        trans = conn.begin()
        try:
            # Create tables
            Base.metadata.create_all(bind=engine)
            trans.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            trans.rollback()
            logger.error(f"Error creating tables: {e}")
            raise

def get_db():
    """
    Get database session with proper exception handling to ensure
    connections are returned to the pool.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def reset_db_connection():
    """Reset database connection pool when issues are detected"""
    logger.info("Attempting to reset database connection pool")
    try:
        # Dispose of the current engine's connections
        engine.dispose()
        logger.info("Engine connections disposed")
        
        # Test a new connection
        with contextlib.closing(engine.connect()) as conn:
            # Test if vector operations are working
            try:
                # Simple query to test connection
                conn.execute(sqlalchemy.text("SELECT 1"))
                # Test vector extension
                conn.execute(sqlalchemy.text("SELECT '[1,2,3]'::vector"))
                logger.info("Database connection reset successful")
                return True
            except Exception as e:
                logger.error(f"Vector operations not working: {e}")
                return False
                
    except Exception as e:
        logger.error(f"Database connection reset failed: {e}")
        return False

# Add this function to properly handle vector conversion
def convert_to_pg_vector(embedding):
    """Convert embedding to proper pgvector format"""
    if embedding is None:
        return None
    
    # If embedding is numpy array, convert to list
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    
    # Ensure all values are floats
    try:
        embedding = [float(val) for val in embedding]
    except (TypeError, ValueError) as e:
        logger.error(f"Error converting embedding values to float: {e}")
        return None
    
    return embedding

# Update generate_and_store_embedding function
def generate_and_store_embedding(entry_id: int, text: str):
    try:
        # Create a new session for background task
        db = SessionLocal()
        # Import here to avoid circular imports
        from embeddings import get_embedding
        embedding_vector = get_embedding(text)
        # Convert to proper pgvector format
        pg_vector = convert_to_pg_vector(embedding_vector)
        db_embedding = Embedding(entry_id=entry_id, vector=pg_vector)
        db.add(db_embedding)
        db.commit()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
    finally:
        db.close() 