from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv
from sqlalchemy.pool import QueuePool
import sqlalchemy
import logging

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Configure engine with explicit pool settings
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
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
    
    embedding = relationship("Embedding", back_populates="entry", uselist=False, cascade="all, delete")

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    entry_id = Column(Integer, ForeignKey("entries.id", ondelete="CASCADE"))
    embedding = Column(Vector(384))  # Dimension for all-MiniLM-L6-v2
    
    entry = relationship("Entry", back_populates="embedding")

# Create tables
def create_tables():
    global engine
    with engine.connect() as conn:
        # Begin a transaction
        trans = conn.begin()
        try:
            # Create vector extension
            conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
            # Commit explicitly
            trans.commit()
            logging.info("Vector extension created successfully")
        except Exception as e:
            # Roll back on error
            trans.rollback()
            logging.error(f"Error creating vector extension: {e}")
            # Continue anyway - tables might still work
        
        # Start a new transaction for table creation
        trans = conn.begin()
        try:
            # Create tables
            Base.metadata.create_all(bind=engine)
            trans.commit()
            logging.info("Database tables created successfully")
        except Exception as e:
            trans.rollback()
            logging.error(f"Error creating tables: {e}")
            raise

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def reset_db_connection():
    """Reset database connection and verify pgvector is installed"""
    global engine
    with engine.connect() as conn:
        try:
            # Check if vector extension exists
            result = conn.execute(sqlalchemy.text(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'"))
            extension_exists = result.scalar() is not None
            
            if not extension_exists:
                # Try to create if not exists
                trans = conn.begin()
                try:
                    conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    trans.commit()
                    logging.info("Vector extension created during reset")
                except Exception as e:
                    trans.rollback()
                    logging.error(f"Failed to create vector extension: {e}")
            
            # Test a simple vector operation to verify functioning
            try:
                conn.execute(sqlalchemy.text("SELECT '{1,2,3}'::vector"))
                logging.info("Vector operations working correctly")
                return True
            except Exception as e:
                logging.error(f"Vector operations not working: {e}")
                return False
                
        except Exception as e:
            logging.error(f"Database connection reset failed: {e}")
            return False 