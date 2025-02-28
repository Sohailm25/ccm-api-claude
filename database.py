from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv
from sqlalchemy.pool import QueuePool
import sqlalchemy

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
    # Create the vector extension first
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    
    # Then create all tables
    Base.metadata.create_all(bind=engine)

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 