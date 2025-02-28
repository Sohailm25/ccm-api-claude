from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import Entry, Embedding
from typing import List, Dict, Any
import logging

# Load a smaller model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 61MB vs 90MB for all-MiniLM-L6-v2

def get_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text"""
    return model.encode(text).tolist()

def search_entries(db: Session, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search using vector similarity"""
    query_embedding = get_embedding(query)
    
    try:
        # Try pgvector's <-> operator first
        result = db.execute(
            text("""
            SELECT 
                e.id, 
                e.url, 
                e.content, 
                e.thoughts, 
                e.created_at,
                1 - (emb.embedding <-> :query_embedding::vector) AS similarity
            FROM 
                entries e
            JOIN 
                embeddings emb ON e.id = emb.entry_id
            ORDER BY 
                similarity DESC
            LIMIT :limit
            """),
            {
                "query_embedding": query_embedding,
                "limit": limit
            }
        )
        
        entries = []
        for row in result:
            entries.append({
                "id": row.id,
                "url": row.url,
                "content": row.content,
                "thoughts": row.thoughts,
                "created_at": row.created_at,
                "similarity": float(row.similarity)
            })
        
        return entries
        
    except Exception as e:
        # Fallback to a simple query without vector search
        logging.error(f"Vector search failed: {e}")
        
        # Just return the most recent entries as fallback
        result = db.query(Entry).order_by(Entry.created_at.desc()).limit(limit).all()
        
        entries = []
        for entry in result:
            entries.append({
                "id": entry.id,
                "url": entry.url,
                "content": entry.content,
                "thoughts": entry.thoughts,
                "created_at": entry.created_at,
                "similarity": 0.0  # No similarity score in fallback mode
            })
        
        return entries 