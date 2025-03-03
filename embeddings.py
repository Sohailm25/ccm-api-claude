from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import Entry, Embedding, convert_to_pg_vector
from typing import List, Dict, Any
import logging

# Load a smaller model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 61MB vs 90MB for all-MiniLM-L6-v2

def get_embedding(text):
    """Generate a vector embedding for the given text"""
    try:
        # Old API might be using: embedding_vector = model.encode(text, embedding=...)
        embedding_vector = model.encode(text)
        
        # Convert to proper format for pgvector
        # Ensure the embedding is properly formatted for pgvector
        if isinstance(embedding_vector, list):
            return embedding_vector
        return embedding_vector.tolist()
        
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None

def search_entries(db: Session, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search using vector similarity"""
    query_embedding = get_embedding(query)
    
    try:
        # Use the convert_to_pg_vector function for consistency
        query_embedding = convert_to_pg_vector(query_embedding)
        if query_embedding is None:
            raise ValueError("Failed to generate valid embedding for search query")
        
        # Convert embedding to string format PostgreSQL understands
        embedding_str = str(query_embedding).replace('[', '{').replace(']', '}')
        
        # Use direct string interpolation for vector (safe here because we control the input)
        result = db.execute(
            text(f"""
            SELECT 
                e.id, 
                e.url, 
                e.content, 
                e.thoughts, 
                e.created_at,
                1 - (emb.embedding <-> '{embedding_str}'::vector) AS similarity
            FROM 
                entries e
            JOIN 
                embeddings emb ON e.id = emb.entry_id
            ORDER BY 
                similarity DESC
            LIMIT :limit
            """),
            {
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
        
        # Roll back transaction to prevent 'transaction aborted' errors
        db.rollback()
        
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