-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the necessary index for faster vector searches
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops); 