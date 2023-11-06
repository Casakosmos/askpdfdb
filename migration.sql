-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the schema
CREATE SCHEMA IF NOT EXISTS my_schema;

-- Set the search_path to include the new schema
SET search_path TO my_schema, public;

-- extracted_texts table will be embedded by Cohere
-- Create the extracted_texts table
CREATE TABLE philosophical_works (
    id SERIAL PRIMARY KEY,
    extracted_texts TEXT,
    embedding VECTOR(1024),
    author TEXT,
    work TEXT,
    date_written DATE
);


-- Create the HNSW indexes
CREATE INDEX IF NOT EXISTS extracted_texts_headline_embedding_idx ON extracted_texts USING gist (headline_embedding gist_l2_ops);

-- Grant permissions to a specific user
GRANT USAGE ON SCHEMA my_schema TO myuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE extracted_texts TO myuser;


-- Questions and answers (answers will include responses) are embedded by OpenAI

-- Create the questions table
CREATE TABLE IF NOT EXISTS questions (
    id bigserial PRIMARY KEY,
    question text,
    embedding vector(1536) -- OpenAI embeddings go here
);

-- Create the answers table
CREATE TABLE IF NOT EXISTS answers (
    id bigserial PRIMARY KEY,
    question_id bigint REFERENCES questions(id),
    text_id bigint REFERENCES texts(id),
    answer text,
    embedding vector(1536) -- OpenAI embeddings go here
);


-- Create the HNSW indexes
CREATE INDEX IF NOT EXISTS extracted_texts_embedding_idx ON texts USING hnsw (embedding vector_l2_ops);
CREATE INDEX IF NOT EXISTS questions_embedding_idx ON questions USING hnsw (embedding vector_l2_ops);
CREATE INDEX IF NOT EXISTS answers_embedding_idx ON answers USING hnsw (embedding vector_l2_ops);

-- Grant permissions to a specific user
GRANT USAGE ON SCHEMA my_schema TO myuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE extracted_texts TO myuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE questions TO myuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE answers TO myuser;

