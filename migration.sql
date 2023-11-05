-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the schema
CREATE SCHEMA IF NOT EXISTS my_schema;

-- Set the search_path to include the new schema
SET search_path TO my_schema, public;


## Texts table will be embedded by openai
-- Create the texts table
CREATE TABLE IF NOT EXISTS texts (
    id bigserial PRIMARY KEY,
    text text,
    embedding vector(1536)
);

## Questions and answers (answers will include responses) are embedded by cohere

-- Create the questions table
CREATE TABLE IF NOT EXISTS questions (
    id bigserial PRIMARY KEY,
    question text,
    embedding vector(1024)
);

-- Create the answers table
CREATE TABLE IF NOT EXISTS answers (
    id bigserial PRIMARY KEY,
    question_id bigint REFERENCES questions(id),
    text_id bigint REFERENCES texts(id),
    answer text,
    embedding vector(1024)
);

-- Create the responses table
CREATE TABLE IF NOT EXISTS responses (
    id bigserial PRIMARY KEY,
    response text,
    embedding vector(1024)
);

-- Create the HNSW indexes
CREATE INDEX IF NOT EXISTS texts_embedding_idx ON texts USING hnsw (embedding vector_l2_ops);
CREATE INDEX IF NOT EXISTS questions_embedding_idx ON questions USING hnsw (embedding vector_l2_ops);
CREATE INDEX IF NOT EXISTS answers_embedding_idx ON answers USING hnsw (embedding vector_l2_ops);
CREATE INDEX IF NOT EXISTS responses_embedding_idx ON responses USING hnsw (embedding vector_l2_ops);

-- Grant permissions to a specific user
GRANT USAGE ON SCHEMA my_schema TO myuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE texts TO myuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE questions TO myuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE answers TO myuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE responses TO myuser;
