import json
import textract
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import codecs
import re
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import cohere
import numpy as np
import os
import shutil


cohere_key = "{YOUR_COHERE_API_KEY}"
openai_key = "{YOUR_OPENAI_API_KEY}"

co = cohere.Client(cohere_key)
openai.api_key = openai_key




os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"






class PDFEmbedder:
    def __init__(self, cohere_key, openai_key):
        self.cohere_client = cohere.Client(cohere_key)
        self.openai_embedding_model = openai.model('text-embedding-ada-002')
        self.openai_tokenizer_model = openai.model('text-davinci-003')

    def get_cohere_embedding(self, text):
        embedding = self.cohere_client.embed([text], input_type="search_document", model="embed-multilingual-v3.0").embeddings
        return np.asarray(embedding)

    def get_openai_embedding(self, text):
        if isinstance(text, list):
            response = self.openai_embedding_model.embed_many(text)
        else:
            response = self.openai_embedding_model.embed(text)
        response['model'] = 'text-embedding-ada-002'
        return response['embeddings']

    def get_token_count(self, text):
        return self.openai_tokenizer_model.token_count(text)




class PDFProcessor:
    def __init__(self, folder_name, chunk_size=1024, chunk_overlap=50):
        self.folder_name = folder_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = PDFEmbedder(cohere_key, openai_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = self.embedder.get_token_count,
        )

    def process_folder(self):
        for filename in os.listdir(self.folder_name):
            if filename.endswith(".pdf"):
                filepath = os.path.join(self.folder_name, filename)
                text = self._extract_text(filepath)
                self._save_text_to_file(filename, text)
                chunks = self._split_text(text)
                for chunk in chunks:
                    self._append_chunk_to_file(filename, chunk)

    def _extract_text(self, filepath):
        doc = textract.process(filepath)
        text = doc.decode('utf-8')
        return self._sanitize_text(text)

    def _sanitize_text(self, text):
        sanitized_text = re.sub(r'\W', ' ', text)
        sanitized_text = sanitized_text.lower()
        sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip()
        return sanitized_text

    def _save_text_to_file(self, filename, text):
        with codecs.open('./text/' + filename[:-4] + '.txt', 'w', 'utf-8') as f:
            f.write(text)

    def _append_chunk_to_file(self, filename, chunk):
        with codecs.open('./text/' + filename[:-4] + '.txt', 'a', 'utf-8') as f:
            f.write(chunk)

    def _split_text(self, text):
        return self.text_splitter.create_documents([text])

    def prompt_for_download(self):
        for filename in os.listdir(self.folder_name):
            if filename.endswith(".pdf"):
                download = input(f"Do you want to download {filename}? (yes/no): ")
                if download.lower() != "yes":
                    os.remove(os.path.join(self.folder_name, filename))
                    print(f"{filename} has been deleted.")
                else:
                    destination = "/tmp"
                    shutil.move(os.path.join(self.folder_name, filename), os.path.join(destination, filename))
                    print(f"{filename} has been moved to {destination}.")
                    print(f"To retrieve the file, use the following command (you may need to use sudo):")
                    print(f"sudo mv /tmp/{filename} ~/")




class DatabaseManager:
    def __init__(self, database, user, password, host, port):
        self.conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE EXTENSION IF NOT EXISTS pgvector')

    def create_table(self):
        self.cursor.execute('CREATE TABLE IF NOT EXISTS vss_articles (id serial primary key, headline_embedding vector(1536))')

    def insert_vectors(self, table_name, vectors):
        # Prepare the SQL query
        sql = f"INSERT INTO {table_name} (headline_embedding) VALUES %s"
        # Prepare the data
        data = [(vector.tolist(),) for vector in vectors]
        # Execute the SQL query
        self.cursor.executemany(sql, data)
        # Commit the changes
        self.conn.commit()
        
    def search_vectors(self, vector):
        self.cursor.execute("SELECT * FROM vss_articles ORDER BY headline_embedding <-> ? LIMIT 10;", (vector.tolist(),))
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()


    def perform_vector_similarity_query(self, table_name, embedding, match_threshold=0.78, match_count=10, min_content_length=50):
        # Convert the embedding to a list
        embedding_list = embedding.tolist()

        # Execute the SQL query
        self.cursor.execute(f"""
            SELECT * 
            FROM {table_name} 
            WHERE LENGTH(content) >= %s
            ORDER BY embedding <-> %s 
            LIMIT %s
        """, (min_content_length, embedding_list, match_count))

        # Fetch the results
        results = self.cursor.fetchall()

        # Filter the results based on the match_threshold
        filtered_results = [result for result in results if result['similarity'] >= match_threshold]

        return filtered_results  

db = DatabaseManager("my_database", "username", "password", "localhost", "5432")
db.create_table()
# insert and search vectors...
db.close()
###


class QueryProcessor:
    def __init__(self, embedder, database):
        self.embedder = embedder
        self.database = database
        self.temp_embeddings = {}
        self.sanitized_query = None

    def sanitize_query(self, query):
        self.sanitized_query = re.sub(r'\W', ' ', query)
        self.sanitized_query = self.sanitized_query.lower()
        self.sanitized_query = re.sub(r'\s+', ' ', self.sanitized_query).strip()

    def generate_embeddings(self):
        # Generate the OpenAI embedding
        openai_embedding = self.embedder.get_openai_embedding(self.sanitized_query)
        # Perform vector similarity query on the 'questions' table using the OpenAI embedding
        similar_questions = self.database.perform_vector_similarity_query('questions', openai_embedding)
        # Retrieve the corresponding answers for the similar questions
        similar_answers = [self.database.retrieve_answer(question) for question in similar_questions]
        # Tokenize the answers and extract key concepts
        tokenized_answers = [self.embedder.get_token_count(answer) for answer in similar_answers]
        # Synthesize the tokenized answers into a context of up to 8000 tokens
        context = self.synthesize_context(tokenized_answers)
        # Generate the Cohere embedding for the context
        cohere_embedding = self.embedder.get_cohere_embedding(context)
        return openai_embedding, cohere_embedding

    def store_embeddings(self, openai_embedding, cohere_embedding):
        # Store the OpenAI embedding in the 'questions' table
        self.database.store_embedding('questions', self.sanitized_query, openai_embedding)
        # Perform vector similarity query on the 'texts' table using the Cohere embedding
        similar_texts = self.database.perform_vector_similarity_query('texts', cohere_embedding)
        # Delete the Cohere embedding from memory
        del self.temp_embeddings[self.sanitized_query]
        return similar_texts

class UserInteraction:
    def __init__(self, query_processor):
        self.query_processor = query_processor
        self.chat_history = []

    def get_user_query(self):
        query = input("Enter a query (type 'exit' to quit): ")
        return query

    def interact(self):
        while True:
            raw_query = self.get_user_query()
            if raw_query.lower() == "exit":
                break
            self.query_processor.sanitize_query(raw_query)
            self.query_processor.generate_and_store_embeddings()
            response = self.generate_response()
            self.chat_history.append((self.query_processor.sanitized_query, response))
            print(response)


        # Generate a response based on the most similar texts
        response = self.generate_response_based_on_similar_texts(similar_questions, similar_answers)
        
        return response

        # Embed the response using the OpenAI API
        response_embedding = self.query_processor.openai_client.embed(response)

        # Build the prompt
        prompt = self.build_prompt(self.query_processor.sanitized_query, similar_texts)
        

        # Make a text completion request via the OpenAI API
        event_stream = self.text_completion(prompt)
        

        # Process the event stream and generate the final response
        response = self.process_event_stream(event_stream)

        
        # Store the OpenAI embedding in the 'questions' table
        self.database.store_embedding('questions', self.sanitized_query, openai_embedding)

    def generate_context_and_prompt(self, query, similar_questions, similar_answers, key_concepts):
        # Start with the user's query
        context = f"The user has asked the following question: {query}\n\n"

        # Add similar questions and answers
        context += "Here are some similar questions that have been asked before, along with their answers:\n"
        for question, answer in zip(similar_questions, similar_answers):
            context += f"Question: {question}\nAnswer: {answer}\n\n"

        # Add key concepts
        context += "Based on these questions and answers, the following key concepts have been identified:\n"
        for concept in key_concepts:
            context += f"{concept}\n"

        # Add a conclusion that summarizes the key points
        context += "\nIn response to the user's query, consider the following points:\n"

        # Ensure the context does not exceed the maximum token limit
        if self.embedder.get_token_count(context) > 12000:
            context = self.summarize_context(context, 12000)

        # Generate a prompt from the context
        prompt = self.summarize_context(context, 1000)

        # Use Cohere to find semantic associations in the text database based on the context
        cohere_embedding = self.embedder.get_cohere_embedding(context)
        similar_texts = self.database.perform_vector_similarity_query('texts', cohere_embedding)

        # Select the most relevant texts up to the token limit
        selected_texts = self.select_most_relevant_texts(similar_texts, 1200)

        # Append the selected texts to the final prompt for GPT-4
        final_prompt = prompt + "\n\nSimilar texts:\n" + "\n".join(selected_texts)

        return context, prompt, final_prompt


        # Use Cohere to find semantic associations in the text database based on the context
        cohere_embedding = self.embedder.get_cohere_embedding(context)
        similar_texts = self.database.perform_vector_similarity_query('texts', cohere_embedding)

        def summarize_context(self, context, max_tokens):
            # Use the gpt-3.5-turbo-16k model to summarize the context
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-16k",
                prompt=context,
                max_tokens=max_tokens,
                temperature=0.5,
            )
        return response.choices[0].text.strip()
        
        





class FinalAnalysis:
    def prompt_gpt(self):
        return "You are a scholar that possesses knowledge of philosophy at a PhD level and excel at explaining while at the same time being concise, clear, and demonstrating a masterful use of logic and philosophical terminology"

    def text_completion(self, prompt, max_tokens):
        # Send the prompt to the OpenAI API to generate a text completion
        final_response = openai.Completion.create(
            engine="gpt-4",
            prompt=final_prompt,
            max_tokens=max_tokens,
            temperature=0,
            stream=True,
        )
        return final_response

# Embed the final response using the OpenAI API
final_response_embedding = self.query_processor.embedder.get_openai_embedding(final_response)
# Store the final response and its embedding in the 'answers' table
self.query_processor.database.store_response('answers', final_response, final_response_embedding)



    def process_event_stream(self, event_stream):
        final_response = ""
        try:
            # Parse the event stream and concatenate the text completions
            for event in event_stream:
                if 'choices' in event:
                    final_response += event['choices'][0]['text']
        except Exception as e:
            print(f"An error occurred: {e}")
        return final_response





    # Existing code...
    chat_history.append((query, response))
    print(response)


# Main Execution Code

def main():
    conn, cursor = connect_to_database()
    create_match_documents_function(cursor)
    all_chunks = process_pdf_folder("./pdf", "./text")
    embedder = PDFEmbedder()
    for i, chunk in enumerate(all_chunks):
        embedding = embedder.create_embedding(chunk)
        conn.execute('INSERT INTO vss_articles(headline_embedding) VALUES (?)', (embedding.tolist(),))
        interact_with_user()
    conn.commit()
    conn.close()
    

if __name__ == "__main__":
    main()


print("Exited!!!")
