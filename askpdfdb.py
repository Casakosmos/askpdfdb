import sqlite3
import json
import textract
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import codecs
import re

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"


def get_embeddings():
    return openai.model('text-embedding-ada-002')

class PDFEmbedder:
    def create_embedding(self, content):
        embeddings = get_embeddings()
        return embeddings.embed_query(content)

    def embedding(text, **kw):
        model = kw.get('model','text-embedding-ada-002')
        llm = openai.model(model)
        resp = llm.embed(text, **kw)
        resp['model'] = model
        return resp

    tokenizer_model = openai.model('text-davinci-003')
    def get_token_count(text):
        return tokenizer_model.token_count(text)

class PDFProcessor:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 24,
            length_function = get_token_count,
        )

    def process_folder(self):
        all_chunks = []
        for filename in os.listdir(self.folder_name):
            if filename.endswith(".pdf"):
                filepath = os.path.join(self.folder_name, filename)
                text = self._extract_text(filepath)
                self._save_text_to_file(filename, text)
                chunks = self._split_text(text)
                all_chunks.extend(chunks)
        return all_chunks

    def _extract_text(self, filepath):
        doc = textract.process(filepath)
        return doc.decode('utf-8')

    def _save_text_to_file(self, filename, text):
        with codecs.open('./text/' + filename[:-4] + '.txt', 'w', 'utf-8') as f:
            f.write(text)

    def _split_text(self, text):
        return self.text_splitter.create_documents([text])



def connect_to_database():

    # Connect to SQLite database and load sqlite-vss extension
    conn = sqlite3.connect('my_database.db')
    conn.enable_load_extension(True)
    conn.load_extension('./vss0')

    # Create vss0 table
    conn.execute('CREATE VIRTUAL TABLE vss_articles USING vss0(headline_embedding(384))')

    # Array to hold all chunks
    all_chunks = []


    return conn


def interact_with_user():

# Create embeddings using ai.py's embedding function
embeddings = [embedding(chunk) for chunk in all_chunks]

chat_history = []
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), embeddings)

while True:
    # Get user query



    def sanitize_query(query):
        # Remove special characters
        sanitized_query = re.sub(r'\W', ' ', query)
        
        # Convert to lowercase
        sanitized_query = sanitized_query.lower()
        
        # Remove extra spaces
        sanitized_query = re.sub(r'\s+', ' ', sanitized_query).strip()
        
        return sanitized_query

    query = input("Enter a query (type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    # Use ai.py's embedding function to generate a response
    result = embedding({"question": query, "chat_history": chat_history})

    def build_prompt(query, similar_sections):
        # Combine the query and the similar sections
        prompt = f"Question: {query}\n\nContext sections:\n{similar_sections}"
        return prompt


    def text_completion(prompt):
        # Send the prompt to the OpenAI API to generate a text completion
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=1000,
            temperature=0,
            stream=True,
        )
        return response

    def process_event_stream(event_stream):
        final_response = ""
        try:
            # Parse the event stream and concatenate the text completions
            for event in event_stream:
                if 'choices' in event:
                    final_response += event['choices'][0]['text']
        except Exception as e:
            print(f"An error occurred: {e}")
        return final_response

    # Convert the result into a query vector
    query_vector = result['embeddings']

    # Search the database of vectors using the query vector
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vss_articles WHERE vss_search(headline_embedding, ?) LIMIT 10;", (json.dumps(query_vector.tolist()),))

    
    def generate_response(similar_vectors):
        # TODO: Implement logic to generate a response based on the most similar vectors
        pass
    # Generate a response based on the most similar vectors
    response = generate_response(cursor.fetchall())

    # Existing code...
    chat_history.append((query, response))
    print(response)


# Main Execution Code
def main():


    def process_pdf_folder(pdf_folder, text_folder):
        # TODO: Implement logic to process PDF files in the given folder and return all text chunks
        pass
  connect_to_database()
    all_chunks = process_pdf_folder("./pdf", "./text")
    embedder = PDFEmbedder()
    for i, chunk in enumerate(all_chunks):
    embedding = embedder.create_embedding(chunk)
    conn.execute('INSERT INTO vss_articles(rowid, headline_embedding) VALUES (?, ?)', (i, json.dumps(embedding.tolist())))
    
    interact_with_user()
# Commit changes and close connection
    conn.commit()
    conn.close()

    

if __name__ == "__main__":
    main()


print("Exited!!!")
