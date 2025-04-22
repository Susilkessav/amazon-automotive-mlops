from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import requests
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai.embeddings import OpenAIEmbeddings
from ollama import Client as OllamaClient
from pydantic_settings import BaseSettings
from langchain_community.embeddings import HuggingFaceEmbeddings
import glob
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from anthropic import Anthropic

app = Flask(__name__)


def process_pdfs_in_folder(folder_path, collection):
    """Processes all PDFs in a folder and adds their content to ChromaDB."""
    try:
        # Get all PDF files in the folder
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in the folder: {folder_path}")
            return

        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path}")
            process_pdf_and_add_to_chromadb(pdf_path, collection)
            print(f"Added '{pdf_path}' to the knowledge base.")
    except Exception as e:
        print(f"Error processing PDFs in folder {folder_path}: {e}")


# Step 1: PDF Extraction and Text Splitting


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF {pdf_path}: {e}")


def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    """Splits text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Step 2: Initialize ChromaDB


def initialize_chromadb():
    """Initializes the ChromaDB client with persistent storage."""
    # Use the same persistent directory
    persist_dir = os.path.join(os.getcwd(), "chroma_db")
    
    # Initialize client with persistent storage
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Use the proper embedding function from ChromaDB
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get or create collection with embedding function
    collection = client.get_or_create_collection(
        name="amazon_metadata",
        embedding_function=embedding_function
    )
    
    return collection


def add_content_to_chromadb(content, source_id, collection):
    """Adds content to ChromaDB after embedding."""
    chunks = split_text_into_chunks(content)
    for i, chunk in enumerate(chunks):
        try:
            # The collection already has the embedding function, so we don't need to generate embeddings manually
            collection.add(
                documents=[chunk],
                metadatas=[{"source": f"{source_id}_chunk_{i}"}],
                ids=[f"{source_id}_chunk_{i}"]
            )
        except Exception as e:
            print(f"Error adding chunk {i} to ChromaDB: {e}")


def process_pdf_and_add_to_chromadb(pdf_path, collection):
    """Processes a PDF and adds its content to ChromaDB."""
    try:
        content = extract_text_from_pdf(pdf_path)
        add_content_to_chromadb(content, pdf_path, collection)
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")

# Step 4: Document Retrieval


def retrieve_documents(query, collection):
    """Retrieve relevant documents from ChromaDB."""
    try:
        # The collection already has the embedding function attached
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        return results["documents"]
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []


# Step 5: Initialize Ollama Client and Generate Response


def generate_response(query, context):
    """Generate a response using the local Ollama server."""
    try:
        # Define the prompt with context and query
        prompt = f"""You are an intelligent assistant. You should always use only the provided context to answer the query.
                    If the relevant answer is not found in the context or if the context is missing or empty, ONLY respond with "Not found."

                    Context:
                    {context}

                    Query:
                    {query}

                    Answer:
                    """

        # API endpoint of the local Ollama server
        url = "http://127.0.0.1:11434/api/generate"

        # Payload for the POST request
        payload = {
            "model": "mistral",  # Replace with the valid model available on your server
            "prompt": prompt
        }

        # Send the request to the local server
        response = requests.post(url, json=payload)

        # Check the response status
        if response.status_code == 200:
            return response.json().get("text", "No response text found.")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error generating response: {e}. Please ensure the server is running and the model name is correct."


def generate_response_claude(query, context):
    """Generate a response using Claude API."""
    try:
        # Define the prompt with context and query
        messages = [
            {
                "role": "user",
                "content": f"""You are an intelligent assistant. You should only use the provided context to answer the query .
                    If the relevant answer is not found in the context or if the context is missing or empty, ONLY respond with "Not found." 

                    Context:
                    {context}

                    Query:
                    {query}

                    Answer:
                    """
            }
        ]

        # API endpoint for Claude
        url = "https://api.anthropic.com/v1/messages"

        # Headers with API key
        headers = {
            # Recommended: use environment variable
            "x-api-key": "",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"  # Update to the current API version
        }

        # Payload for the API request
        payload = {
            "model": "claude-3-haiku-20240307",  # Updated model name
            "messages": messages,
            "max_tokens": 300
        }

        # Send the POST request
        response = requests.post(url, json=payload, headers=headers)

        # Check the response status
        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error generating response: {e}. Please ensure the API key and configuration are correct."


def rag_pipeline(query, collection):
    """Complete RAG pipeline: retrieve context and generate response."""
    retrieved_docs = retrieve_documents(query, collection)
    if not retrieved_docs:
        return "No relevant documents found."

    # Flatten nested lists in retrieved_docs
    flattened_docs = [doc for sublist in retrieved_docs for doc in sublist]

    # Join all documents into a single string
    context = "\n".join(flattened_docs)

    print("\n\nContext for the query:", context)
    return generate_response_claude(query, context)


def interactive_cli():
    """Runs an interactive CLI for the application."""
    print("\n\nWelcome to the RAG-based Interactive System!")
    try:
        collection = initialize_chromadb()
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return

    while True:
        print("\nMenu:")
        print("1. Add a single PDF to the knowledge base")
        print("2. Add multiple PDFs from a folder")
        print("3. Ask a question")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            pdf_path = input("Enter the path to the PDF file: ")
            process_pdf_and_add_to_chromadb(pdf_path, collection)
            print(f"PDF '{pdf_path}' added to the knowledge base.")

        elif choice == "2":
            folder_path = input("Enter the folder path containing PDF files: ")
            process_pdfs_in_folder(folder_path, collection)

        elif choice == "3":
            query = input("Enter your question: ")
            try:
                response = rag_pipeline(query, collection)
                print(f"\n\nResponse:\n{response}")
            except Exception as e:
                print(f"Error retrieving answer: {e}")

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and file.filename.endswith('.pdf'):
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save the file temporarily and process it
            collection = initialize_chromadb()
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)

            try:
                # Process the PDF and create embeddings
                process_pdf_and_add_to_chromadb(file_path, collection)
            finally:
                # Clean up temp file even if processing fails
                if os.path.exists(file_path):
                    os.remove(file_path)

            return jsonify({
                'message': 'File processed successfully',
                'status': 'complete'
            })
        else:
            return jsonify({'error': 'Invalid file type. Please upload a PDF file.'})

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})
    
@app.route('/process_dataset', methods=['POST'])
def process_dataset_route():
    try:
        data = request.json
        dataset_path = data.get('dataset_path')
        text_column = data.get('text_column')
        id_column = data.get('id_column')
        
        if not dataset_path or not text_column:
            return jsonify({'error': 'Dataset path and text column are required'})
            
        collection = initialize_chromadb()
        success = add_dataset_to_chromadb(dataset_path, text_column, id_column, collection)
        
        if success:
            return jsonify({
                'message': 'Dataset processed successfully',
                'status': 'complete'
            })
        else:
            return jsonify({'error': 'Failed to process dataset'})
            
    except Exception as e:
        return jsonify({'error': f'Error processing dataset: {str(e)}'})


@app.route('/query', methods=['POST'])
def query():
    print("Query endpoint called")
    try:
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'Query is required'})
        
        # Initialize ChromaDB and get the collection
        collection = initialize_chromadb()
        
        # Increase number of results from 5 to 10
        n_results = 10
        print(f"\n=== QUERY: '{user_query}' ===")
        print(f"Retrieving top {n_results} chunks...")
        
        # Query the collection
        results = collection.query(
            query_texts=[user_query],
            n_results=n_results
        )
        print("gathered results from chroma")
        # Extract the relevant documents and metadata
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        print(f"\n=== RETRIEVED {len(documents)} CHUNKS ===")
        
        # Check if we have any results
        if len(documents) == 0:
            print("No relevant documents found in the database.")
            response = "I don't have specific information about Suzuki Outboard Oil filters in my database. Please try a different query related to automotive products."
            return jsonify({
                'response': response,
                'sources': [],
                'embedding_model': "HuggingFaceEmbeddings (all-MiniLM-L6-v2)"
            })
        
        # Format the context from retrieved documents
        context = ""
        sources = []
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            # Print chunk details to terminal
            print(f"\nCHUNK {i+1}:")
            print(f"Title: {meta.get('title', 'N/A')}")
            print(f"Brand: {meta.get('brand', 'N/A')}")
            print(f"Category: {meta.get('category', 'N/A')}")
            print(f"Content: {doc[:200]}..." if len(doc) > 200 else f"Content: {doc}")
            print("-" * 50)
            
            # Add to context for LLM
            context += f"Document {i+1}:\n"
            context += f"Title: {meta.get('title', 'N/A')}\n"
            context += f"Brand: {meta.get('brand', 'N/A')}\n"
            context += f"Category: {meta.get('category', 'N/A')}\n"
            context += f"Content: {doc}\n\n"
            
            # Add to sources for display
            sources.append({
                'snippet': doc[:200] + '...' if len(doc) > 200 else doc
            })
        
        print("\n=== SENDING TO LLM ===")
        
        # Create the prompt for Anthropic
        prompt = f"""You are a helpful assistant for automotive product information.
            
Use the following retrieved information to answer the user's question.
If you don't know the answer, just say that you don't know.

Retrieved Information:
{context}

User Question: {user_query}

Answer:"""
        
        # Initialize the Anthropic client directly
        anthropic_client = Anthropic(api_key="")
        
        # Call the Anthropic API
        message = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0,
            system="You are a helpful assistant for automotive product information.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the response
        response = message.content[0].text
        print(f"\n=== LLM RESPONSE ===\n{response}\n")
        
        return jsonify({
            'response': response,
            'sources': sources,
            'embedding_model': "HuggingFaceEmbeddings (all-MiniLM-L6-v2)"
        })
        
    except Exception as e:
        error_msg = f'Error processing query: {str(e)}'
        print(f"\n=== ERROR ===\n{error_msg}\n")
        return jsonify({'error': error_msg})


@app.route('/database_sample', methods=['GET'])
def database_sample():
    try:
        # Initialize ChromaDB and get the collection
        collection = initialize_chromadb()
        
        # Get sample size from query parameter, default to 10
        sample_size = int(request.args.get('size', 10))
        
        # Query a few documents
        results = collection.query(
            query_texts=["product"],
            n_results=sample_size
        )
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Build a list of sample dicts
        samples = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            samples.append({
                'index': i + 1,
                'title':    meta.get('title', 'N/A'),
                'brand':    meta.get('brand', 'N/A'),
                'category': meta.get('main_category', meta.get('category', 'N/A')),
                'content':  doc[:200] + '...' if len(doc) > 200 else doc,
                'metadata': meta
            })
        
        # Build the per-sample HTML fragments
        sample_html = ""
        for s in samples:
            sample_html += f"""
            <div class="sample">
                <h3>Sample #{s['index']}: {s['title']}</h3>
                <p><strong>Brand:</strong> {s['brand']}</p>
                <p><strong>Category:</strong> {s['category']}</p>
                <div class="content">{s['content']}</div>
                <details>
                  <summary>Full Metadata</summary>
                  <div class="metadata">{s['metadata']}</div>
                </details>
            </div>
            """
        
        # Now inject that into a single f‑string
        html_response = f"""
        <html>
        <head>
            <title>Database Sample</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .sample {{ border:1px solid #ddd; padding:15px; margin-bottom:15px; border-radius:5px; }}
                .metadata {{ font-family:monospace; background:#eee; padding:10px; overflow-x:auto; }}
                .content {{ background:#f0f8ff; padding:10px; margin-top:10px; border-left:3px solid #0066cc; }}
            </style>
        </head>
        <body>
            <h1>ChromaDB Collection Sample</h1>
            <p>Total documents in collection: {collection.count()}</p>
            <h2>Sample Documents ({len(samples)})</h2>
            {sample_html}
            <p><a href="/">Back to Chat</a></p>
        </body>
        </html>
        """
        
        return html_response
        
    except Exception as e:
        error_msg = f'Error retrieving database sample: {e}'
        return f"<h1>Error</h1><p>{error_msg}</p><p><a href='/'>Back to Chat</a></p>"

@app.route('/database_info', methods=['GET'])
def database_info():
    try:
        # Initialize ChromaDB client (not collection)
        persist_dir = os.path.join(os.getcwd(), "chroma_db")
        client = chromadb.PersistentClient(path=persist_dir)
        
        # Get all collection names
        collections = client.list_collections()
        
        # Build HTML response
        html_response = f"""
        <html>
        <head>
            <title>ChromaDB Information</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .collection {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; }}
                .collection:hover {{ background: #f9f9f9; }}
                .empty {{ color: #999; }}
                pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>ChromaDB Database Information</h1>
            
            <p>Database location: <code>{persist_dir}</code></p>
            <p>Total collections: {len(collections)}</p>
            
            <h2>Collections</h2>
            """
        
        if not collections:
            html_response += "<p class='empty'>No collections found in the database.</p>"
        else:
            for coll in collections:
                collection = client.get_collection(coll.name)
                count = collection.count()
                
                html_response += f"""
                <div class="collection">
                    <h3>Collection: {coll.name}</h3>
                    <p>Document count: {count}</p>
                    """
                
                if count > 0:
                    # Get a sample document if collection is not empty
                    sample = collection.query(query_texts=["product"], n_results=1)
                    html_response += f"""
                    <details>
                        <summary>Sample document</summary>
                        <pre>{sample}</pre>
                    </details>
                    """
                else:
                    html_response += "<p class='empty'>Collection is empty</p>"
                
                html_response += "</div>"
        
        html_response += """
            <h2>Troubleshooting</h2>
            <ol>
                <li>Make sure you've run the ingest_metadata.py script to populate the database</li>
                <li>Check that the collection name in your query matches the one used during ingestion</li>
                <li>Verify that the chroma_db directory contains data files</li>
            </ol>
            
            <p><a href="/">Back to Chat</a></p>
        </body>
        </html>
        """
        
        return html_response
        
    except Exception as e:
        error_msg = f'Error retrieving database info: {str(e)}'
        print(f"\n=== ERROR ===\n{error_msg}\n")
        return f"<h1>Error</h1><p>{error_msg}</p><p><a href='/'>Back to Chat</a></p>"

if __name__=="__main__":
    load_dotenv()
    # disable debug reloader so your imports in .venv don't trigger reloads
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)