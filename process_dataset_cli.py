import argparse
from chromadb import Client
from langchain.embeddings import HuggingFaceEmbeddings
from dataset_processor import add_dataset_to_chromadb

def initialize_chromadb():
    """Initializes the ChromaDB client."""
    client = Client()
    collection = client.get_or_create_collection("rag_content")
    return collection

def main():
    parser = argparse.ArgumentParser(description='Process a dataset and add to ChromaDB')
    parser.add_argument('--dataset', required=True, help='Path to the dataset file (CSV, JSON, etc.)')
    parser.add_argument('--text-column', required=True, help='Column containing the text to embed')
    parser.add_argument('--id-column', help='Column containing unique IDs (optional)')
    
    args = parser.parse_args()
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize ChromaDB
    collection = initialize_chromadb()
    
    # Process the dataset
    success = add_dataset_to_chromadb(args.dataset, args.text_column, args.id_column, collection)
    
    if success:
        print("Dataset processed successfully!")
    else:
        print("Failed to process dataset.")

if __name__ == "__main__":
    main() 