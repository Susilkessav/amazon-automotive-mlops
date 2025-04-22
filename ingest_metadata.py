#!/usr/bin/env python3
import os
import sys
from datasets import load_dataset
import pandas as pd
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import json
import traceback
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import atexit

def initialize_chromadb():
    """Initializes the ChromaDB client with persistent storage."""
    # Create directory for persistent storage if it doesn't exist
    persist_dir = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(persist_dir, exist_ok=True)
    
    print(f"Using ChromaDB persistent directory: {persist_dir}")
    
    # Initialize client with persistent storage
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Create a proper embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get or create collection with embedding function
    collection = client.get_or_create_collection(
        name="amazon_metadata",
        embedding_function=embedding_function
    )
    
    print(f"Initialized ChromaDB with persistent storage at: {persist_dir}")
    return client, collection

def debug_value(value, label="Value"):
    """Print detailed debug information about a value"""
    print(f"\n--- DEBUG {label} ---")
    print(f"Type: {type(value)}")
    
    if isinstance(value, (list, np.ndarray)):
        print(f"Length: {len(value)}")
        print(f"First few elements: {value[:5] if len(value) > 5 else value}")
        if len(value) > 0:
            print(f"Type of first element: {type(value[0])}")
    elif isinstance(value, dict):
        print(f"Keys: {list(value.keys())}")
        print(f"Sample values: {list(value.values())[:3] if len(value) > 3 else list(value.values())}")
    else:
        try:
            print(f"Value: {value}")
        except:
            print("Value cannot be printed")
    
    try:
        # Try to convert to JSON to see if it's serializable
        json_str = json.dumps(value)
        print("Can be converted to JSON: Yes")
    except:
        print("Can be converted to JSON: No")
    
    print("--- END DEBUG ---\n")

def is_valid_value(value):
    """Check if a value is valid for processing."""
    # Handle arrays specifically
    if isinstance(value, (list, np.ndarray)):
        # For arrays, check if it has any non-NaN elements
        if len(value) == 0:
            return False
        # For arrays of arrays or complex structures, just return True if it has elements
        return True
    
    # For scalar values, use pd.isna
    if pd.isna(value):
        return False
    
    return True

def safe_str_conversion(value):
    """Safely convert various types to string."""
    try:
        if isinstance(value, (list, np.ndarray)):
            # For arrays, convert each element to string and join
            result = []
            for item in value:
                if isinstance(item, (list, np.ndarray)):
                    # Handle nested arrays
                    nested_items = [str(x) for x in item if not pd.isna(x)]
                    result.append("[" + ", ".join(nested_items) + "]")
                elif not pd.isna(item):
                    result.append(str(item))
            return ", ".join(result)
        elif isinstance(value, dict):
            # Convert dictionary to string
            try:
                return json.dumps(value)
            except:
                return str(value)
        elif isinstance(value, (np.integer, np.floating)):
            return str(value.item())
        elif pd.isna(value):
            return ""
        else:
            return str(value)
    except Exception as e:
        print(f"Error converting value to string: {e}")
        debug_value(value, "Failed conversion")
        return ""

def process_metadata_for_embeddings(metadata_df, text_columns, collection):
    """Process metadata and create embeddings in ChromaDB"""
    print("\n=== STARTING METADATA PROCESSING ===")
    print(f"Total rows to process: {len(metadata_df)}")
    print(f"Text columns being used: {text_columns}")
    print(f"Non-text columns (for metadata): {[col for col in metadata_df.columns if col not in text_columns]}")
    
    # Initialize text splitter with larger chunk size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,  # Increased from 500 to 2000
        chunk_overlap=1000  # Increased from 50 to 100
    )
    print(f"Text splitter configured with chunk_size=2000, chunk_overlap=100")
    
    # Print sample row to understand structure
    print("\n=== SAMPLE ROW STRUCTURE ===")
    sample_row = metadata_df.iloc[0]
    for col in metadata_df.columns:
        try:
            print(f"\nColumn: {col}")
            debug_value(sample_row[col], f"Column {col}")
        except Exception as e:
            print(f"Error inspecting column {col}: {e}")
    print("=== END SAMPLE ROW ===\n")
    
    documents = []
    metadatas = []
    ids = []
    
    total_chunks = 0
    total_rows_processed = 0
    total_rows_skipped = 0
    total_batches_added = 0
    
    # Process each row in the metadata
    for idx, row in metadata_df.iterrows():
        try:
            # Debug more frequently for first 10 rows, then every 1000 rows
            should_debug = idx < 10 or idx % 1000 == 0
            
            if should_debug:
                print(f"\n--- Processing row {idx}... ---")
                print(f"Stats so far: {total_rows_processed} rows processed, {total_chunks} chunks created, {total_rows_skipped} rows skipped, {total_batches_added} batches added")
            
            # Combine specified text columns into a single text
            combined_text = ""
            for col in text_columns:
                if col in row.index:
                    try:
                        if is_valid_value(row[col]):
                            value_str = safe_str_conversion(row[col])
                            if value_str:  # Only add if not empty
                                combined_text += f"{col.upper()}: {value_str}\n\n"
                                if should_debug:
                                    print(f"Added {col} with {len(value_str)} chars")
                    except Exception as e:
                        print(f"Error processing column {col} in row {idx}: {e}")
                        print(traceback.format_exc())
                        continue
            
            if not combined_text:
                if should_debug:
                    print(f"Row {idx} has no valid text content, skipping")
                total_rows_skipped += 1
                continue
                
            # Split text into chunks
            chunks = text_splitter.split_text(combined_text)
            if should_debug:
                print(f"Split into {len(chunks)} chunks")
                if chunks:
                    print(f"First chunk sample: {chunks[0][:100]}...")
            
            # Generate a unique ID
            doc_id = f"doc_{idx}"
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                total_chunks += 1
                
                # Create metadata from all columns
                metadata = {}
                for col in metadata_df.columns:
                    if col not in text_columns and col in row.index:
                        try:
                            if is_valid_value(row[col]):
                                metadata[col] = safe_str_conversion(row[col])
                        except Exception as e:
                            print(f"Error processing metadata column {col} in row {idx}: {e}")
                            print(traceback.format_exc())
                            continue
                
                metadatas.append(metadata)
                ids.append(f"{doc_id}_{i}")
                
                # Process in batches to avoid memory issues - increased batch size
                if len(documents) >= 3000:  # Increased from 1000 to 2000
                    try:
                        collection.add(
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids
                        )
                        print(f"Added batch of {len(documents)} chunks to the knowledge base.")
                        total_batches_added += 1
                        documents = []
                        metadatas = []
                        ids = []
                    except Exception as e:
                        print(f"Error adding batch to collection: {e}")
                        print(traceback.format_exc())
                        documents = []
                        metadatas = []
                        ids = []
            
            total_rows_processed += 1
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            print(traceback.format_exc())
            total_rows_skipped += 1
            continue
    
    # Add any remaining documents
    if documents:
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added final batch of {len(documents)} chunks to the knowledge base.")
            total_batches_added += 1
            total_chunks += len(documents)
            print("Embeddings created successfully!")
        except Exception as e:
            print(f"Error adding final batch to collection: {e}")
            print(traceback.format_exc())
    else:
        print("No valid documents found for embedding.")
    
    # Print final statistics
    print("\n=== EMBEDDING PROCESS COMPLETE ===")
    print(f"Total rows processed: {total_rows_processed}")
    print(f"Total rows skipped: {total_rows_skipped}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Total batches added: {total_batches_added}")
    print("===================================\n")

def main():
    load_dotenv()  # Load environment variables
    
    dataset_name = "McAuley-Lab/Amazon-Reviews-2023"
    config_name = "raw_meta_Automotive"  # metadata for Automotive category
    
    # Create a client variable at the module level
    global client
    client = None
    
    try:
        print(f"Downloading dataset '{config_name}' from {dataset_name}...")
        # Only load the first 20,000 rows directly
        metadata_dataset = load_dataset(
            dataset_name, 
            config_name, 
            # split="full",  # Only get first 20,000 rows
            trust_remote_code=True
        )
        
        # Summarize loaded metadata
        num_items = len(metadata_dataset)
        columns = metadata_dataset.column_names
        print(f"Loaded metadata for {num_items} Automotive products (limited to first 20,000).")
        print(f"Metadata columns: {columns}")
        
        # Print dataset features to understand structure
        print("\nDataset Features:")
        for feature_name, feature in metadata_dataset.features.items():
            print(f"{feature_name}: {feature}")
        
        # Convert to pandas DataFrame for processing
        print("\nConverting to pandas DataFrame...")
        metadata_df = pd.DataFrame(metadata_dataset)
        
        # Print DataFrame info
        print("\nDataFrame Info:")
        print(metadata_df.info())
        
        # Print sample data types
        print("\nSample Data Types:")
        for col in metadata_df.columns:
            sample_val = metadata_df[col].iloc[0]
            print(f"{col}: {type(sample_val)}")
        
        # Text columns to use for embeddings - reduced to most important columns
        text_columns = [
            'title', 
            'description', 
            'features', 
            'categories'
        ]
        
        # Initialize ChromaDB
        client, collection = initialize_chromadb()
        
        # Register cleanup function
        def cleanup():
            if client is not None:
                print("\nClosing ChromaDB client to ensure data persistence...")
        
        atexit.register(cleanup)
        
        # Process metadata and create embeddings
        print("\nCreating embeddings from metadata...")
        process_metadata_for_embeddings(metadata_df, text_columns, collection)
        
        # Verify the collection has data
        count = collection.count()
        print(f"\nVerification: Collection now has {count} documents")
        
        # Add a small test document to verify persistence is working
        test_id = "test_persistence"
        collection.add(
            documents=["This is a test document to verify persistence is working"],
            metadatas=[{"source": "test"}],
            ids=[test_id]
        )
        
        print("\nAdded test document. Checking if it exists...")
        results = collection.get(ids=[test_id])
        if results and len(results['documents']) > 0:
            print(f"Test document found: {results['documents'][0]}")
            print("Database persistence appears to be working correctly.")
        else:
            print("WARNING: Test document not found. Database persistence may not be working.")
        
        # Explicitly persist before script ends
        print("\nPersisting data to disk...")
        
    except Exception as e:
        print(f"ERROR: Failed to process metadata.\n{e}")
        print(traceback.format_exc())
        sys.exit(1)
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()