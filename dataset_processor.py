import os
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_dataset(dataset_path):
    """Load dataset from CSV or other formats"""
    if dataset_path.endswith('.csv'):
        return pd.read_csv(dataset_path)
    elif dataset_path.endswith('.json'):
        return pd.read_json(dataset_path)
    elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
        return pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")

def process_dataset(dataset, text_column, id_column=None, chunk_size=500, chunk_overlap=50):
    """Process dataset and split text into chunks for embedding"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in dataset.iterrows():
        text = row[text_column]
        if not isinstance(text, str) or pd.isna(text):
            continue
            
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Generate a unique ID if not provided
        doc_id = str(row[id_column]) if id_column else f"doc_{idx}"
        
        # Add each chunk with metadata
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            
            # Create metadata from all columns except the text column
            metadata = {col: row[col] for col in dataset.columns if col != text_column and not pd.isna(row[col])}
            metadata['chunk_id'] = i
            metadata['source_id'] = doc_id
            
            metadatas.append(metadata)
            ids.append(f"{doc_id}_chunk_{i}")
    
    return documents, metadatas, ids

def add_dataset_to_chromadb(dataset_path, text_column, id_column=None, collection=None):
    """Process a dataset and add its content to ChromaDB"""
    try:
        # Load the dataset
        df = load_dataset(dataset_path)
        
        if text_column not in df.columns:
            print(f"Error: Column '{text_column}' not found in dataset.")
            return False
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Initialize embeddings model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        documents = []
        metadatas = []
        ids = []
        
        # Process each row in the dataset
        for idx, row in df.iterrows():
            # Get the text to embed
            text = str(row[text_column])
            
            # Skip empty texts
            if not text or text.isspace():
                continue
                
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            
            # Generate a unique ID
            doc_id = str(row[id_column]) if id_column and id_column in row else f"doc_{idx}"
            
            # Add each chunk with metadata
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                
                # Create metadata from all columns
                metadata = {}
                for col in df.columns:
                    if col != text_column and pd.notna(row[col]):
                        # Convert numpy types to Python native types
                        if isinstance(row[col], (np.integer, np.floating)):
                            metadata[col] = row[col].item()
                        else:
                            metadata[col] = str(row[col])
                
                metadatas.append(metadata)
                ids.append(f"{doc_id}_{i}")
        
        if documents:
            # Add to collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Added {len(documents)} chunks from dataset to the knowledge base.")
            return True
    except Exception as e:
        print(f"Error processing dataset {dataset_path}: {e}")
        return False 