#!/usr/bin/env python3
import os
import sys
from datasets import load_dataset

def main():
    dataset_name = "McAuley-Lab/Amazon-Reviews-2023"
    config_name = "raw_meta_Automotive"  # metadata for Automotive category
    output_path = "data/metadata/raw/metadata.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        print(f"Downloading dataset '{config_name}' from {dataset_name}...")
        metadata_dataset = load_dataset(dataset_name, config_name, split="full", trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: Failed to load dataset '{config_name}'.\n{e}")
        sys.exit(1)
    
    # Summarize loaded metadata
    num_items = len(metadata_dataset)
    columns = metadata_dataset.column_names
    print(f"Loaded metadata for {num_items} Automotive products.")
    print(f"Metadata columns: {columns}")
    
    try:
        # Save all metadata fields to CSV
        metadata_dataset.to_csv(output_path)
        print(f"Saved metadata to {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save metadata to {output_path}.\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
