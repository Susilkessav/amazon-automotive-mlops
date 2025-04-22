#!/usr/bin/env python3
import os, sys
from datasets import load_dataset
import pathlib
import pandas as pd

def main():
    dataset    = "McAuley-Lab/Amazon-Reviews-2023"
    config     = "raw_review_Automotive"
    split_name = "full"
    out_dir    = os.path.join("data", "reviews", "raw")
    out_path   = os.path.join(out_dir, "reviews.jsonl")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Ensured directory: {out_dir!r}")

    # Attempt to load the main reviews dataset
    try:
        print(f"Loading split='{split_name}' from {dataset}, config='{config}'...")
        ds = load_dataset(dataset, config, split=split_name, trust_remote_code=True)
    except Exception as e:
        print(f"❌  Failed to load main dataset {dataset} ({config}):\n{e}", file=sys.stderr)
        print("ℹ️  Falling back to smaller built-in dataset sample...")
        try:
            ds = load_dataset("amazon_us_reviews", "Automotive_v1_00", split="train[:1%]")
            print(f"Loaded fallback dataset with {len(ds)} records — columns: {list(ds.features.keys())}")
        except Exception as e2:
            print(f"❌  Fallback dataset load failed:\n{e2}", file=sys.stderr)
            sys.exit(1)

    print(f"✅  Loaded {len(ds)} records — columns: {ds.column_names if hasattr(ds, 'column_names') else list(ds.features.keys())}")
    print(f"➡️  Saving reviews to {out_path!r} as JSONL...")
    # Save dataset to JSON Lines
    try:
        ds.to_json(out_path, lines=True)
        count = len(ds)
    except Exception:
        # Fallback to pandas for writing if needed
        df = pd.DataFrame(ds)
        # Standardize column names for consistency
        if "product_id" in df.columns:
            rename_map = {}
            if "product_id" in df.columns:    # product ASIN
                rename_map["product_id"] = "parent_asin"
            if "review_body" in df.columns:
                rename_map["review_body"] = "text"
            if "review_headline" in df.columns:
                rename_map["review_headline"] = "title"
            if "star_rating" in df.columns:
                rename_map["star_rating"] = "rating"
            if "review_date" in df.columns:
                rename_map["review_date"] = "date"
            df = df.rename(columns=rename_map)
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)
        count = len(df)
    print(f"✅  Saved {count} review records to {out_path}")
    
if __name__ == "__main__":
    main()
