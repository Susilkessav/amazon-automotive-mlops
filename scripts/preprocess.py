#!/usr/bin/env python3
import os
import json
import random

import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Paths
    reviews_path = os.path.join("data", "reviews", "raw", "reviews.jsonl")
    meta_path    = os.path.join("data", "metadata", "raw", "metadata.csv")
    out_dir      = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Stream + sample reviews
    MAX_SAMPLES = 10000 # Limit for sampling
    print(f"Streaming + sampling up to {MAX_SAMPLES} reviews from {reviews_path!r}")
    sample = []
    with open(reviews_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            if i < MAX_SAMPLES:
                sample.append(data)
            else:
                j = random.randint(0, i)
                if j < MAX_SAMPLES:
                    sample[j] = data
    print(f"Kept {len(sample)} reviews in sample")
    df_rev = pd.DataFrame(sample)

    # 2) Normalize rating field
    if 'rating' not in df_rev.columns and 'overall' in df_rev.columns:
        df_rev['rating'] = df_rev['overall']

    # 3) Parse & normalize date → string
    if 'timestamp' in df_rev.columns:
        df_rev['date'] = pd.to_datetime(df_rev['timestamp'], unit='ms', errors='coerce')
    elif 'date' in df_rev.columns:
        df_rev['date'] = pd.to_datetime(df_rev['date'], errors='coerce')
    else:
        raise KeyError("No 'timestamp' or 'date' in reviews")

    df_rev = df_rev.dropna(subset=['date']).reset_index(drop=True)
    df_rev['date'] = df_rev['date'].dt.normalize().astype(str)

    # 4) Load metadata
    print(f"Loading metadata from {meta_path!r}")
    df_meta = pd.read_csv(meta_path, low_memory=False)
    print(f"Loaded {len(df_meta)} metadata rows")

    # 5) Determine merge key
    if 'parent_asin' in df_rev.columns and 'parent_asin' in df_meta.columns:
        join_key = 'parent_asin'
    elif 'asin' in df_rev.columns and 'asin' in df_meta.columns:
        join_key = 'asin'
    else:
        raise KeyError("No matching 'asin' or 'parent_asin' for merge")

    # 6) Merge
    df = df_rev.merge(df_meta, on=join_key, how='inner')
    print(f"After merge on '{join_key}': {len(df)} records")

    # 7) Build LLM records
    rev_cols  = df_rev.columns.tolist()
    meta_cols = df_meta.columns.tolist()
    records   = []
    for _, row in df.iterrows():
        rec_review = {k: row.get(k, row.get(f"{k}_x")) for k in rev_cols}
        rec_meta   = {k: row.get(k, row.get(f"{k}_y")) for k in meta_cols}
        records.append({
            'review': rec_review,
            'meta':   rec_meta,
            'label':  row['rating']
        })

    # 8) Proper 80/10/10 split using sklearn
    train_recs, temp_recs = train_test_split(records, test_size=0.2, random_state=42)
    val_recs,   test_recs = train_test_split(temp_recs, test_size=0.5, random_state=42)
    splits = {
        'train': train_recs,
        'val':   val_recs,
        'test':  test_recs
    }

    for name, recs in splits.items():
        path = os.path.join(out_dir, f"{name}.jsonl")
        print(f"Writing {name} → {path!r}")
        with open(path, 'w', encoding='utf-8') as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"Wrote {len(recs)} records to {name}.jsonl")

    # 9) Write processed CSV with **all** merged columns
    csv_path = os.path.join(out_dir, 'processed.csv')
    # Simply dump the entire merged DataFrame (df) to CSV
    df.to_csv(csv_path, index=False)
    print(f"Wrote full merged DataFrame to {csv_path!r}")

if __name__ == '__main__':
    main()
