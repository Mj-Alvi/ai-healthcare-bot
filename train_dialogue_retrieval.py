"""
Training Script for Doctor-Patient Q&A Retrieval Engine
======================================================
Dataset: d:\\dialogues.csv (256,916 conversations)
Method: TF-IDF Vector Space Model + Cosine Similarity Search Index
Output: models/dialogue_search_index.joblib
"""

import os
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = r"d:\dialogues.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  DOCTOR-PATIENT Q&A RETRIEVAL ENGINE - INDEXING PIPELINE")
print("=" * 65)

# Step 1: Load and clean dialogues
print(f"\n[1/3] Loading conversations from: {DATA_PATH} ...")
t0 = time.time()
df = pd.read_csv(DATA_PATH, usecols=['Patient', 'Doctor'])
df = df.dropna().drop_duplicates(subset=['Patient'])

# Keep clean, high-quality entries with reasonable length
df = df[
    (df['Patient'].str.len() > 20) &
    (df['Patient'].str.len() < 1200) &
    (df['Doctor'].str.len() > 30)
]

# For smooth runtime memory and lightning-fast search in Streamlit,
# select 75,000 diverse Q&As
SAMPLE_SIZE = min(75000, len(df))
df_indexed = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
print(f"      Selected {len(df_indexed):,} high-quality dialogues in {time.time() - t0:.2f}s")

# Step 2: Fit TF-IDF Vectorizer
print("\n[2/3] Building TF-IDF Search Matrix...")
t0 = time.time()
vectorizer = TfidfVectorizer(
    max_features=40000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(df_indexed['Patient'])
print(f"      TF-IDF Matrix shape: {tfidf_matrix.shape}")
print(f"      Vectorized in {time.time() - t0:.2f}s")

# Step 3: Save Index Artifact
print("\n[3/3] Saving search index to models/dialogue_search_index.joblib ...")
index_data = {
    'vectorizer': vectorizer,
    'tfidf_matrix': tfidf_matrix,
    'patients': df_indexed['Patient'].tolist(),
    'doctors': df_indexed['Doctor'].tolist()
}

out_file = os.path.join(OUTPUT_DIR, "dialogue_search_index.joblib")
joblib.dump(index_data, out_file, compress=3)
print(f"      Saved: {out_file} ({os.path.getsize(out_file) / (1024*1024):.2f} MB)")
print("\nRetrieval index built successfully!")
