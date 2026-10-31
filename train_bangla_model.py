"""
Training Script for Bangla Medical Query Intent & Urgency Classifier
====================================================================
Course: CSE AI / NLP Healthcare Project
Dataset: d:\\bangla_health_queries_500k.csv (500,000 Bangla health queries)

This script trains two NLP pipelines:
1. Urgency Classifier: Routine vs Urgent Review vs Emergency
2. Category Classifier: Symptom, Medicine, Disease, Emergency, etc.

Method:
- TF-IDF (Term Frequency-Inverse Document Frequency) n-gram vectorizer
- Linear Support Vector Classification (LinearSVC) / Logistic Regression
- Evaluation: Classification Report (Precision, Recall, F1-Score)
- Export: Serialized joblib models saved into models/
"""

import os
import sys
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Ensure terminal output supports Unicode/Bangla
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = r"d:\bangla_health_queries_500k.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  BANGLA HEALTHCARE QUERY CLASSIFIER - TRAINING PIPELINE")
print("=" * 65)

# Step 1: Load Dataset
print(f"\n[1/5] Loading dataset from: {DATA_PATH} ...")
start_time = time.time()
df = pd.read_csv(DATA_PATH, encoding='utf-8', usecols=['query_bn', 'category', 'urgency'])
df = df.dropna()
print(f"      Loaded {len(df):,} valid records in {time.time() - start_time:.2f}s")
print(f"      Urgency breakdown:\n{df['urgency'].value_counts().to_string()}")

# Step 2: Train/Test Split (80% train, 20% test)
print("\n[2/5] Splitting data into 80% Train and 20% Test sets...")
X = df['query_bn']
y_urgency = df['urgency']
y_category = df['category']

X_train, X_test, y_urg_train, y_urg_test, y_cat_train, y_cat_test = train_test_split(
    X, y_urgency, y_category, test_size=0.20, random_state=42, stratify=y_urgency
)
print(f"      Training samples: {len(X_train):,}")
print(f"      Testing samples:  {len(X_test):,}")

# Step 3: Train Urgency Classifier
print("\n[3/5] Training Urgency Model (TF-IDF + SGDClassifier/Logistic Loss)...")
t0 = time.time()
urgency_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=35000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1))
])

urgency_pipeline.fit(X_train, y_urg_train)
print(f"      Trained in {time.time() - t0:.2f}s")

# Evaluate Urgency
print("\n      --- Urgency Model Evaluation on Test Set (100,000 samples) ---")
urg_preds = urgency_pipeline.predict(X_test)
print(f"      Accuracy: {accuracy_score(y_urg_test, urg_preds) * 100:.2f}%\n")
print(classification_report(y_urg_test, urg_preds, digits=4))

# Step 4: Train Category Classifier
print("\n[4/5] Training Category Model (TF-IDF + SGDClassifier/Logistic Loss)...")
t0 = time.time()
category_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=35000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1))
])

category_pipeline.fit(X_train, y_cat_train)
print(f"      Trained in {time.time() - t0:.2f}s")

# Evaluate Category
print("\n      --- Category Model Evaluation on Test Set (100,000 samples) ---")
cat_preds = category_pipeline.predict(X_test)
print(f"      Accuracy: {accuracy_score(y_cat_test, cat_preds) * 100:.2f}%\n")
print(classification_report(y_cat_test, cat_preds, digits=4))

# Step 5: Save Models
print("\n[5/5] Saving trained models to disk...")
urgency_file = os.path.join(OUTPUT_DIR, "bangla_urgency_pipeline.joblib")
category_file = os.path.join(OUTPUT_DIR, "bangla_category_pipeline.joblib")

joblib.dump(urgency_pipeline, urgency_file, compress=3)
joblib.dump(category_pipeline, category_file, compress=3)

print(f"      Saved: {urgency_file}")
print(f"      Saved: {category_file}")
print(f"\nAll done! Total training time: {(time.time() - start_time) / 60:.2f} minutes.")
