import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = r"d:\healthcare_patient_queries_combined_300k.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  ENGLISH HEALTHCARE QUERY TRIAGE CLASSIFIER - TRAINING PIPELINE")
print("=" * 65)

# Step 1: Load English records
print(f"\n[1/4] Loading English queries from: {DATA_PATH} ...")
t0 = time.time()
df = pd.read_csv(DATA_PATH, usecols=['language', 'patient_query', 'intent', 'triage_level', 'emergency_flag'])
df = df[df['language'] == 'en'].dropna()
print(f"      Loaded {len(df):,} English records in {time.time() - t0:.2f}s")

# Step 2: Split
X = df['patient_query']
y_triage = df['triage_level']
y_intent = df['intent']

X_train, X_test, y_tri_train, y_tri_test, y_int_train, y_int_test = train_test_split(
    X, y_triage, y_intent, test_size=0.20, random_state=42, stratify=y_triage
)
print(f"      Train: {len(X_train):,}, Test: {len(X_test):,}")

# Step 3: Train Triage Model
print("\n[2/4] Training Triage Model (TF-IDF + SGDClassifier)...")
t0 = time.time()
triage_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1))
])
triage_pipeline.fit(X_train, y_tri_train)
print(f"      Trained in {time.time() - t0:.2f}s")

# Evaluate
tri_preds = triage_pipeline.predict(X_test)
print(f"      Triage Accuracy: {accuracy_score(y_tri_test, tri_preds) * 100:.2f}%\n")
print(classification_report(y_tri_test, tri_preds, digits=4))

# Step 4: Train Intent Model
print("\n[3/4] Training Intent Model (TF-IDF + SGDClassifier)...")
t0 = time.time()
intent_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1))
])
intent_pipeline.fit(X_train, y_int_train)
print(f"      Trained in {time.time() - t0:.2f}s")

# Evaluate
int_preds = intent_pipeline.predict(X_test)
print(f"      Intent Accuracy: {accuracy_score(y_int_test, int_preds) * 100:.2f}%\n")
print(classification_report(y_int_test, int_preds, digits=4))

# Step 5: Save
print("[4/4] Saving English models...")
tri_file = os.path.join(OUTPUT_DIR, "english_triage_pipeline.joblib")
int_file = os.path.join(OUTPUT_DIR, "english_intent_pipeline.joblib")

joblib.dump(triage_pipeline, tri_file, compress=3)
joblib.dump(intent_pipeline, int_file, compress=3)
print(f"      Saved: {tri_file}")
print(f"      Saved: {int_file}")
print("\nEnglish models training complete!")
