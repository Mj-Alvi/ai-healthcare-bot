import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "disease_symptoms.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  SYMPTOM-TO-DISEASE CLASSIFIER - TRAINING PIPELINE")
print("=" * 65)

# Step 1: Load Dataset
print(f"\n[1/4] Loading dataset: {DATA_PATH} ...")
df = pd.read_excel(DATA_PATH)
df['Disease'] = df['Disease'].str.strip()

# Extract all unique symptoms
all_symptoms = set()
for row in df.iloc[:, 1:].values:
    for item in row:
        if pd.notna(item) and str(item).strip():
            all_symptoms.add(str(item).strip().lower().replace(" ", "_"))

symptom_list = sorted(list(all_symptoms))
symptom_to_idx = {s: i for i, s in enumerate(symptom_list)}
print(f"      Total unique symptoms cataloged: {len(symptom_list)}")
print(f"      Total diseases cataloged: {df['Disease'].nunique()}")

# Step 2: Create Multi-Hot Feature Matrix (X) and Target (y)
print("\n[2/4] Transforming symptoms into binary multi-hot feature vectors...")
X = np.zeros((len(df), len(symptom_list)), dtype=int)
y = np.array(df['Disease'].tolist(), dtype=object)

for i, row in enumerate(df.iloc[:, 1:].values):
    for item in row:
        if pd.notna(item) and str(item).strip():
            sym = str(item).strip().lower().replace(" ", "_")
            if sym in symptom_to_idx:
                X[i, symptom_to_idx[sym]] = 1

# Step 3: Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"      Training records: {len(X_train):,}")
print(f"      Testing records:  {len(X_test):,}")

# Step 4: Train Random Forest Model
print("\n[3/4] Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
preds = rf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"      Test Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, preds, digits=4))

# Step 5: Save Artifacts
print("[4/4] Saving symptom model and vocabulary...")
model_data = {
    'model': rf,
    'symptoms': symptom_list,
    'symptom_to_idx': symptom_to_idx,
    'classes': rf.classes_.tolist()
}

out_path = os.path.join(OUTPUT_DIR, "symptom_disease_model.joblib")
joblib.dump(model_data, out_path, compress=3)
print(f"      Model saved to: {out_path}")
print("\nSymptom-to-Disease model training complete!")
