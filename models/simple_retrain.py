import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# Load the data
df = pd.read_csv('data/features.csv')
print(f"Loaded {len(df)} records")

# Create quality labels
def create_quality_labels(row):
    word_count = row['word_count']
    readability = row['flesch_reading_ease']
    
    if word_count >= 1500 and readability >= 30:
        return 'High'
    elif word_count >= 800 or (word_count >= 500 and readability >= 40):
        return 'Medium'
    else:
        return 'Low'

df['quality_label'] = df.apply(create_quality_labels, axis=1)
print("Quality distribution:")
print(df['quality_label'].value_counts())

# Prepare features
X = df[['word_count', 'sentence_count', 'flesch_reading_ease']].fillna(0)
y = df['quality_label']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Save model
os.makedirs('models', exist_ok=True)
with open('models/quality_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("Model retrained and saved successfully!")

# Test
accuracy = rf_model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
