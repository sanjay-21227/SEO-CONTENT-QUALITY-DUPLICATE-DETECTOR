#!/usr/bin/env python3
"""
Retrain the SEO Content Quality Model
This script retrains the model to fix compatibility issues with current scikit-learn version
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def retrain_model():
    """Retrain the quality model using existing feature data"""
    
    print("🔄 Retraining SEO Content Quality Model...")
    
    # Load existing feature data
    try:
        df = pd.read_csv('../data/features.csv')
        print(f"✅ Loaded {len(df)} records from features.csv")
    except Exception as e:
        print(f"❌ Error loading features.csv: {e}")
        return False
    
    # Create quality labels based on word count and readability
    def create_quality_labels(row):
        word_count = row['word_count']
        readability = row['flesch_reading_ease']
        
        if word_count >= 1500 and readability >= 30:
            return 'High'
        elif word_count >= 800 or (word_count >= 500 and readability >= 40):
            return 'Medium'
        else:
            return 'Low'
    
    # Apply quality labeling
    df['quality_label'] = df.apply(create_quality_labels, axis=1)
    
    print("📊 Quality Distribution:")
    print(df['quality_label'].value_counts())
    
    # Prepare features and labels
    feature_columns = ['word_count', 'sentence_count', 'flesch_reading_ease']
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    
    X = df[feature_columns].fillna(0)
    y = df['quality_label']
    
    print(f"📈 Features shape: {X.shape}")
    print(f"🎯 Labels shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🔀 Training set: {X_train.shape[0]} samples")
    print(f"🔀 Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest model
    print("🌳 Training Random Forest model...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Accuracy: {accuracy:.3f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_dir = '../models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'quality_model.pkl')
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"✅ Model saved to: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def test_model():
    """Test the retrained model"""
    print("\n🧪 Testing the retrained model...")
    
    try:
        with open('../models/quality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Test with sample data
        test_features = np.array([
            [1500, 85, 45],  # High quality
            [800, 40, 35],   # Medium quality
            [300, 15, 20]    # Low quality
        ])
        
        predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features)
        
        print("📊 Test Predictions:")
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            print(f"Sample {i+1}: {pred} (confidence: {max(proba):.3f})")
        
        print("✅ Model loaded and tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Model Retraining Process...")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = retrain_model()
    
    if success:
        test_success = test_model()
        if test_success:
            print("\n🎉 Model retraining completed successfully!")
            print("You can now run the Streamlit app without compatibility issues.")
        else:
            print("\n⚠️ Model retrained but testing failed.")
    else:
        print("\n❌ Model retraining failed.")
