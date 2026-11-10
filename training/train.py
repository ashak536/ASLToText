"""
Train a Logistic Regression classifier on ASL fingerspelling landmark data
and export the model weights to JSON for browser-side inference.

Usage:
    python train.py

Reads:  training/data/asl_landmarks.csv
Writes: model.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_data(csv_path):
    """Load landmark CSV and return feature matrix X and label array y."""
    df = pd.read_csv(csv_path)
    y = df['label'].values
    X = df.drop(columns=['label']).values.astype(np.float64)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")
    return X, y


def train_model(X_train, y_train):
    """Train a multinomial Logistic Regression classifier."""
    clf = LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        C=1.0,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def export_model(clf, output_path, decimals=6):
    """Export model weights to JSON for client-side inference."""
    model_dict = {
        'classes': clf.classes_.tolist(),
        'coef': np.round(clf.coef_, decimals).tolist(),
        'intercept': np.round(clf.intercept_, decimals).tolist(),
    }
    with open(output_path, 'w') as f:
        json.dump(model_dict, f)
    file_size = os.path.getsize(output_path) / 1024
    print(f"Model exported to {output_path} ({file_size:.1f} KB)")
    print(f"  Classes: {model_dict['classes']}")
    print(f"  Coefficient matrix: {len(model_dict['coef'])} × {len(model_dict['coef'][0])}")


if __name__ == '__main__':
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'data', 'asl_landmarks.csv')
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, 'model.json')

    # Load
    X, y = load_data(csv_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # Train
    print("\nTraining Logistic Regression...")
    clf = train_model(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Export
    export_model(clf, model_path)
    print("\nDone!")
