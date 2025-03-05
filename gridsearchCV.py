# -*- coding: utf-8 -*-
"""
Created on Wed Mar 05 2025 12:07:13 
@author: khari

This script is for running cross validation manually.

Dependency:
- cuml from Rapids
- Pandas
- Sklearn
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize model
model = LogisticRegression(max_iter=200)

# Lists to store results
train_scores = []
val_scores = []

# Manual cross-validation loop
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on training and validation sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Store results
    train_scores.append(train_accuracy)
    val_scores.append(val_accuracy)

# Print results
print(f"Training accuracies: {train_scores}")
print(f"Validation accuracies: {val_scores}")
print(f"Average training accuracy: {np.mean(train_scores):.2f}")
print(f"Average validation accuracy: {np.mean(val_scores):.2f}")