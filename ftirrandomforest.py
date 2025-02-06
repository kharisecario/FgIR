# -*- coding: utf-8 -*-
"""
Created on Thr Feb 05 2025 13:13:13 
@author: khari

Dependency:
- pandas
- sklearn randomforest

"""
### Import all required dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

### Defining functions used in this file


####################
#   Main program   #
####################
# Import dataset as df
X = pd.read_csv('XYnorm.csv')
dfy = pd.read_csv('label_fg_present.csv')
fg_y = dfy.columns.tolist()
Y = dfy[fg_y[1]]

# Split dataset by 1/3 for test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=34)

# Generate Classifier Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=34)
rf_model.fit(X_train, Y_train)  # train our model

# Make Prediction based on Test
Y_pred = rf_model.predict(X_test)

# Evaluate Prediction Results
accuracy = accuracy_score(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)