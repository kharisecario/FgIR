# -*- coding: utf-8 -*-
"""
Created on Thr Feb 24 2025 11:02:13 
@author: khari

This script is for running single label random forest using cuml

Dependency:
- cuml from Rapids
- Pandas

"""
### Import all required dependencies
import cuml
from cuml import RandomForestClassifier
from cuml.model_selection import train_test_split
import pandas as pd

### Defining functions used in this file


####################
#   Main program   #
####################
# import dataset
X = pd.read_csv('XYsnv.csv') #XYnorm XYsnv XYsnvnorm
dfy = pd.read_csv('label_fg_present.csv')
fg_name = dfy.columns.tolist()
Y = dfy[fg_name[1]]    #only 1 label for test

# Split the dataset into train test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=34, stratify=Y)

# Create Random forest model
rf_model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=34, n_streams=1)
rf_model.fit(X_train, Y_train)  # train our model

# Prediction based on the test set
Y_pred = rf_model.predict(X_test)
Y_pred

# Model Accuracy
cu_accuracy = cuml.metrics.accuracy_score(Y_test, Y_pred)
print( " cuml accuracy: ", cu_accuracy )

# To save the model
'''
import pickle
pickle.dump(rf_model, open("kmeans_model.pkl", "wb"))
'''


# CALCULATE the importance
importance = rf_model.feature_importances_
col_feature = X.columns
i=0
while i < len(col_feature):
    print(f"The importance of feature '{col_feature[i]}' is {round(importances[i]*100, 2)}%.")
    i += 1
