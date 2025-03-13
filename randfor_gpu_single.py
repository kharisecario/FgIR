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
'''
# To compensate pickle: (not working)
X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()
'''

# Create Random forest model
rf_model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=34)     #, n_streams=1
rf_model.fit(X_train, Y_train)  # train our model

# Prediction based on the test set
Y_pred = rf_model.predict(X_test)
Y_pred

# Model Accuracy
cu_accuracy = cuml.metrics.accuracy_score(Y_test, Y_pred)
print( " cuml accuracy: ", cu_accuracy )

# To save the model

import joblib
joblib.dump(rf_model, "kmeans_model.pkl")
rf_model = None # To close the model so that there is no attribute error


'''
# CALCULATE the importance
importance = rf_model.feature_importances_
col_feature = X.columns
i=0
while i < len(col_feature):
    print(f"The importance of feature '{col_feature[i]}' is {round(importances[i]*100, 2)}%.")
    i += 1
'''

# [OK] option 1: use cuml param to generate sklearn model and importance
# limitation: model created by sklearn may differ with cuml (need to check. how?)
'''
from sklearn.ensemble import RandomForestClassifier as sk_rf
from sklearn.metrics import accuracy_score, classification_report

sk_model = sk_rf(n_estimators=100, max_features='sqrt', random_state=34)
sk_model.fit(X_train.to_numpy(), Y_train.to_numpy())  # train our model

sk_predict = sk_model.predict(X_test.to_numpy())
sk_accuracy = accuracy_score(Y_test.to_numpy(), sk_predict)
print( " skml accuracy: ", sk_accuracy )

importances = sk_model.feature_importances_
col_feature = X.columns
i = 0

while i < len(col_feature):
    print(f"The importance of feature '{col_feature[i]}' is {round(importances[i]*100, 2)}%.")
    i += 1
'''


# [Failed in memory usage] Option 2: use permutation importance
# problem >>> CUDA out of memory when doing permutation importance, also very slow.
'''
from sklearn.inspection import permutation_importance
#### Data from df need to be converted to numpy array before training & fitting
X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()
result = permutation_importance(rf_model, X_train, Y_train, scoring='accuracy') #    values.reshape(-1,).tolist()

# Display feature importances
importances = result.importances_mean
for i, importance in enumerate(importances):
    print(f"Feature {i}: Importance {importance}")
'''