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
from cuml.model_selection import train_test_split, GridSearchCV
import pandas as pd
import time

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
start_time = time.time()
param_grid = {'max_depth':[5,10,15],
              'max_features':['log2','sqrt'],
              'max_leaf_nodes':[None,2,3,4,5],
              }
rf_model = GridSearchCV(RandomForestClassifier(), param_grid)
#rf_model.fit(X_train, Y_train)  # train our model, get() used to construct numpy array explicitly
rf_model.fit(X_train.to_numpy(), Y_train.to_numpy())

# Get the best parameters and best score
best_params = rf_model.best_params_
best_score = rf_model.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")
#print(rf_model.cv_results_)

print (f"Elapsed time {time.time()-start_time}")

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