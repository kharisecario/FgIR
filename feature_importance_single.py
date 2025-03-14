# -*- coding: utf-8 -*-
"""
Created on Thr Mar 13 2025 17:05 
@author: khari secario

This script is for obtaining feature importance (sklearn) 
based on the best parameter obtain through CUML previously.

Dependency:
- cuml from Rapids
- pandas
- cupy
- sklearn
- numpy

"""
### Import all required dependencies
from sklearn.ensemble import RandomForestClassifier as sk_rf
from cuml.model_selection import train_test_split
import pandas as pd
import time


### Defining functions used in this file



####################
#   Main program   #
####################
# Prepare blank placeholder
importance_value = {}

# import dataset
X = pd.read_csv('XYsnv.csv') #XYnorm XYsnv XYsnvnorm
dfy = pd.read_csv('label_fg_present.csv')
fg_name = dfy.columns.tolist()
Y = dfy[fg_name[1]]    #only 1 label for test

# Split dataset to train-test >>>
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=34, stratify=Y)

start_time = time.time()

rf_model = sk_rf(n_estimators=100, max_features='sqrt', random_state=34)
rf_model.fit(X_train.to_numpy(), Y_train.to_numpy())  # train our model

print(f"    elapsed test time >>> {time.time()-start_time}")

importances = rf_model.feature_importances_
col_feature = X.columns
i = 0

importance_value[fg_name[1]] = importances
cm_df = pd.DataFrame.from_dict(importance_value).set_index(X.columns)
#cm_df = cm_df.set_index(X.columns)
cm_df.to_csv('feature_importance.csv', index=True, header=True)


print(cm_df)
'''
while i < len(col_feature):
    print(f"The importance of feature '{col_feature[i]}' is {round(importances[i]*100, 2)}%.")
    i += 1

'''
