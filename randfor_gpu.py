# -*- coding: utf-8 -*-
"""
Created on Thr Feb 24 2025 11:02:13 
@author: khari

This script is for LOOPING single label random forest using cuml

Dependency:
- cuml from Rapids
- Pandas

"""
### Import all required dependencies
import cuml
from cuml import RandomForestClassifier
from cuml.model_selection import train_test_split
import pandas as pd
import time

### Defining functions used in this file


####################
#   Main program   #
####################
# import dataset
X = pd.read_csv('XYsnv.csv') #XYnorm XYsnv XYsnvnorm
dfy = pd.read_csv('label_fg_present.csv')

model_accuracy = {}

# Remove fg that has <2 appearance
less_appear_label = ['alkyl_carbamate','barbitur','benzodiazepine','diazo','dihydropyridine','guanido','lactam','prisulfonamd','quatN','tetrazole']
for i in less_appear_label:
    del dfy[i]

# Check for every available funct group >>>
start_time = time.time()
for fg_name in dfy.columns.to_list():
    Y = dfy[fg_name]

    # Split dataset to train-test >>>
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=34, stratify=Y)

    # Create Random forest model
    rf_model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=34, n_streams=1)
    print(f'Now training for {fg_name} ...')
    rf_model.fit(X_train, Y_train)  # train our model

    # Making Prediction
    Y_pred = rf_model.predict(X_test)

    # Model Accuracy
    cu_accuracy = cuml.metrics.accuracy_score(Y_test, Y_pred)
    print(f"    cuml accuracy for {fg_name} >>> {cu_accuracy:.2f}")
    print(f"    elapsed test time >>> {time.time()-start_time}")

    # Record Accuracy in dictionary
    model_accuracy[fg_name] = [round(cu_accuracy,2)]

print(f'>>> Elapsed Time for all FG = {time.time()-start_time} s')

# Finish Loop, save results
#print(model_accuracy)
model_accuracy_df = pd.DataFrame.from_dict(model_accuracy)
model_accuracy_df.to_csv('randfor_accuracy_report.csv', index=False, header=True)


