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
from sklearn.metrics import f1_score
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
    loop_time = time.time()
    Y = dfy[fg_name]

    # Split dataset to train-test >>>
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=34, stratify=Y)

    # Create Random forest model
    rf_model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=34, n_streams=1)
    print(f'Now training for {fg_name} ...')
    rf_model.fit(X_train, Y_train)  # train our model

    # Making Prediction
    Y_pred_train = rf_model.predict(X_train)    #to calculate F1_train
    Y_pred = rf_model.predict(X_test)

    # Model Accuracy
    cu_accuracy = cuml.metrics.accuracy_score(Y_test, Y_pred)
    print(f"    cuml accuracy for {fg_name} >>> {cu_accuracy:.2f}")

    # Model F1 Score
    f1_test = f1_score(Y_test.to_numpy(), Y_pred.to_numpy())
    print(f"    cuml F1-score for {fg_name} >>> {f1_test:.2f}")

    # F1 Training and accuracy
    train_accuracy = cuml.metrics.accuracy_score(Y_train, Y_pred_train)
    f1_train = f1_score(Y_train.to_numpy(), Y_pred_train.to_numpy())

    # Record Accuracy in dictionary
    print(f"    elapsed test time >>> {time.time()-loop_time}")
    model_accuracy[fg_name] = [round(train_accuracy,2), round(cu_accuracy,2), round(f1_train,2), round(f1_test,2)]

print(f'>>> Elapsed Time for all FG = {time.time()-start_time} s')

# Finish Loop, save results
#print(model_accuracy)
model_accuracy_df = pd.DataFrame.from_dict(model_accuracy)
model_accuracy_df.to_csv('randfor_accuracy_report.csv', index=False, header=True)


