# -*- coding: utf-8 -*-
"""
Updated on Fri Mar 07 2025 10:30
Created on Thr Feb 24 2025 11:02:13 
@author: khari

This script is for running single label random forest using cuml
gridsearchCV done using kfold with F1-score drop

Dependency:
- cuml from Rapids
- Pandas

"""
### Import all required dependencies
import cuml
from cuml import RandomForestClassifier
from cuml.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from cuml.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import roc_auc_score as sklearn_roc
import pandas as pd
#import cudf as pd
import cupy as cp
from numpy import mean, std
import time

# Supress user warning >>> UserWarning: To use pickling first train using float32 data to fit the estimator ret = func(*args, **kwargs)
import warnings
warnings.filterwarnings("ignore")

### Defining functions used in this file
# Create a dictionary of all permutation of hyperparameters
import itertools
def combine_dict_values(input_dict):
    keys = input_dict.keys()
    values = input_dict.values()
    combinations = list(itertools.product(*values))
    
    result = []
    for combination in combinations:
        result.append(dict(zip(keys, combination)))
    
    return result



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
param_grid = {'max_depth':[5,10],
              'max_features':['log2','sqrt'],
              #'max_leaf_nodes':[None,2,3,4,5],     # not supported in cuML
              }

###########################################

# Hyperparameters tuning
best_hyperparameter = []
#CV_record = {}
f1_record = {}
combinations = combine_dict_values(param_grid)
skf = StratifiedKFold(n_splits=3, random_state=34)
f1_drop_best = 1
f1_test_best = 0

#'''
# Change the training df to numpy
numpy_array = X_train.to_numpy()
cupy_x_array = cp.asarray(numpy_array)
numpy_array = Y_train.to_numpy()
cupy_y_array = cp.asarray(numpy_array)

#print(cupy_array)
#'''


for combo in combinations:
    print(combo)
    #print(tuple(combo.values()))
    # Generate Cross Validation fold:
    f1_drop_temp = []
    f1_cvtest_temp = []
    for train_index, test_index in skf.split(X_train, Y_train):     # can only use normal dataframe
        #print(cupy_array[train_index])
        XCV_train, XCV_test = cupy_x_array[train_index], cupy_x_array[test_index]     # index here need use cupy array
        YCV_train, YCV_test = cupy_y_array[train_index], cupy_y_array[test_index]
        #XCV_train, XCV_test = X_train[train_index], X_train[test_index]     # index here need use cupy array
        #YCV_train, YCV_test = Y_train[train_index], Y_train[test_index]

        rf_model = RandomForestClassifier(**combo, random_state=34, n_streams=1)
        rf_model.fit(XCV_train, YCV_train)

        cv_test_pred = rf_model.predict(XCV_test)
        cv_train_pred = rf_model.predict(XCV_train)
        cv_test_accuracy = accuracy_score(YCV_test, cv_test_pred)
        cv_train_accuracy = accuracy_score(YCV_train, cv_train_pred)

        # F1 Score
        f1_test = f1_score(YCV_test.get(), cv_test_pred.get())     #already ndarray, but need use get as it is in gpu
        print(f"kfold f1 test: {f1_test}")
        f1_train = f1_score(YCV_train.get(), cv_train_pred.get())
        #f1_test = f1_score(YCV_test.to_numpy(), cv_test_pred.to_numpy())
        #f1_train = f1_score(YCV_train.to_numpy(), cv_train_pred.to_numpy())
        f1_drop = abs(f1_train - f1_test)

        '''
        cvy_score =rf_model.predict_proba(XCV_test)
        rocauccv = roc_auc_score(YCV_test, cvy_score[:,1])
        print(f"kfold roc auc: {rocauccv}")
        '''
        #collect data
        f1_drop_temp.append(f1_drop)
        f1_cvtest_temp.append(f1_test)

    # make a record
    f1_record[tuple(combo.values())] = f1_drop_temp

    if mean(f1_drop_temp) < f1_drop_best:
        f1_drop_best = mean(f1_drop_temp)
        f1_drop_std = std(f1_drop_temp)
        best_hyperparameter = combo
        f1_cvtest_best = mean(f1_cvtest_temp)
        f1_cvtest_std = std(f1_cvtest_temp)

print(f"Best Parameters: {best_hyperparameter}")
print(f"Best F1_drop: {f1_drop_best} ± {f1_drop_std}")
print(f"Best F1_test: {f1_cvtest_best} ± {f1_cvtest_std}")



# Test best hyperparameter
rf_model = RandomForestClassifier(**best_hyperparameter, random_state=34, n_streams=1, oob_score=True)
rf_model.fit(X_train, Y_train)
print(f"OOB_score: {rf_model.oob_score_}")
'''
print(f">>> printing feature importances...")
#rf_model.feature_importances_.to_csv('trial_feature_importances.csv', index=True)      #its numpy not pandas.
pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).to_csv('trial_feature_importances.csv', index=False)
'''

Y_pred = rf_model.predict(X_test)
Y_pred_train = rf_model.predict(X_train)
cu_accuracy = accuracy_score(Y_test, Y_pred)
#cv_train_accuracy = accuracy_score(Y_train, Y_pred_train)

#obtain yscore
X_test.to_csv('trial_X_test.csv')
y_score =rf_model.predict_proba(X_test)
#y_score.to_csv('trial_y_score.csv')
select_y_score = y_score[1]
#print(f"Y_test type: {type(Y_test)}")
#print(f"select y score type: {type(select_y_score)}")
rocauc = roc_auc_score(Y_test, select_y_score)
print(f"rocauc: {rocauc}")
'''
#just to check. theres only negligible differences between cuml and sklearn.
sklearn_result = sklearn_roc(Y_test, select_y_score) #no need asnumpy!
sklearn_result = sklearn_roc(cp.asnumpy(Y_test), cp.asnumpy(select_y_score)) 
print(f"sklearn: {sklearn_result:.6f}")
print(f"cuML:    {rocauc:.6f}")
print(f"Diff:    {abs(rocauc - sklearn_result):.8f}")
'''

# F1 Score
f1_test = f1_score(Y_test.to_numpy(), Y_pred.to_numpy())     #already ndarray, but need use get as it is in gpu
f1_train = f1_score(Y_train.to_numpy(), Y_pred_train.to_numpy())
f1_drop = abs(f1_train - f1_test)

print(f"RF Accuracy: {cu_accuracy}")
print(f"F1_drop Test Result: {f1_drop}")


####################################################

cm = confusion_matrix(Y_test.to_numpy(), Y_pred.to_numpy())
cm_df = pd.DataFrame(cm.get(), index=[0,1], columns=[0,1])
print(f"Confusion Matrix: {cm}")
#cm_df.to_csv(f'confusion_matrix/cm_{fg_name}.csv', index=True)

cr = classification_report(Y_test.to_numpy(), Y_pred.to_numpy(), output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
print(f"Classification Report: {cr_df}")
#cr_df.to_csv(f'classification_report/cr_{fg_name}.csv', index=True)