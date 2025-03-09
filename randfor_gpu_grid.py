# -*- coding: utf-8 -*-
"""
Created on Sun Mar 09 2025 14:15 
@author: khari

This script is for LOOPING single label random forest using cuml
with gridsearchCV evaluated with F1-score drop

Dependency:
- cuml from Rapids
- pandas
- cupy
- sklearn
- numpy

"""
### Import all required dependencies
import cuml
from cuml import RandomForestClassifier
from cuml.model_selection import train_test_split, StratifiedKFold
from cuml.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import cupy as cp
from numpy import mean
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
# Prepare blank placeholder
model_performance = {}

# import dataset
X = pd.read_csv('XYsnv.csv') #XYnorm XYsnv XYsnvnorm
dfy = pd.read_csv('label_fg_present.csv')

# Remove fg that has <2 appearance
less_appear_label = ['alkyl_carbamate','barbitur','benzodiazepine','diazo','dihydropyridine','guanido','lactam','prisulfonamd','quatN','tetrazole']
for i in less_appear_label:
    del dfy[i]
# Remove fg that has <3% appeareance
less_appear_label = ['pyridine', 'C_S', 'amide', 'Ar_COO', 'isothiocyan', 'aldehyde', 'nitro_arom_nonortho', 'sulfide', 'SH', 'Ar_NH', 'Nhpyrrole', 'thiophene', 'term_acetylene', 'Ndealkylation1', 'imidazole', 'furan', 'imide', 'lactone', 'Ndealkylation2', 'hdrzone', 'piperdine', 'isocyan', 'sulfone', 'epoxide', 'sulfonamd', 'thiazole', 'oxime', 'morpholine', 'amidine', 'azo', 'phos_acid', 'phos_ester', 'hdrzine', 'nitroso', 'piperzine', 'thiocyan', 'urea', 'HOCCN', 'priamide', 'N_O', 'oxazole', 'azide']
for i in less_appear_label:
    del dfy[i]

# Collect hyperparameters for tuning:
param_grid = {'max_depth':[5,10,15],
              'max_features':['log2','sqrt'],
              #'max_leaf_nodes':[None,2,3,4,5],     # not supported in cuML
              }



# Start evaluating every available funct group >>>
start_time = time.time()
for fg_name in dfy.columns.to_list():
    print(f"Evaluating {fg_name}... ... ...")
    loop_time = time.time()
    Y = dfy[fg_name]

    # Split dataset to train-test >>>
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=34, stratify=Y)

    # HYPERPARAMETER TUNING: >>>
    best_hyperparameter = []
    f1_record = {}
    combinations = combine_dict_values(param_grid)
    skf = StratifiedKFold(n_splits=3, random_state=34)
    f1_drop_best = 1

    # Change the training df to numpy then cupy as required for it to work
    numpy_array = X_train.to_numpy()
    cupy_x_array = cp.asarray(numpy_array)
    numpy_array = Y_train.to_numpy()
    cupy_y_array = cp.asarray(numpy_array)

    # GRIDSEARCH CROSS-VALIDATION for all combinations:
    for combo in combinations:

        f1_drop_temp = []
        for train_index, test_index in skf.split(X_train, Y_train):     # can only use normal dataframe
            XCV_train, XCV_test = cupy_x_array[train_index], cupy_x_array[test_index]     # index here need use cupy array
            YCV_train, YCV_test = cupy_y_array[train_index], cupy_y_array[test_index]

            rf_model = RandomForestClassifier(**combo)
            rf_model.fit(XCV_train, YCV_train)

            cv_test_pred = rf_model.predict(XCV_test)
            cv_train_pred = rf_model.predict(XCV_train)
            cv_test_accuracy = accuracy_score(YCV_test, cv_test_pred)
            cv_train_accuracy = accuracy_score(YCV_train, cv_train_pred)

            # F1 Score
            f1_test = f1_score(YCV_test.get(), cv_test_pred.get())     #already ndarray, but need use get as it is in gpu
            f1_train = f1_score(YCV_train.get(), cv_train_pred.get())
            f1_drop = abs(f1_train - f1_test)
            
            #collect data
            f1_drop_temp.append(f1_drop)

        # make a record
        f1_record[tuple(combo.values())] = f1_drop_temp

        if mean(f1_drop_temp) < f1_drop_best:
            f1_drop_best = mean(f1_drop_temp)
            best_hyperparameter = combo

    print(f">>> Best Parameters: {best_hyperparameter}")
    print(f">>> Best CV F1_drop: {f1_drop_best:.2f}")
    

    # EVALUATE BEST HYPERPARAMETER: >>>
    print(f">>> Testing the model...")
    # Test best hyperparameter
    rf_model = RandomForestClassifier(**best_hyperparameter)
    rf_model.fit(X_train, Y_train)

    # Accuracy
    Y_pred = rf_model.predict(X_test)
    Y_pred_train = rf_model.predict(X_train)
    cu_accuracy = accuracy_score(Y_test, Y_pred)

    # F1 Score
    f1_test = f1_score(Y_test.to_numpy(), Y_pred.to_numpy())     #already ndarray, but need use get as it is in gpu
    f1_train = f1_score(Y_train.to_numpy(), Y_pred_train.to_numpy())
    f1_drop = abs(f1_train - f1_test)

    print(f">>> RF Accuracy: {cu_accuracy:.2f}")
    print(f">>> F1_drop Test Result: {f1_drop:.2f}")
    
    print(f"    elapsed test time >>> {time.time()-loop_time}")
    model_performance[fg_name] = [round(cu_accuracy,2), round(f1_drop,2)]


print(f'>>> Elapsed Time for all FG = {time.time()-start_time} s')

# Finish Loop, save results
model_performance_df = pd.DataFrame.from_dict(model_performance)
model_performance_df.to_csv('randfor_performance_report.csv', index=False, header=True)


