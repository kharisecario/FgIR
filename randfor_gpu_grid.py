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
import os, time, joblib
from cuml import RandomForestClassifier
from cuml.model_selection import train_test_split, StratifiedKFold
from cuml.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, classification_report, average_precision_score
import pandas as pd
import cupy as cp
from numpy import mean, std

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

# Create directory for output
out_dir = ['pickled_model', 'confusion_matrix', 'CV_report']
for directory in out_dir:
    if not os.path.exists(directory):
        os.makedirs(directory)


####################
#   Main program   #
####################
# Prepare blank placeholder
model_performance = {}
model_feature_importance = {}

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
    cv_record = {}

    # Split dataset to train-test >>>
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=34, stratify=Y)

    # HYPERPARAMETER TUNING: >>>
    best_hyperparameter = []
    combinations = combine_dict_values(param_grid)
    skf = StratifiedKFold(n_splits=3, random_state=34)
    #cv_f1_drop_best = 1
    cv_pr_auc_best = 0

    # Change the training df to numpy then cupy as required for it to work
    numpy_array = X_train.to_numpy()
    cupy_x_array = cp.asarray(numpy_array)
    numpy_array = Y_train.to_numpy()
    cupy_y_array = cp.asarray(numpy_array)

    # GRIDSEARCH CROSS-VALIDATION for all combinations:
    for combo in combinations:
        cv_f1_drop_temp = []
        cv_f1_test_temp = []
        cv_pr_auc_temp = []
        
        for train_index, test_index in skf.split(X_train, Y_train):     # can only use normal dataframe
            XCV_train, XCV_test = cupy_x_array[train_index], cupy_x_array[test_index]     # index here need use cupy array
            YCV_train, YCV_test = cupy_y_array[train_index], cupy_y_array[test_index]

            rf_model = RandomForestClassifier(**combo, random_state=34, n_streams=1)
            rf_model.fit(XCV_train, YCV_train)

            cv_test_pred = rf_model.predict(XCV_test)
            cv_train_pred = rf_model.predict(XCV_train)
            cv_test_accuracy = accuracy_score(YCV_test, cv_test_pred)
            cv_train_accuracy = accuracy_score(YCV_train, cv_train_pred)

            # F1 Score
            cv_f1_test = f1_score(YCV_test.get(), cv_test_pred.get())     #already ndarray, but need use get as it is in gpu
            cv_f1_train = f1_score(YCV_train.get(), cv_train_pred.get())
            cv_f1_drop = abs(cv_f1_train - cv_f1_test)
            
            #PR-AUC
            ycv_score = rf_model.predict_proba(XCV_test)
            cv_pr_auc = average_precision_score(YCV_test.get(), ycv_score[:,1].get())

            #collect data
            cv_f1_drop_temp.append(cv_f1_drop)
            cv_f1_test_temp.append(cv_f1_test)
            cv_pr_auc_temp.append(cv_pr_auc)

        # make a record
        cv_pr_auc_temp_mean = mean(cv_pr_auc_temp)
        cv_pr_auc_temp_std = std(cv_pr_auc_temp)
        f1_cvtest_temp_mean = mean(cv_f1_test_temp)
        f1_cvtest_temp_std = std(cv_f1_test_temp)
        f1_drop_temp_mean = mean(cv_f1_drop_temp)
        f1_drop_temp_std = std(cv_f1_drop_temp)
        cv_record[str(combo)] = [f"{cv_pr_auc_temp_mean:.3f} ± {cv_pr_auc_temp_std:.3f}", f"{f1_cvtest_temp_mean:.3f} ± {f1_cvtest_temp_std:.3f}", f"{f1_drop_temp_mean:.3f} ± {f1_drop_temp_std:.3f}"]

        if cv_pr_auc_temp_mean > cv_pr_auc_best:
            # Obtain mean and std PR-AUC.
            cv_pr_auc_best = cv_pr_auc_temp_mean
            cv_pr_auc_std = cv_pr_auc_temp_std
            # Obtain mean and std F1-score CV test.
            f1_cvtest_best = f1_cvtest_temp_mean
            f1_cvtest_std = f1_cvtest_temp_std
            # Obtain mean and std F1-score drop of CV test.
            f1_drop_best = f1_drop_temp_mean
            f1_drop_std = f1_drop_temp_std
            # Record best hyperparameter
            best_hyperparameter = combo

    print(f">>> Best Parameters: {best_hyperparameter}")
    #print(f">>> Best CV F1_drop: {cv_f1_drop_best:.2f}")
    print(f">>> Best CV PR-AUC: {cv_pr_auc_best:.2f} ± {cv_pr_auc_std:.2f}")

    # Record CV scores for evaluation
    print(cv_record)
    cv_report = pd.DataFrame.from_dict(cv_record)
    cv_report.index = ['PR-AUC', 'F1-Score', 'F1-drop']
    cv_report.transpose().to_csv(f'CV_report/cm_{fg_name[1]}.csv', index=True)


    ###################################################
    # EVALUATE BEST HYPERPARAMETER: >>>
    print(f">>> Testing the model...")
    # Test best hyperparameter
    rf_model = RandomForestClassifier(**best_hyperparameter, random_state=34, n_streams=1, oob_score=True)
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
    model_performance[fg_name] = [best_hyperparameter, round(cu_accuracy,2), round(f1_test,2), round(f1_drop,2), round(rf_model.oob_score_,2), round(cv_pr_auc_best,2), round(cv_pr_auc_std,2), round(f1_cvtest_best,2), round(f1_cvtest_std,2), round(f1_drop_best,2), round(f1_drop_std,2)]

    # Dumping all important information into file

    cm = confusion_matrix(Y_test.to_numpy(), Y_pred.to_numpy())
    cm_df = pd.DataFrame(cm.get(), index=[0,1], columns=[0,1])
    cm_df.to_csv(f'confusion_matrix/cm_{fg_name}.csv', index=True)

    '''
    # We change how classification report are done in current version, so this parts will be removed
    cr = classification_report(Y_test.to_numpy(), Y_pred.to_numpy(), output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_csv(f'classification_report/cr_{fg_name}.csv', index=True)
    '''

    # Obtain Feature Importance:
    model_feature_importance[fg_name] = rf_model.feature_importances_
    
    #rf_model.feature_importances_.to_csv('trial_feature_importances.csv', index=True)      #its numpy not pandas.
    pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).to_csv('trial_feature_importances.csv', index=False)

    joblib.dump(rf_model, f'pickled_model/cuml_{fg_name}.pkl')


print(f'>>> Elapsed Time for all FG = {time.time()-start_time} s')

# Finish Loop, save results
model_performance_df = pd.DataFrame.from_dict(model_performance)
model_performance_df.index = ['Best Hyperparameter', 'Model Accuracy', 'Model F1_test', 'Model F1_drop', 'Model OOB_Score', 'CV PR-AUC (avg)', 'CV PR-AUC (std)', 'CV F1_Score (avg)', 'CV F1_score (std)', 'CV F1_drop (avg)', 'CV F1_drop (std)']
model_performance_df.transpose().to_csv('randfor_performance.csv', index=True, header=True)

model_feature_importance_df = pd.DataFrame.from_dict(model_feature_importance)
model_feature_importance_df.index = X_train.columns
model_feature_importance_df.to_csv('randfor_performance.csv', index=True, header=True)






# To close the model so that there is no attribute error
rf_model = None 
####################
#   End program    #
####################