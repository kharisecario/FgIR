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

# Remove fg that has <2 appearance
less_appear_label = ['alkyl_carbamate','barbitur','benzodiazepine','diazo','dihydropyridine','guanido','lactam','prisulfonamd','quatN','tetrazole']
for i in less_appear_label:
    del dfy[i]
# Remove fg that has <3% appeareance
less_appear_label = ['pyridine', 'C_S', 'amide', 'Ar_COO', 'isothiocyan', 'aldehyde', 'nitro_arom_nonortho', 'sulfide', 'SH', 'Ar_NH', 'Nhpyrrole', 'thiophene', 'term_acetylene', 'Ndealkylation1', 'imidazole', 'furan', 'imide', 'lactone', 'Ndealkylation2', 'hdrzone', 'piperdine', 'isocyan', 'sulfone', 'epoxide', 'sulfonamd', 'thiazole', 'oxime', 'morpholine', 'amidine', 'azo', 'phos_acid', 'phos_ester', 'hdrzine', 'nitroso', 'piperzine', 'thiocyan', 'urea', 'HOCCN', 'priamide', 'N_O', 'oxazole', 'azide']
for i in less_appear_label:
    del dfy[i]


start_time = time.time()
for fg_name in dfy.columns.to_list():
    print(f"Evaluating {fg_name}... ... ...")
    loop_time = time.time()
    Y = dfy[fg_name]

    # Split dataset to train-test >>>
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=34, stratify=Y)

    rf_model = sk_rf(n_estimators=100, max_features='sqrt', random_state=34)
    rf_model.fit(X_train.to_numpy(), Y_train.to_numpy())  # train our model
    
    print(f"    elapsed test time >>> {time.time()-loop_time}")

    importances = rf_model.feature_importances_
    importance_value[fg_name] = importances
    
    
cm_df = pd.DataFrame.from_dict(importance_value).set_index(X.columns)
cm_df.to_csv('feature_importance.csv', index=True, header=True)


print(f'>>> Elapsed Time for all FG = {round((time.time()-start_time)/60, 2)} min')
