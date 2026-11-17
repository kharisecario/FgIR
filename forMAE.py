# -*- coding: utf-8 -*-
"""
MAE per functional-group (FG) model and mean MAE across all FG models.
"""
import os, re
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error

directory = 'pickled_model_use'

# import dataset
X = pd.read_csv('XYsnv.csv') #XYnorm XYsnv XYsnvnorm
Y = pd.read_csv('label_fg_present.csv')

# Remove fg that has <6 appearance
less_appear_label = ['alkyl_carbamate','barbitur','benzodiazepine','diazo','dihydropyridine',
                     'guanido','lactam','quatN','tetrazole']
for i in less_appear_label:
    del Y[i]
# Remove fg that has <1% appeareance
less_appear_label = ['HOCCN', 'azide', 'N_O', 'oxazole', 'priamide', 
                     'amidine', 'azo', 'epoxide', 'furan', 'hdrzine',
                     'hdrzone', 'imidazole', 'imide', 'isocyan', 'lactone',
                     'morpholine', 'Ndealkylation1', 'Ndealkylation2', 'nitroso', 'oxime', 
                     'phos_acid', 'phos_ester', 'piperdine', 'piperzine', 'prisulfonamd',
                     'sulfonamd', 'sulfone', 'thiazole', 'thiocyan', 'urea']
for i in less_appear_label:
    del Y[i]

# Predict and compute MAE for each FG model
mae = {}
for filename in os.listdir(directory):
    FGname = re.sub(r'cuml_', '', os.path.splitext(filename)[0])
    if FGname not in Y.columns:
        print(f"    skipped {FGname}: no matching column in labels")
        continue

    model = load(os.path.join(directory, filename))
    y_pred = np.asarray(model.predict(X))
    mae[FGname] = mean_absolute_error(Y[FGname].to_numpy(), y_pred)

# Report
mae_series = pd.Series(mae, name='MAE')
print(mae_series.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\nMean MAE across {len(mae_series)} FG models: {mae_series.mean():.4f}")