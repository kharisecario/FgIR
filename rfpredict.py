# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 2025
@author: khari

Dependency:
- 

known issues:
- 

"""
### Import all required dependencies
import os, time, re
from joblib import load
import pandas as pd


### Defining functions used in this file
directory = 'pickled_model'
RFprediction = {}
RFproba = {}

####################
#   Main program   #
####################
start_time = time.time()

### Import X data as dataframe
X = pd.read_csv('ftir_snv.csv', index_col=0) #XYnorm XYsnv XYsnvnorm

### Prepare all pickled model:
for filename in os.listdir(directory):
    loop_time = time.time()
    #filepath = os.path.join(directory, filename)
    model = load(os.path.join(directory, filename))
    pred_result = model.predict(X)
    pred_probs = model.predict_proba(X)[1]        # Added for calculate prediction probabilty
    #print(pred_result)
    FGname = re.sub(r'cuml_', '', os.path.splitext(filename)[0])
    RFprediction[FGname] = pred_result
    RFproba[FGname] = pred_probs        # Added for calculate prediction proba

    # To close the model so that there is no attribute error
    rf_model = None 

    print(f"    time for predict {FGname} >>> {(time.time()-loop_time):.3f} s")


print(f"    Total elapsed time >>> {(time.time()-start_time):.3f} s")
RFprediction_df = pd.DataFrame.from_dict(RFprediction)
RFprediction_df.index = X.index.to_list()
RFprediction_df.to_csv('RF_prediction.csv', index=True, header=True)

# Save probability data
RFproba_df = pd.DataFrame.from_dict(RFproba)
RFproba_df.index = X.index.to_list()
RFproba_df.to_csv('RF_probability.csv', index=True, header=True)
