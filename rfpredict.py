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
    #print(pred_result)
    FGname = re.sub(r'cuml_', '', os.path.splitext(filename)[0])
    RFprediction[FGname] = pred_result

    # To close the model so that there is no attribute error
    rf_model = None 

    print(f"    time for predict {FGname} >>> {(time.time()-loop_time):.3f} s")


print(f"    Total elapsed time >>> {(time.time()-start_time):.3f} s")
RFprediction_df = pd.DataFrame.from_dict(RFprediction)
RFprediction_df.index = X.index.to_list()
RFprediction_df.to_csv('RF_prediction.csv', index=True, header=True)

