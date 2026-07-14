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
import os, time, re, shutil
from pathlib import Path            # For creating backup
from datetime import datetime       # For creating backup
from joblib import load
import pandas as pd


### Defining functions used in this file
directory = 'pickled_model'
RFprediction = {}
RFproba = {}

# ─── Prepare a backup for consecutive output file ───────────────
files_to_backup = ['RF_prediction.csv', 'RF_probability.csv']
backup_dir = Path('backups')
backup_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for file in files_to_backup:
    file_path = Path(file)
    if file_path.exists():
        backup_path = backup_dir / f"{file_path.stem}_{timestamp}.bak"
        shutil.copy2(file_path, backup_path)

####################
#   Main program   #
####################
### Import X data as dataframe
X = pd.read_csv('ftir_snv.csv', index_col=0) #XYnorm XYsnv XYsnvnorm

### Prepare all pickled model:
start_time = time.time()
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
