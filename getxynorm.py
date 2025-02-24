# -*- coding: utf-8 -*-
"""
Created on Thr Feb 05 2025 13:13:13 
@author: khari

Dependency:
- pandas
- numpy

"""
### Import all required dependencies
import pandas as pd
import numpy as np

### Defining functions used in this file
#Standard normal variate where mean 0 and std 1 (sample is is based on row)
def snv(row): 
    mean = np.mean(row)
    std_dev = np.std(row)
    return(row-mean)/std_dev

####################
#   Main program   #
####################
df = pd.read_csv('XYfeature.csv')

#apply standard normal variate
xy_snv = df.apply(snv,axis=1)

#apply normalized SNV:
xy_snv_norm = xy_snv.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

#apply normalized raw
xy_norm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

#write to file
xy_snv.to_csv('XYsnv.csv', index=False, header=True)
xy_snv_norm.to_csv('XYsnvnorm.csv', index=False, header=True)
xy_norm.to_csv('XYnorm.csv', index=False, header=True)

