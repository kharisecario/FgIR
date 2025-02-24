# -*- coding: utf-8 -*-
"""
Changelog:
Mon Feb 24 2025 16:34
> Adjusting snv definition to remove numpy dependency
> CSV output for the new version hasn't been checked!


Created on Thr Feb 05 2025 13:13:13 
@author: khari

Dependency:
- pandas
- numpy (removed)

"""
### Import all required dependencies
import pandas as pd

### Defining functions used in this file
#Standard normal variate where mean 0 and std 1 (sample is is based on row)
def snv(df): 
    return (df - df.mean()) / df.std()

####################
#   Main program   #
####################
df = pd.read_csv('XYfeature.csv')

#apply standard normal variate
xy_snv = snv(df.T)

#apply normalized SNV:
#xy_snv_norm = xy_snv.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
xy_snv_norm = xy_snv.apply(lambda x: (x - x.min()) / (x.max() - x.min()))   # axis=1 are removed as the snv df now transposed

#apply normalized raw
xy_norm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

#write to file
xy_snv.T.to_csv('XYsnv.csv', index=False, header=True)  # Need to be transposed for readability
xy_snv_norm.T.to_csv('XYsnvnorm.csv', index=False, header=True) # Need to be transposed for readability
xy_norm.to_csv('XYnorm.csv', index=False, header=True)

