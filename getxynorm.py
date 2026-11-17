# -*- coding: utf-8 -*-
"""
Changelog:
Thr Apr 09 2026 
> Changing normalization to x/xmax to preserve the baseline.



Created on Thr Feb 05 2025 13:13:13 
Updated on Thr Apr 09 2026
@author: khari

Dependency:
- pandas
- numpy (removed)

"""
### Import all required dependencies
import pandas as pd

### Defining functions used in this file
# Standard normal variate where mean 0 and std 1 (sample is is based on row)
def snv(df): 
    return (df - df.mean()) / df.std()



####################
#   Main program   #
####################
df = pd.read_csv('XYfeature.csv')

# apply normalized raw
xy_norm = df.apply(lambda x: x / x.max(), axis=1) # WARNING THIS ONLY SCALE MAX, works if its absorbance with proper 0 baseline!
#xy_norm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# apply standard normal variate, because based on row, we need to transpose the df first to make each molecules per row.
xy_snv = snv(df.T)


# apply normalized SNV: (the results are similar to SNV, so we do not use it for model training, but just for comparison)
#xy_snv_norm = snv(xy_norm.T)   # we do max normalization first before snv.


# write to file
xy_norm.to_csv('XYnorm.csv', index=False, header=True)
xy_snv.T.to_csv('XYsnv.csv', index=False, header=True)  # Need to be transposed for readability
#xy_snv_norm.T.to_csv('XYsnvnorm.csv', index=False, header=True) # Need to be transposed for readability

