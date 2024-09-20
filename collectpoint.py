# -*- coding: utf-8 -*-
"""
Created on Thr Sep 10 2024 10:29:05 
@author: khari

Dependency:
- scipy for spinal cubic interpolation
- pandas
- numpy

"""
### Import all required dependencies
import os
import readjcamp
from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np


### Defining functions used in this file
def get_Yinterpolate(row_df, xcs):
    cs = CubicSpline(np.array(row_df['XDATA']), np.array(row_df['YDATA']))
    ycs = cs(xcs)
    return ycs


####################
#   Main program   #
####################
moldataset_df = readjcamp.obtain_molinfo()

### Define new x-axis for interpolation
xmax = 3844
xmin = 552
npoints = 1647
xrange = 2
xinterpolate = [xmax]
ncounts = 1

'''
while xmax >= xmin:
    xmax -= xrange
    xinterpolate.append(xmax)
'''
while ncounts < npoints:
    xmax -= xrange
    xinterpolate.append(xmax)
    ncounts += 1

xinterpolate = np.array(xinterpolate)

yinterpolate = np.zeros(shape=(len(moldataset_df.index), len(xinterpolate)))
for index, row in moldataset_df.iterrows():
    yinterpolate[index,:] = get_Yinterpolate(row, xinterpolate)

XYinterpolate_df = pd.DataFrame(yinterpolate, columns=xinterpolate)
#print(XYinterpolate_df)

XYinterpolate_df.to_csv('XY_interpolation.csv', index=False, header=True)