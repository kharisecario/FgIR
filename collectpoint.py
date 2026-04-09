# -*- coding: utf-8 -*-
"""
Created on Thr Sep 10 2024 10:29:05 
Updated on Thr Apr 09 2026
@author: khari

Dependency:
- scipy for spinal cubic interpolation
- pandas
- numpy

known issues:
- When inchi label used, it give error due to stripped "InChI="
  Next fix should consider keeping those strings.

"""
### Import all required dependencies
import readjcamp, re
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

### Prepare inchi database (as dictionary):
#inchi_all = pd.read_csv('jdx_files/inchidata_all.txt', sep='\t', header=None)      ### Make it as panda generate more problem when matching!
#inchi_all.columns = ['FILENAME', 'INCHI']
inchi_all = {}
with open('jdx_files/inchidata_all.txt') as infile:
    for inline in infile:
        (key, val) = inline.split()
        inchi_all[key] = re.sub(r'^.*?=', '', val)

### prepare original database:
moldataset_df = readjcamp.obtain_molinfo()

### Define new x-axis for interpolation
xmax = 3844
xmin = 552
npoints = 1647
xrange = 2
xinterpolate = [xmax]
ncounts = 1

while ncounts < npoints:
    xmax -= xrange
    xinterpolate.append(xmax)
    ncounts += 1

### Prepare placeholder
xinterpolate = np.array(xinterpolate)
yinterpolate = np.zeros(shape=(len(moldataset_df.index), len(xinterpolate)))
#INCHIlabel = np.zeros(shape=(len(moldataset_df.index),1))
INCHIlabel = []

### Generate interpolate Y
for index, row in moldataset_df.iterrows():
    yinterpolate[index,:] = get_Yinterpolate(row, xinterpolate)
    INCHIlabel.append(inchi_all[moldataset_df['FILENAME'][index]])


XYfeature_df = pd.DataFrame(yinterpolate, columns=xinterpolate)

### Write into file:
XYfeature_df.to_csv('XYfeature.csv', index=False, header=True)
print(INCHIlabel)
with open('INCHIlabel.txt', 'w+') as infile:
    for item in INCHIlabel:
        infile.write(f'{item}\n')
    infile.close()

