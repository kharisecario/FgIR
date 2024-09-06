# -*- coding: utf-8 -*-
"""
Created on Thr Sep 05 14:09:05 2024
@author: khari

Dependency:
- scipy for spinal cubic interpolation

"""
### Import all required dependencies
import os
directory = '../absorbancedata'
#from scipy.interpolate import CubicSpline
#import pandas as pd


### Defining functions used in this file
def ReadAbsJDX (filename):
    with open(filename, 'r') as file:
        for line in file:
            # strip leading and trailing whitespace
            stripped_line = line.strip()
            if stripped_line.startswith('##'):
                break
    return


####################
#   Main program   #
####################

for filename in os.listdir(directory):
    filepath = os.path.join(directory,filename)
    if os.path.isfile(filepath):
        ReadAbsJDX(filepath)
