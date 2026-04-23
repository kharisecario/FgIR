# -*- coding: utf-8 -*-
"""
Created on Thr Apr 23 2026 14:17 
@author: khari

Dependency:
- pandas

Changelog:
> Created as a part of separating reusable functions.
"""
### Import all required dependencies
from basefunction import normalize


####################
#   Main program   #
####################
result = normalize('data_522_3844.csv')

result.norm.to_csv("ftir_norm.csv", index=True, header=True)
result.snv.to_csv("ftir_snv.csv",  index=True, header=True)
