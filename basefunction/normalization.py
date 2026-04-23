# -*- coding: utf-8 -*-
"""
Created on Thr Apr 23 2026 14:17 
@author: khari

Dependency:
- pandas

Changelog:
> Created as a part of separating reusable functions.
"""

# ─── Import all required dependencies ──────────────────
import pandas as pd

# ─── Defining functions used in this file ───────────────
# Standard normal variate where mean 0 and std 1 (sample is is based on row)

class NormResult:
    def __init__(self, data):
        self._df = data
        self._norm = None
        self._snv = None

    @property
    def norm(self):
        if self._norm is None:
            row_max = self._df.max(axis=1)
            self._norm = self._df.div(row_max, axis=0)
        return self._norm
    
    @property
    def snv(self):
        if self._snv is None:
            row_mean = self._df.mean(axis=1)
            row_std = self._df.std(axis=1)
            self._snv = self._df.sub(row_mean, axis=0).div(row_std, axis=0)
        return self._snv

####################
#   Main program   #
####################
def normalize(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    return NormResult(df)

### Usage:
# out = normalize('file.csv')
# out.snv
# out.norm

