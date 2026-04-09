# -*- coding: utf-8 -*-
"""
Created on Thr Sep 24 2024 10:29:05 
Updated on Thr Apr 09 2026
@author: khari

Dependency:
- scipy for spinal cubic interpolation
- pandas
- numpy

known issue:
- This version use stripped inchi as an input. so "InChI=" are added again. Next version should ignore adding that strings (probably won't do it anyway).

"""
### Import all required dependencies
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Fragments

####################
#   Main program   #
####################

inchi = []
with open('INCHIlabel.txt') as infile:
    for inline in infile:
        inchi.append(f'InChI={inline.split()[0]}')
    infile.close()

### Create a placeholder
func_gr_num = {}
func_gr_present = {}
fragment_list = []
with open('fragments_rdkit.txt') as infile:
    for inline in infile:
        fragment_list.append(inline.split()[0])
        func_gr_num[inline.split()[0]] = []
        func_gr_present[inline.split()[0]] = []
    infile.close()

n=1
### check through all inchi, for each fragments:
for each_mol in inchi:
    print(f"Current molecule: {n}", end='\r', flush=True)
    #print(each_mol)
    #if n == 2399:  #remove problematic INCHI
        #break
    mol = Chem.MolFromInchi(each_mol, sanitize=False)   #sanitize false to avoid error when inchi have problem.
    for func_group in fragment_list:
        call_func = f'fr_{func_group}'
        obtain_func_group = getattr(rdkit.Chem.Fragments, call_func)
        total_presence = obtain_func_group(mol)
        func_gr_num[func_group].append(total_presence)
        if total_presence >= 1:
            func_gr_present[func_group].append(1)
        else:
            func_gr_present[func_group].append(0)
    n+=1

### collect to pandas df and write to file
func_gr_num_DF = pd.DataFrame.from_dict(func_gr_num)
func_gr_present_DF = pd.DataFrame.from_dict(func_gr_present)
func_gr_num_DF.to_csv('label_fg_count.csv', index=False, header=True)
func_gr_present_DF.to_csv('label_fg_present.csv', index=False, header=True)



