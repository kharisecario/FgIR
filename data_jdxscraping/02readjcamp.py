# -*- coding: utf-8 -*-
"""
Created on Thr Sep 06 14:49:05 2024
@author: khari

Dependency:
- pandas

"""
### Import all required dependencies
import os, re
directory = '../testabs'
import pandas as pd

### Collection of Functions
def find_matching_string(line, string_list):
    for string in string_list:
        if string in line:
            return string
    return None

def extract_ydata(line, file):
    ydata = []
    next(file)
    while "##END" not in line:
        ydata.append(line.split(' ')[1:])
        next(file)
    return ydata

def collect_xdata(firstx, lastx, deltax):
    xdata = [firstx]
    while firstx <= lastx:
        firstx += firstx + deltax
        xdata.append(firstx)
    return xdata


####################
#   Main program   #
####################

### Create pandas dataframe for storing important informations
mol_info = ['TITLE', 'CAS REGISTRY NO', 'MOLFORM', 'XUNITS', 'YUNITS', 'YFACTOR', 'DELTAX', 'FIRSTX', 'LASTX', 'XDATA', 'YDATA']
moldataset_df = pd.DataFrame(columns=mol_info)


for filename in os.listdir(directory):
    filepath = os.path.join(directory,filename)
    if os.path.isfile(filepath):
        with open(filename, 'r') as infile:
            molinfo_df = pd.DataFrame(columns=mol_info)
            for line in infile:
                # strip leading and trailing whitespace
                stripped_line = line.strip()
                if stripped_line.startswith('##'):
                    #if "TITLE" in stripped_line:
                    matching_string = find_matching_string(line, mol_info[:-2])
                    if matching_string is not None:
                        molinfo_df[matching_string] = re.sub(r'^.*?=', '', line)
                    elif 'XYDATA' in line:
                        molinfo_df.at[0, 'YDATA'] = extract_ydata(line, infile)
            infile.close()
            molinfo_df.at[0, 'XDATA'] = collect_xdata(molinfo_df.at[0,'FIRSTX'], molinfo_df.at[0,'LASTX'], molinfo_df.at[0,'DELTAX'])
    moldataset_df.append(molinfo_df, ignore_index=True)

print(moldataset_df)

