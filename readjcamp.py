# -*- coding: utf-8 -*-
"""
Created on Thr Sep 06 14:49:05 2024
Updated on Thr Apr 09 2026
@author: khari


Dependency:
- pandas

"""
### Import all required dependencies
import os, re, time
directory = 'jdx_files/absorbance'
import pandas as pd

### Collection of Functions
def find_matching_string(line, string_list):
    for string in string_list:
        if string in line:
            return string
    return None

def extract_ydata(line, file):
    ydata = []
    line = next(file)
    while "##END" not in line:
        ydata = ydata + [float(x) for x in line.strip().split(' ')[1:]]
        line = next(file)
    return ydata

def XYdata_type(input_dict):
    ### define data format that need to be converted
    float_data = ['DELTAX', 'MAXX', 'MINX']
    for i in float_data:
        if input_dict[i] != None:
            input_dict[i] = float(input_dict[i])
    input_dict['NPOINTS'] = int(input_dict['NPOINTS'])
    return input_dict

def collect_xdata(input_dict):
    ### delta = (lastx-firstx)/(npoints-1)
    if input_dict['DELTAX'] == None:
        input_dict['DELTAX'] = (input_dict['MAXX'] - input_dict['MINX'])/(input_dict['NPOINTS'] - 1)
    
    minx = round(input_dict['MINX'], 3)
    xdata = [minx]
    countern = 1
    while countern < input_dict['NPOINTS']:
        minx += round(input_dict['DELTAX'], 3)
        xdata.append(minx)
        countern += 1
    return xdata

####################
#   Main program   #
####################

### Create pandas dataframe for storing important informations
#mol_info = ['TITLE', 'CAS REGISTRY NO', 'MOLFORM', 'XUNITS', 'YUNITS', 'YFACTOR', 'DELTAX', 'FIRSTX', 'LASTX', 'XDATA', 'YDATA'] #using first and last X cause error
mol_info = ['FILENAME', 'TITLE', 'CAS REGISTRY NO', 'MOLFORM', 'XUNITS', 'YUNITS', 'YFACTOR', 'DELTAX', 'MAXX', 'MINX', 'NPOINTS', 'XDATA', 'YDATA']
moldataset_df = pd.DataFrame(columns=mol_info)

start_time = time.time()
for filename in os.listdir(directory):
    if not filename.endswith(':Zone.Identifier'):
        filepath = os.path.join(directory,filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as infile:
                molinfo_df = pd.DataFrame(columns=mol_info)
                singlemol_info = dict.fromkeys(mol_info, None)
                singlemol_info['FILENAME'] = re.sub(r'_.*', '', filename)
                for line in infile:
                    # strip leading and trailing whitespace
                    stripped_line = line.strip()
                    if stripped_line.startswith('##'):
                        #if "TITLE" in stripped_line:
                        matching_string = find_matching_string(line, mol_info[:-2])
                        #if matching_string is not None:
                        if matching_string != None:
                            singlemol_info[matching_string] = re.sub(r'^.*?=', '', line.strip()) #remove all chars up to = sign to obtain the value
                        elif 'XYDATA' in line:
                            singlemol_info['YDATA'] = extract_ydata(line, infile)
                #print(type(singlemol_info['YFACTOR']))
                #break
                singlemol_info['YDATA'] = list(map(lambda x: x * float(singlemol_info['YFACTOR']), singlemol_info['YDATA']))    # Multiply YDATA with YFACTOR
                XYdata_type(singlemol_info)
                singlemol_info['XDATA'] = collect_xdata(singlemol_info)
            infile.close()
        #moldataset_df.append(molinfo_df, ignore_index=True) #append DEPRECATED
        moldataset_df = pd.concat([moldataset_df, pd.DataFrame([singlemol_info])], ignore_index=True)

end_time = time.time()
### Write dataframe to CSV tab-delimiter
print(moldataset_df)
print (f"Elapsed time to create dataset: {round(end_time - start_time, 2)} seconds")
print('writing to file...')
moldataset_df.to_csv('moldataset_df.txt', sep='\t', encoding='utf-8', index=False, header=True)
moldataset_df.to_csv('moldataset_df.csv', sep='\t', encoding='utf-8', index=False, header=True)

### This part to be used with collectpoint.py
def obtain_molinfo():
    return moldataset_df