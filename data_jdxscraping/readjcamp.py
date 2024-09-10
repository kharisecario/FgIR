# -*- coding: utf-8 -*-
"""
Created on Thr Sep 06 14:49:05 2024
@author: khari
The second file in collecting jdx IR data.

Dependency:
- pandas

"""
### Import all required dependencies
import os, re, time
directory = '../absorbancedata'
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
        #ydata.append(line.split(' ')[1:])
        ydata = ydata + [float(x) for x in line.strip().split(' ')[1:]]
        line = next(file)
    return ydata

'''
def collect_xdata(minx, maxx, deltax):
    xdata = [minx]
    while minx < maxx:
        #minx += minx + "{:.3f}".format(deltax)
        minx += round(deltax, 3)
        xdata.append(round(minx, 3))
    return xdata
'''

def collect_xdata(minx, npoints, deltax):
    xdata = [minx]
    countern = 1
    while countern < npoints:
        #minx += minx + "{:.3f}".format(deltax)
        minx += round(deltax, 3)
        xdata.append(round(minx, 3))
        countern += 1
    return xdata

def XYdata_type(input_dict):
    # define data that need to be converted
    float_data = ['DELTAX', 'MINX', 'NPOINTS']
    for i in float_data:
        input_dict[i] = float(input_dict[i])    
    return input_dict


####################
#   Main program   #
####################

### Create pandas dataframe for storing important informations
#mol_info = ['TITLE', 'CAS REGISTRY NO', 'MOLFORM', 'XUNITS', 'YUNITS', 'YFACTOR', 'DELTAX', 'FIRSTX', 'LASTX', 'XDATA', 'YDATA'] #using first and last X cause error
mol_info = ['TITLE', 'CAS REGISTRY NO', 'MOLFORM', 'XUNITS', 'YUNITS', 'YFACTOR', 'DELTAX', 'MAXX', 'MINX', 'NPOINTS', 'XDATA', 'YDATA']
moldataset_df = pd.DataFrame(columns=mol_info)

start_time = time.time()
for filename in os.listdir(directory):
    if not filename.endswith(':Zone.Identifier'):
        filepath = os.path.join(directory,filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as infile:
                #molinfo_df = pd.DataFrame(columns=mol_info)
                singlemol_info = dict.fromkeys(mol_info, None)
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
                            #ydata = extract_ydata(line, infile)
                            singlemol_info['YDATA'] = extract_ydata(line, infile)
                #singlemol_info['XDATA'] = collect_xdata(molinfo_df.at[0,'FIRSTX'], molinfo_df.at[0,'LASTX'], molinfo_df.at[0,'DELTAX'])
                XYdata_type(singlemol_info)
                singlemol_info['XDATA'] = collect_xdata(singlemol_info['MINX'], singlemol_info['NPOINTS'], singlemol_info['DELTAX'])
            infile.close()
        #moldataset_df.append(molinfo_df, ignore_index=True) #append DEPRECATED
        moldataset_df = pd.concat([moldataset_df, pd.DataFrame([singlemol_info])], ignore_index=True)

#print(moldataset_df)
end_time = time.time()
print (f"Elapsed time to create dataset: {round(end_time - start_time, 2)} seconds")

### Write dataframe to CSV tab-delimiter
#moldataset_df.to_csv('moldataset_df.csv', sep='\t', encoding='utf-8', index=True, header=True)

def obtain_molinfo():
    return moldataset_df