# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 18:27:04 2024
@author: khari

prerequisite(s):
- nistchempy 

known issue:
2024/09/18
    - Not able to use indextoskip

"""
import os,re, time
import nistchempy as nist

directories = ['jdx_files/','jdx_files/IR_spec_all/']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

start_time = time.time()
alldata_df = nist.get_all_data()
log_file_path = 'log_scraping.txt'
### Use for continuing scraping from last time it disconnected
if os.path.exists(log_file_path):
    with open(log_file_path) as logfile:
        for line in logfile:
            alldata_df = alldata_df.drop(index=[x for x in range(int(line)+1)])
        logfile.close()
alldata_df = alldata_df.loc[(alldata_df['inchi'] != '<NA>') & (alldata_df['cIR'] == True)]                     #obtain only species with IR and inchi
print(alldata_df)

end_time = time.time()
print (f"Elapsed time to collect all NIST data: {round(end_time - start_time, 2)} seconds")
print (f"Total data obtained: {len(alldata_df)} molecules")

### collect all the inchi in dict
### currently disabled to avoid missing data when gathering IR spectrum
#inchidata_df = {}
start_time = time.time()
for index,row in alldata_df.iterrows():
    print(f"Processing molecule number {index} ...")
    nistID = nist.Compound(row['ID'])                                                #obtain info of species with NIST ID
    nistID.get_ir_spectra()                                                          #obtain list of all its IR spectra
    with open('jdx_files/inchidata_all.txt', 'a+') as file:
        file.write(f'{str(nistID)}\t{row['inchi']}\n')
        file.close()
    for spec in nistID.IR:
        with open(f'jdx_files/IR_spec_all/{str(nistID)}_{spec.spec_idx}.jdx', 'w+') as file:
            file.write(spec.jdx_text)
            file.close()
    del nistID
    with open(log_file_path, 'w+') as logfile:
        logfile.write(f'{index}')
        logfile.close()
end_time = time.time()
print (f"Elapsed time to repopulate absorbance data: {round(end_time - start_time, 2)} seconds")
print('remove log file...')
os.remove('log_scraping.txt')

