# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:06:04 2023
@author: khari

prerequisite(s):
- nistchempy 

known issue:
2024/09/18
    - Not able to use indextoskip

"""
import os,re, time
import nistchempy as nist
#conditions = ['##STATE=gas','##YUNITS=TRANSMITTANCE']

#directory = 'jdx_files/'
directories = ['jdx_files/','jdx_files/absorbance/']
#os.makedirs(os.path.dirname(filename), exist_ok=True)
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

### check index to skip:
'''
skip_file_path = 'indextoskip.txt'
if os.path.exists(skip_file_path):
    skiplist = [int(x) for x in open(skip_file_path).read().splitlines()]
'''

start_time = time.time()
alldata_df = nist.get_all_data()
log_file_path = 'log_scraping.txt'
### Use for continuing scraping from last time it disconnected
if os.path.exists(log_file_path):
    with open(log_file_path) as logfile:
        for line in logfile:
            alldata_df = alldata_df.drop(index=[x for x in range(int(line)+1)])
        logfile.close()
#alldata_df = alldata_df.loc[(alldata_df['cas_rn'] != '<NA>') & (alldata_df['cIR'] == True)]                     #obtain only species with IR and CAS
alldata_df = alldata_df.loc[(alldata_df['inchi'] != '<NA>') & (alldata_df['cIR'] == True)]                     #obtain only species with IR and inchi
### drop index that included in skip list
#alldata_df = alldata_df.drop(index=skiplist)
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
    for spec in nistID.IR:
        ### this part is updated case INsensitive
        if re.search('STATE=GAS', spec.jdx_text, re.IGNORECASE):
            if re.search('YUNITS=ABSORBANCE', spec.jdx_text, re.IGNORECASE):
                print("Absorbance found... Writing into file...")
                with open(f'jdx_files/absorbance/{str(nistID)}_{spec.spec_idx}.jdx', 'w+') as file:
                    file.write(spec.jdx_text)
                    file.close()
                with open('jdx_files/inchidata_abs.txt', 'a+') as file:
                    file.write(f'{str(nistID)}_{spec.spec_idx}.jdx\t{row['inchi']}\n')
                    file.close()
                with open('jdx_files/absorbanceindex.txt', 'a+') as file:
                    file.write(f'{index}\n')
                    file.close()
                #inchidata_df[nistID] = alldata_df['inchi']         #currently disabled
            '''
            else:
                with open('jdx_files/indextoskip.txt', 'a+') as file:
                    file.write(f'{index}\n')
                    file.close()
            '''
            '''
            # ignore transmittance:
            elif re.search('YUNITS=TRANSMITTANCE', spec.jdx_text, re.IGNORECASE):
                with open('jdx_files/transmittance/'+str(nistID)+'_'+spec.spec_idx+'.jdx', 'w+') as file:
                    file.write(spec.jdx_text)
                    file.close()
            '''
        '''
        else: ### Use this to record list of the compound that dont have IR absorbance spectrum
            with open(f'jdx_files/indextoskip.txt', 'a+') as file:
                file.write(f'{index}\n')
                file.close()
        '''
    del nistID
    with open(log_file_path, 'w+') as logfile:
        logfile.write(f'{index}')
        logfile.close()
end_time = time.time()
print (f"Elapsed time to repopulate absorbance data: {round(end_time - start_time, 2)} seconds")
print('remove log file...')
os.remove('log_scraping.txt')


#currently disabled
'''
with open('jdx_files/absorbance/inchidata.txt', 'w+') as file:
    for key, value in inchidata_df.items():
        file.write('%s \t %s \n' % (key, value))
    file.close()
'''
