# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:06:04 2023
@author: khari

prerequisite(s):
- nistchempy 

"""
import os,re
import nistchempy as nist
conditions = ['##STATE=gas','##YUNITS=TRANSMITTANCE']

#directory = 'jdx_files/'
directories = ['jdx_files/','jdx_files/absorbance/','jdx_files/transmittance/','jdx_files/testtrans/']
#os.makedirs(os.path.dirname(filename), exist_ok=True)
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

df = nist.get_all_data()
#df_casir = df.loc[(df['cas_rn'] != '<NA>') & (df['cIR'] == True)]
df = df.loc[(df['cas_rn'] != '<NA>') & (df['cIR'] == True)]                     #obtain only species with IR and CAS

#print(df.cas_rn.iloc[2])                                                       #TEST
for index,row in df.iterrows():
    #print(row['cas_rn'])                                                       
    X = nist.Compound(row['ID'])                                                #obtain info of species with NIST ID
    #print(X)
    X.get_ir_spectra()                                                          #obtain list of all its IR spectra
    #print(X.IR)
    for spec in X.IR:
        ''' #this part is works! but transmittance is empty
        if '##STATE=gas' in spec.jdx_text:
            if '##YUNITS=TRANSMITTANCE' in spec.jdx_text:
                with open('jdx_files/transmittance/'+str(row['cas_rn'])+'_'+spec.spec_idx+'.jdx', 'w+') as file:
                    file.write(spec.jdx_text)
                    file.close()
            elif '##YUNITS=ABSORBANCE' in spec.jdx_text:
                with open('jdx_files/absorbance/'+str(row['cas_rn'])+'_'+spec.spec_idx+'.jdx', 'w+') as file:
                    file.write(spec.jdx_text)
                    file.close()
        '''
        ''' #this part only for transmittance gas
        if re.search('YUNITS=TRANSMITTANCE', spec.jdx_text, re.IGNORECASE):
            with open('jdx_files/testtrans/'+str(row['cas_rn'])+'_'+spec.spec_idx+'.jdx', 'w+') as file:
                file.write(spec.jdx_text)
                file.close()
        '''
        #this part is updated case INsensitive
        if re.search('STATE=GAS', spec.jdx_text, re.IGNORECASE):
            if re.search('YUNITS=TRANSMITTANCE', spec.jdx_text, re.IGNORECASE):
                with open('jdx_files/transmittance/'+str(row['cas_rn'])+'_'+spec.spec_idx+'.jdx', 'w+') as file:
                    file.write(spec.jdx_text)
                    file.close()
            elif re.search('YUNITS=ABSORBANCE', spec.jdx_text, re.IGNORECASE):
                with open('jdx_files/absorbance/'+str(row['cas_rn'])+'_'+spec.spec_idx+'.jdx', 'w+') as file:
                    file.write(spec.jdx_text)
                    file.close()
        

    del X

