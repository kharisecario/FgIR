# -*- coding: utf-8 -*-
"""
Created on Thr Sep 20 14:49:05 2024
@author: khari
The second file in collecting jdx IR data.
This file work by separating absorption spectra files.

Dependency:
-shutil
"""
### Import all required dependencies
import os, re, time, shutil
#import pandas as pd
directories = ['jdx_files/IR_spec_all', 'jdx_files/absorbance']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

start_time = time.time()
src_directory = 'jdx_files/IR_spec_all'
dst_directory = 'jdx_files/absorbance'
for filename in os.listdir(src_directory):
    filepath = os.path.join(src_directory,filename)
    with open(filepath, 'r') as infile:
        content = infile.read()
        if re.search('STATE=GAS', content, re.IGNORECASE):
            if re.search('YUNITS=ABSORBANCE', content, re.IGNORECASE):
                shutil.copy(filepath, dst_directory)
                pass
            pass
        infile.close()

end_time = time.time()
print (f"Elapsed time to selecting gas absorbance data: {round(end_time - start_time, 2)} seconds")
