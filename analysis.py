import numpy as np
import pandas as pd
from analysis_tools import *
from os import listdir

#check the time, for counting time necessary for analysis.
curr = time.time()

#Get list of files from time point 1
filelist = [n.split('.')[0] for n in listdir('analiza_treatment/') \
            if ('vhdr' in n and 'tp1' in n.lower())]

#preprocess each file from time point 1, if not present.
for file in filelist:
    try:        
        if f"{file}.xlsx" not in listdir('preprocessed_treatment/'):
            analysis(f'analiza_treatment/{file}.vhdr',
                     f'analiza_treatment/{file}.vmrk',
                     f'preprocessed_treatment/{file}')
    except Exception as e:
        print(e)
        print(f'Problem with {file}')
        

#Get list of files from time point 5
filelist = [n.split('.')[0] for n in listdir('analiza_treatment/') if \
            ('vhdr' in n and 'tp5' in n.lower())]
#preprocess each file from time point 5, if not present.
for file in filelist:
    try:
        if f"{file}.xlsx" not in listdir('preprocessed_treatment/'):
            analysis(f'analiza_treatment/{file}.vhdr',
                     f'analiza_treatment/{file}.vmrk',
                     f'preprocessed_treatment/{file}')
    except Exception as e:
        print(e)
        print(f'Problem with {file}')
        
        
#Get list of files from time point 6
filelist = [n.split('.')[0] for n in listdir('analiza_follow up/') if \
            ('acq' in n and 'tp6' in n.lower())]
#preprocess each file from time point 6, if not present.
for file in filelist:
    try:
        if f"{file}.xlsx" not in listdir('preprocessed_fup/'):
            analysis_acknowledge(f'analiza_follow up/{file}.acq',
                             f'analiza_follow up/{file}.acq',
                             f'preprocessed_fup/{file}')
    except Exception as e:
        print(e)
        print(f'Problem with {file}')
        
#Get list of files from time point 7
filelist = [n.split('.')[0] for n in listdir('analiza_follow up/') \
            if ('acq' in n and 'tp7' in n.lower())]
#preprocess each file from time point 7, if not present.
for file in filelist:
    try:
        if f"{file}.xlsx" not in listdir('preprocessed_fup_TP7/'):
            analysis_acknowledge(f'analiza_follow up/{file}.acq',
                                 f'analiza_follow up/{file}.acq',
                                 f'preprocessed_fup_TP7/{file}')
    except Exception as e:
        print(e)
        print(f'Problem with {file}')
        
        
#check post-time, count difference and show how much time was spent.
curr2 = time.time()
diff_time = np.round(curr2-curr)
print(f'Job was done after {diff_time}')
