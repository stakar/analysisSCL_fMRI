import numpy as np
import pandas as pd
from analysis_tools import *
from os import listdir

filelist = [n.split('.')[0] for n in listdir('analiza_treatment/') if 'vhdr' in n][:3]
for file in filelist:
    analysis(f'analiza_treatment/{file}.vhdr',f'analiza_treatment/{file}.vmrk',file)
