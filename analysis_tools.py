import mne
import bioread
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from pandas import DataFrame, Index, read_excel
import pandas as pd
from os import listdir
from scipy import stats
import re

ORIGINF = 5000 #Original frequency
FREQ=100 #Frequency after resampling (source freq: 5kHz, resampling 50x lower to 100Hz)
import numpy as np 
import pandas as pd
from analysis_tools import *
from os import listdir
import time
import re
from scipy import stats
ORIGINF=5000
FREQ=100

def get_ssa(sig):
    """
    Get Slope Sign Alterations
    See Baksys repo for original version.
    """
    res1 = []
    res2 = []
    for n in range(len(sig)-2):
        uno = (sig[n-1]-sig[n])
        if uno!=0:
            uno = uno/np.abs(sig[n-1]-sig[n])
            res1.append(uno)
        dos = (sig[n+1]-sig[n])
        if dos!=0:
            dos = dos/np.abs(sig[n+1]-sig[n])
            res2.append(dos)
    # in case of 0 at preceding, but not at following
    # sample
    if len(res1)>len(res2):
        res1.pop()
    return np.sum(np.abs(np.array(res1)+np.array(res2))/2)

def get_slope(arr):
    """
    Calculate mean slope of signal
    """
    return stats.linregress(np.arange(0,len(arr)),arr)[0]
def get_slope_2_points(arr):
    """
    Calculate slope between begining and the end of signal
    """
    return stats.linregress(np.array([0,1]), np.array([arr[0],arr[-1]]))[0]

def calc_normalized_signal_change(signal,preceding_signal):
    """
    Calculate signal with:
    100✕(SCLStim-SCLbaseline/SCLbaseline)
    where SCLStim is the mean signal value during the 
    stimulus and SCLbaseline is an SCL reaction 
    during the baseline preceding the first part in 
    each scenario (Sugimine et al. 2020).
    """
    stimSCL = np.mean(signal)
    stimBSL = np.mean(preceding_signal)
    return 100*((stimSCL-stimBSL)/stimBSL)
def get_normalized_signal_change(signal,
                                 point,
                                 time_window=27*100,
                                 time_window_preceding=3*100,
                                baseline = None):
    partSig = signal[point:(point+time_window)]
    if baseline==None:
        precedSig = signal[point-time_window_preceding:point]
    else:
        precedSig = baseline
    return calc_normalized_signal_change(partSig,precedSig)

def get_acq_medfiltered_signal(filename_signal, 
                               filename_events, 
                               output, 
                               freq=FREQ, 
                               originf=ORIGINF, 
                               cutoff=2, 
                               correction=1,
                               time_window_analyzed=27
                              ):
    """
    Conduct analysis- open file, preprocess data (resample, filter with lowpass filter, 
    conduct median smoothing, get stimuli time points and calculate results.
    """
    print(f'opening {filename_signal}...')
    data = convert_acq_mne(filename_signal)
    resampled = data.copy().resample(sfreq=freq).get_data()[0]
    filtered = mne.filter.filter_data(resampled,freq,None,cutoff)
    medfiltered=medfilt(filtered,int((FREQ/2)-1))
    stimlist = create_stimlist_file_from_data(filename_events)
    stimarr = get_stimulus_array(stimlist)
    stimnames = get_stimulus_names(stimlist)
    new_stims= resample_events(stimarr,originf/freq)
    return medfiltered,new_stims,stimnames

def plot_acq_medfiltered_signal(filename_signal, 
                               filename_events, 
                               output, 
                               freq=FREQ, 
                               originf=ORIGINF, 
                               cutoff=2, 
                               correction=1,
                               time_window_analyzed=27
                              ):
    """
    Conduct analysis- open file, preprocess data (resample, filter with lowpass filter, 
    conduct median smoothing, get stimuli time points and calculate results.
    """
    print(f'opening {filename_signal}...')
    data = convert_acq_mne(filename_signal)
    resampled = data.copy().resample(sfreq=freq).get_data()[0]
    filtered = mne.filter.filter_data(resampled,freq,None,cutoff)
    medfiltered=medfilt(filtered,int((FREQ/2)-1))
    stimlist = create_stimlist_file_from_data(filename_events)
    stimarr = get_stimulus_array(stimlist)
    stimnames = get_stimulus_names(stimlist)
    new_stims= resample_events(stimarr,originf/freq)
    plt.plot([n for n in range(0,len(resampled),1)],resampled)
    
    plt.plot([n for n in range(0,len(resampled),1)],filtered)
    
    plt.plot([n for n in range(0,len(resampled),1)],medfiltered)
    for stim in new_stims:
        plt.axvspan(stim-100, stim+100, facecolor='r',alpha=0.5)
    plt.xlim(100000,110000)
    return medfiltered,new_stims,stimnames

def calculate_save_results(signal,
                           events,
                           events_times,
                           filename_signal,
                           output,
                           freq=FREQ,
                           time_window_analyzed=27,
                           output_aggr = 'output2',
                          parts=['1','2','3','4','123'],
                          conditions=['crit','neut'],
                          regexp_set=f'S 11[]$'):
    """
    Conduct analysis- open file, preprocess data (resample, filter with lowpass filter, 
    conduct median smoothing, get stimuli time points and calculate results.
    """
    print('calculating results...')
    for part in parts:
        for condition in conditions:
            if condition == 'crit':
                regexp = fr'S [12345][{part}]$'
            elif condition=='neut':
                regexp = fr'S [6789][{part}]$|S 10[{part}]$'
            else:
                regexp = regexp_set
            triggers = list(map(lambda x: bool(re.match(regexp, x)), events))
            triggers = events_times[triggers][0::2]
            
            result,avg = calculate_results(signal,triggers,freq,time_window_analyzed)
            result = DataFrame(result, 
                               index = pd.Index([re.search(r'B\d+',filename_signal)[0]]))
            avg = pd.DataFrame(avg,columns=['averaged_signal'])
            try:
                with pd.ExcelWriter(f'{output}.xlsx',
                                mode='a',
                                if_sheet_exists='overlay') as writer:  
                    result.to_excel(writer, 
                                sheet_name=f'{condition}_{part}')  
                    avg.to_excel(writer, 
                             sheet_name=f'{condition}_{part}_avg')
            except FileNotFoundError:
                result.to_excel(f'{output}.xlsx', 
                                sheet_name=f'{condition}_{part}')  
                with pd.ExcelWriter(f'{output}.xlsx',
                                mode='a',
                                if_sheet_exists='overlay') as writer: 
                    avg.to_excel(writer, 
                             sheet_name=f'{condition}_{part}_avg')
            try:
                with pd.ExcelWriter(f'{output_aggr}.xlsx',
                                    mode='a',
                                    if_sheet_exists='overlay') as writer:  
                    avg.to_excel(writer, 
                                 sheet_name=f'{condition}_{part}_avg',
                                 header=False)
                    result.to_excel(writer, 
                                    sheet_name=f'{condition}_{part}',
                                    startrow=writer.sheets[f'{condition}_{part}'].max_row,
                                    header=False)
            except FileNotFoundError:
                result.to_excel(f'{output_aggr}.xlsx', 
                                sheet_name=f'{condition}_{part}')
                with pd.ExcelWriter(f'{output_aggr}.xlsx',
                                    mode='a',
                                    if_sheet_exists='overlay') as writer:  
                    avg.to_excel(writer, 
                                 sheet_name=f'{condition}_{part}_avg',
                                 header=False)
            except KeyError:
                with pd.ExcelWriter(f'{output_aggr}.xlsx',
                                                mode='a',
                                                if_sheet_exists='overlay') as writer:  
                    avg.to_excel(writer, 
                                 sheet_name=f'{condition}_{part}_avg',
                                 header=False)
                    result.to_excel(writer, 
                                    sheet_name=f'{condition}_{part}',
                                    header=False)
                    
def get_bv_medfiltered_signal(filename_signal,
                              filename_events,
                              output,
                              freq=FREQ,
                              originf=ORIGINF,
                              cutoff=2,
                              correction=1,
                             time_window_analyzed=27):
    """
    Conduct analysis- open file, preprocess data (resample, filter with lowpass filter, 
    conduct median smoothing, get stimuli time points and calculate results.
    """
    print(f'opening {filename_signal}...')
    data = mne.io.read_raw_brainvision(filename_signal)
    resampled = data.copy().resample(sfreq=freq).get_data().flatten()
    filtered = mne.filter.filter_data(resampled,freq,None,cutoff)
    medfiltered=medfilt(filtered,int((freq/2)-1))
    stimlist = get_stimlist(filename_events)
    stimarr = get_stimulus_array(stimlist)
    stimnames = get_stimulus_names(stimlist)
    new_stims= resample_events(stimarr,originf/freq)
    return medfiltered,new_stims,stimnames

def get_bv_zscore_signal(filename_signal,
                              filename_events,
                              output,
                              freq=FREQ,
                              originf=ORIGINF,
                              cutoff=2,
                              correction=1,
                             time_window_analyzed=27,
                               medfilt_kernel = int((FREQ/2)-1)):
    """
    Conduct analysis- open file, preprocess data (resample, filter with lowpass filter, 
    conduct median smoothing, get stimuli time points and calculate results.
    """
    print(f'opening {filename_signal}...')
    data = mne.io.read_raw_brainvision(filename_signal)
    resampled = data.copy().resample(sfreq=freq).get_data().flatten()
    filtered = mne.filter.filter_data(resampled,freq,None,cutoff)
    medfiltered=medfilt(filtered,medfilt_kernel)
    stimlist = get_stimlist(filename_events)
    stimarr = get_stimulus_array(stimlist)
    stimnames = get_stimulus_names(stimlist)
    new_stims= resample_events(stimarr,originf/freq)
    zscored = stats.zscore(medfiltered)
    return medfiltered,new_stims,stimnames,zscored

def get_acq_zscore_signal(filename_signal,
                              filename_events,
                              output,
                              freq=FREQ,
                              originf=ORIGINF,
                              cutoff=2,
                              correction=1,
                             time_window_analyzed=27,
                               medfilt_kernel = int((FREQ/2)-1)):
    """
    Conduct analysis- open file, preprocess data (resample, filter with lowpass filter, 
    conduct median smoothing, get stimuli time points and calculate results.
    """
    print(f'opening {filename_signal}...')
    data = convert_acq_mne(filename_signal)
    resampled = data.copy().resample(sfreq=freq).get_data().flatten()
    filtered = mne.filter.filter_data(resampled,freq,None,cutoff)
    medfiltered=medfilt(filtered,medfilt_kernel)
    stimlist = create_stimlist_file_from_data(filename_events)
    stimarr = get_stimulus_array(stimlist)
    stimnames = get_stimulus_names(stimlist)
    new_stims= resample_events(stimarr,originf/freq)
    zscored = stats.zscore(medfiltered)
    return medfiltered,new_stims,stimnames,zscored

def calculate_results(signal,triggers,freq,time_window_analyzed,bsl_ind=None):
    plc = np.zeros([len(triggers),freq*time_window_analyzed])
    for n in range(len(triggers)):
        plc[n] = signal[triggers[n]:(triggers[n]+freq*time_window_analyzed)]
    avg = np.mean(plc,axis=0)
    if bsl_ind==None:
        mnsc =  np.mean([get_normalized_signal_change(signal,n) for n in triggers])
    else:
        baseline = signal[bsl_ind-3*freq:bsl_ind]
        mnsc = np.mean([get_normalized_signal_change(signal,n) for n in triggers])
    return ({'mean_slope' : np.mean([get_slope(plc[n]) for n in range(len(plc))]),
            'mean_normalized_signal_change' : mnsc,
            'mean_sig_sign_alteration' : np.mean([get_ssa(plc[n]) for n in range(len(plc))]),
            'mean_slope_2pt' : np.mean([get_slope_2_points(plc[n]) for n in range(len(plc))]),
           'mean_signal' : np.mean([np.mean(plc[n]) for n in range(len(plc))]),
           'mean_max_signal' : np.mean([np.max(plc[n]) for n in range(len(plc))]),
           'mean_min_signal' : np.mean([np.min(plc[n]) for n in range(len(plc))])},
           avg)
            

                    
def calculate_save_results_bsl(signal,
                           events,
                           events_times,
                           filename_signal,
                           output,
                           freq=FREQ,
                           time_window_analyzed=27,
                           output_aggr = 'output2',
                          parts=['1','2','3','4','123'],
                          conditions=['crit','neut'],
                          regexp_set=f'S 11[]$'):
    """
    Conduct analysis- open file, preprocess data (resample, filter with lowpass filter, 
    conduct median smoothing, get stimuli time points and calculate results.
    """
    print('calculating results...')
    for part in parts:
        for condition in conditions:
            if condition == 'crit':
                regexp = fr'S [12345][{part}]$'
            elif condition=='neut':
                regexp = fr'S [6789][{part}]$|S 10[{part}]$'
            else:
                regexp = fr'{regexp_set}[{part}]$'
            triggers = list(map(lambda x: bool(re.match(regexp, x)), events))
            triggers = events_times[triggers][0::2]
            bsl_name = events[list(map(lambda x: bool(re.match("S [1-9]", x)), events))][0]
            bsl_index = events_times[list(map(lambda x: bool(re.match("S [1-9]", x)), events))][0]
            print(f'Baseline event name is: {bsl_name}, index is {bsl_index}')
            
            result,avg = calculate_results(signal,triggers,freq,time_window_analyzed,bsl_ind=bsl_index)
            result = DataFrame(result, 
                               index = pd.Index([re.search(r'B\d+',filename_signal)[0]]))
            avg = pd.DataFrame(avg,columns=['averaged_signal'])
            try:
                with pd.ExcelWriter(f'{output}.xlsx',
                                mode='a',
                                if_sheet_exists='overlay') as writer:  
                    result.to_excel(writer, 
                                sheet_name=f'{condition}_{part}')  
                    avg.to_excel(writer, 
                             sheet_name=f'{condition}_{part}_avg')
            except FileNotFoundError:
                result.to_excel(f'{output}.xlsx', 
                                sheet_name=f'{condition}_{part}')  
                with pd.ExcelWriter(f'{output}.xlsx',
                                mode='a',
                                if_sheet_exists='overlay') as writer: 
                    avg.to_excel(writer, 
                             sheet_name=f'{condition}_{part}_avg')
            try:
                with pd.ExcelWriter(f'{output_aggr}.xlsx',
                                    mode='a',
                                    if_sheet_exists='overlay') as writer:  
                    avg.to_excel(writer, 
                                 sheet_name=f'{condition}_{part}_avg',
                                 header=False)
                    result.to_excel(writer, 
                                    sheet_name=f'{condition}_{part}',
                                    startrow=writer.sheets[f'{condition}_{part}'].max_row,
                                    header=False)
            except FileNotFoundError:
                result.to_excel(f'{output_aggr}.xlsx', 
                                sheet_name=f'{condition}_{part}')
                with pd.ExcelWriter(f'{output_aggr}.xlsx',
                                    mode='a',
                                    if_sheet_exists='overlay') as writer:  
                    avg.to_excel(writer, 
                                 sheet_name=f'{condition}_{part}_avg',
                                 header=False)
            except KeyError:
                with pd.ExcelWriter(f'{output_aggr}.xlsx',
                                                mode='a',
                                                if_sheet_exists='overlay') as writer:  
                    avg.to_excel(writer, 
                                 sheet_name=f'{condition}_{part}_avg',
                                 header=False)
                    result.to_excel(writer, 
                                    sheet_name=f'{condition}_{part}',
                                    header=False)
