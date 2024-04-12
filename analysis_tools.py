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

def get_stimlist(filename):
    """Opens file (text, name of file) in vmrk format, reads events and return list
    with tuples [(name_of_stimulus,stimulus_data_point)]"""
    with open(filename,'r') as file:
        data = file.read()
    stimlist=[n for n in data.split('\n') if 'Stimulus' in n]
    return [(n.split(',')[1],int(n.split(',')[2])) for n in stimlist]

def get_stimulus_array(stimlist):
    """Export array of stimuli data poitns from stimlist"""
    return np.array(stimlist)[:,1].astype(int)

def get_stimulus_names(stimlist):
    """Export array of stimuli names from stimlist"""
    return np.array(stimlist)[:,0]

def resample_events(stim_array,resample_coef=50):
    """Resample events using given coefficient, accepts 
    returns numpy array with rounded resampled stimulus."""
    return np.round(stim_array/resample_coef).astype(int)


def analysis(filename_signal,filename_events,output,freq=FREQ,originf=ORIGINF,cutoff=2,correction=1000000):
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
    print('calculating results...')
    result = []
    for name,datapoint in zip(stimnames,new_stims):
        tmp_bsle=np.mean(filtered[datapoint-(3*FREQ):datapoint])*correction
        tmp_mean30=np.mean(filtered[datapoint:datapoint+(30*FREQ)])*correction
        tmp_mean15=np.mean(filtered[datapoint:datapoint+(15*FREQ)])*correction
        tmp_mean12=np.mean(filtered[datapoint:datapoint+(12*FREQ)])*correction

        tmp_bsle=np.min(filtered[datapoint-(3*FREQ):datapoint])*correction
        tmp_min30=np.min(filtered[datapoint:datapoint+(30*FREQ)])*correction
        tmp_min15=np.min(filtered[datapoint:datapoint+(15*FREQ)])*correction
        tmp_min12=np.min(filtered[datapoint:datapoint+(12*FREQ)])*correction

        tmp_bsle=np.max(filtered[datapoint-(3*FREQ):datapoint])*correction
        tmp_max30=np.max(filtered[datapoint:datapoint+(30*FREQ)])*correction
        tmp_max15=np.max(filtered[datapoint:datapoint+(15*FREQ)])*correction
        tmp_max12=np.max(filtered[datapoint:datapoint+(12*FREQ)])*correction

        for suffix,suf_value in zip(['_baseline','_mean30','_mean15','_mean12','_min30',
                                     '_min15','_min12','_max30','_max15','_max12'],
                                    [tmp_bsle,tmp_mean30,tmp_mean15,tmp_mean12,tmp_min30,
                                     tmp_min15,tmp_min12,tmp_max30,tmp_max15,tmp_max12]):
            value_name = f'{name}{suffix}'
            try:
                if value_name in np.array(result)[:,0]:
                    for n in range(2,10):
                        if f'{value_name}_{n}' in np.array(result)[:,0]:
                            pass
                        else:
                            result.append((f'{value_name}_{n}',suf_value))
                            break
                else:
                    result.append((value_name,suf_value))
            except:
                result.append((value_name,suf_value))
    result = DataFrame(result)
    result.to_excel(f'{output}.xlsx')
    return result

def create_stimlist_file_from_data(filename):
    data = bioread.read_file(filename)
    x = lambda a : a>0
    result = []
    print('creating stimlist...')
    for idx in range(2,len(data.channels[0].data)):
        uno = [str(int(x(n.data[idx]))) for n in list(reversed(data.channels[1:]))]
        duo = [str(int(x(n.data[idx-1]))) for n in list(reversed(data.channels[1:]))]
        uno = int(''.join(uno),2)
        duo = int(''.join(duo),2)
        if (uno>1 and duo<1):
            code = code_generator(uno,'S')
            result.append((code,idx))
    print('done!')
    return result

def code_generator(n,firstLetter = 'S'):
    return '{}{: 3d}'.format(firstLetter,n)

def convert_acq_mne(filename='analiza_follow up\JM22a_B140_TP6_evt.acq'):
    data = bioread.read_file(filename)
    n_channels = len(data.channels)
    sampling_freq = int(data.channels[0].samples_per_second)  # in Hertz
    channel_names = [n.name for n in data.channels]

    times = int(data.channels[0].samples_per_second)
    data = np.array([n.data for n in data.channels])
    info = mne.create_info(
        ch_names=channel_names, ch_types=['misc']*n_channels, sfreq=sampling_freq
    )
    print(info)    
    raw = mne.io.RawArray(data, info)
    return raw

def create_placeholder(reference='preprocessed_fup\JM22A_B011_TP6.xlsx', output='result.xlsx'):
    if output not in listdir():
        tmp = read_excel(filename)
        columns = tmp[0].values
        DataFrame(columns = columns).to_excel(output)
create_placeholder()

def add_result(file,code):
    tmp = read_excel(file,index_col=0)
    columns = tmp[0].values
    transposed = tmp.T.drop(0)
    transposed.columns = columns
    transposed.index = Index([code])
    return transposed

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
                                 time_window_preceding=3*100):
    partSig = signal[point:(point+time_window)]
    precedSig = signal[point-time_window_preceding:point]
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
    return medfiltered,new_stims,stimnames

def plot_bv_medfiltered_signal(filename_signal,
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
    plt.plot([n for n in range(0,len(resampled),1)],resampled)
    plt.plot([n for n in range(0,len(resampled),1)],filtered)
    plt.plot([n for n in range(0,len(resampled),1)],medfiltered)
    for stim in new_stims:
        plt.axvspan(stim-100, stim+100, facecolor='r',alpha=0.5)
    return medfiltered,new_stims,stimnames

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

def calculate_results(signal,triggers,freq,time_window_analyzed):
    plc = np.zeros([len(triggers),freq*time_window_analyzed])
    for n in range(len(triggers)):
        plc[n] = signal[triggers[n]:(triggers[n]+freq*time_window_analyzed)]
    avg = np.mean(plc,axis=0)
    return ({'mean_slope' : np.mean([get_slope(plc[n]) for n in range(len(plc))]),
            'mean_normalized_signal_change' : np.mean([get_normalized_signal_change(signal,n)\
                                                                 for n in triggers]),
            'mean_sig_sign_alteration' : np.mean([get_ssa(plc[n]) for n in range(len(plc))]),
            'mean_slope_2pt' : np.mean([get_slope_2_points(plc[n]) for n in range(len(plc))]),
           'mean_signal' : np.mean([np.mean(plc[n]) for n in range(len(plc))]),
           'mean_max_signal' : np.mean([np.max(plc[n]) for n in range(len(plc))]),
           'mean_min_signal' : np.mean([np.min(plc[n]) for n in range(len(plc))])},
           avg)
            
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
