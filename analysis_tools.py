from mne.io import read_raw_brainvision
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt,resample, butter,filtfilt
from pandas import DataFrame

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

def signal_alignment_correction(stim_array,correction_value):
    """
    Correct datapoints array of stimulus
    """
    return stim_array - correction_value

def resample_events(stim_array,resample_coef=50):
    """Resample events using given coefficient, accepts 
    returns numpy array with rounded resampled stimulus."""
    return np.round(stim_array/resample_coef).astype(int)

def butter_lowpass_filter(data, cutoff=2, fs=100, order=2):
    """Accepts array with signal (data). Lowpass Butter filter, filters given signal."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def analysis(filename_signal,filename_events,output,FREQ=FREQ):
    print(f'opening {filename_signal}...')
    data = read_raw_brainvision(filename_signal)
    data = data.get_data().flatten()
    stimlist = get_stimlist(filename_events)
    where = stimlist[0][1]-15000
    data = data[where:]
    print('resampling...')
    data = resample(data,round(len(data)/50))
    print('filtering...')
    filtered = butter_lowpass_filter(data)
    medfiltered=medfilt(filtered)
    stimarr = get_stimulus_array(stimlist)
    aligned = signal_alignment_correction(stimarr,where)
    new_stims= resample_events(aligned,50)
    stimnames = get_stimulus_names(stimlist)
    print('calculating results...')
    result = []
    for name,datapoint in zip(stimnames,new_stims):
        tmp_bsle=np.mean(filtered[datapoint-(3*FREQ):datapoint])*1000000
        tmp_mean30=np.mean(filtered[datapoint:datapoint+(30*FREQ)])*1000000
        tmp_mean15=np.mean(filtered[datapoint:datapoint+(15*FREQ)])*1000000
        tmp_mean12=np.mean(filtered[datapoint:datapoint+(12*FREQ)])*1000000

        tmp_bsle=np.min(filtered[datapoint-(3*FREQ):datapoint])*1000000
        tmp_min30=np.min(filtered[datapoint:datapoint+(30*FREQ)])*1000000
        tmp_min15=np.min(filtered[datapoint:datapoint+(15*FREQ)])*1000000
        tmp_min12=np.min(filtered[datapoint:datapoint+(12*FREQ)])*1000000

        tmp_bsle=np.max(filtered[datapoint-(3*FREQ):datapoint])*1000000
        tmp_max30=np.max(filtered[datapoint:datapoint+(30*FREQ)])*1000000
        tmp_max15=np.max(filtered[datapoint:datapoint+(15*FREQ)])*1000000
        tmp_max12=np.max(filtered[datapoint:datapoint+(12*FREQ)])*1000000

        result.append((f'{name}_baseline',tmp_bsle))
        result.append((f'{name}_mean30',tmp_mean30))
        result.append((f'{name}_mean15',tmp_mean15))
        result.append((f'{name}_mean12',tmp_mean12))
        result.append((f'{name}_min30',tmp_min30))
        result.append((f'{name}_min15',tmp_min15))
        result.append((f'{name}_min12',tmp_min12))
        result.append((f'{name}_max30',tmp_max30))
        result.append((f'{name}_max15',tmp_max15))
        result.append((f'{name}_max12',tmp_max12))
    result = DataFrame(result)
    result.to_excel(f'{output}.xlsx')
    return result