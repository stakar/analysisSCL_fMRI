import mne
import bioread
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from pandas import DataFrame

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
    filtered = mne.filter.filter_data(resampled,FREQ,None,cutoff)
    medfiltered=medfilt(filtered,int((FREQ/2)-1))
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

        result.append((f'{name}_baseline'.replace(' ',''),tmp_bsle))
        result.append((f'{name}_mean30'.replace(' ',''),tmp_mean30))
        result.append((f'{name}_mean15'.replace(' ',''),tmp_mean15))
        result.append((f'{name}_mean12'.replace(' ',''),tmp_mean12))
        result.append((f'{name}_min30'.replace(' ',''),tmp_min30))
        result.append((f'{name}_min15'.replace(' ',''),tmp_min15))
        result.append((f'{name}_min12'.replace(' ',''),tmp_min12))
        result.append((f'{name}_max30'.replace(' ',''),tmp_max30))
        result.append((f'{name}_max15'.replace(' ',''),tmp_max15))
        result.append((f'{name}_max12'.replace(' ',''),tmp_max12))
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

def analysis_acknowledge(filename_signal, filename_events, output, freq=FREQ, originf=ORIGINF, cutoff=2, correction=1):
    """
    Conduct analysis- open file, preprocess data (resample, filter with lowpass filter, 
    conduct median smoothing, get stimuli time points and calculate results.
    """
    print(f'opening {filename_signal}...')
    data = convert_acq_mne('analiza_follow up\JM22A_B052_TP6.acq')
    resampled = data.copy().resample(sfreq=freq).get_data()[0]
    filtered = mne.filter.filter_data(resampled,FREQ,None,cutoff)
    medfiltered=medfilt(filtered,int((FREQ/2)-1))
    stimlist = create_stimlist_file_from_data(filename_events)
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

        result.append((f'{name}_baseline'.replace(' ',''),tmp_bsle))
        result.append((f'{name}_mean30'.replace(' ',''),tmp_mean30))
        result.append((f'{name}_mean15'.replace(' ',''),tmp_mean15))
        result.append((f'{name}_mean12'.replace(' ',''),tmp_mean12))
        result.append((f'{name}_min30'.replace(' ',''),tmp_min30))
        result.append((f'{name}_min15'.replace(' ',''),tmp_min15))
        result.append((f'{name}_min12'.replace(' ',''),tmp_min12))
        result.append((f'{name}_max30'.replace(' ',''),tmp_max30))
        result.append((f'{name}_max15'.replace(' ',''),tmp_max15))
        result.append((f'{name}_max12'.replace(' ',''),tmp_max12))
    result = DataFrame(result)
    result.to_excel(f'{output}.xlsx')
    return result
