"""
This module taes in ecg data and removes baseline wander,
that being the data 'wandering' and moving off course. 
It utilises a butterworth filter and the linear
filtfilt function to return the data, now straightened
up with no loss of data shape.
"""

from scipy.signal import butter, filtfilt

def remove_baseline_wander(ecg_data):
    """removes baseline wander from given signal"""

    samp_frequency = 360

    normal_cutoff = 0.5 / (0.5 * samp_frequency)

    b, a = butter(1, normal_cutoff, btype='high', analog=False)

    return filtfilt(b, a, ecg_data)
