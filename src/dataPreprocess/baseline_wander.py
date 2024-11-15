import numpy as np
from scipy.signal import butter, filtfilt

def remove_baseline_wander(ecg_data):
    sampFrequency = 360

    normal_cutoff = 0.5 / (0.5 * sampFrequency)

    b, a = butter(1, normal_cutoff, btype='high', analog=False)

    return filtfilt(b, a, ecg_data)
