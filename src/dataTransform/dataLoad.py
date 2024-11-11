import ecg_plot

import wfdb 
import numpy as np
from scipy import signal
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from biosppy.signals import ecg
from ecgdetectors import Detectors

from PIL import Image
from denoise import denoise_signal
#from baseline-wander import remove_baseline_wander

# record_db = MIT-BIH database
record_db = "../../MIT-BIH-DB"

# split all files ending in .dat, returns list of filenames (001, 002 etc)
records = [f.split('.')[0] for f in os.listdir(record_db) if f.endswith('.dat')]

sampling_rate = 360
detectors = Detectors(sampling_rate)

# Loop through each patients record. Each record has a set of 3 files:
# 001.dat, 001.hea, 001.atr, which are contained inside the locally stored MIT-BIH-DB
for patient in records:
    print(f"Processing record: {patient}")
    record = wfdb.rdsamp(f'../../MIT-BIH-DB/{patient}')
    annotation = wfdb.rdann(f'../../MIT-BIH-DB/{patient}', 'atr')
    
    # Get the ECG values from the file and choose the channel to process (e.g., channel 0)
    ecg_data = record[0].transpose()
    channel = ecg_data[0]  # Assuming you only want the first channel, adjust index as needed

    # Denoise the data
    channel = denoise_signal(channel)

    # Find R-peaks in the ECG data
    #method1
    rpeaks_indices = detectors.pan_tompkins_detector(channel)
    rpeaks = np.zeros_like(channel, dtype='float')
    rpeaks[rpeaks_indices] = 1.0


    #method2: in use
    out = ecg.ecg(signal=channel, sampling_rate=360, show=False)
    rpeaks = np.zeros_like(channel, dtype='float')
    rpeaks[out['rpeaks']] = 1.0

    # Define the window size around the R-peak for each beat
    pre_R_window = 100  # Number of samples before the R-peak
    post_R_window = 100  # Number of samples after the R-peak

    for idx, rpeak_idx in enumerate(out['rpeaks']):
        # Find the annotation closest to the R-peak
        closest_annotation_idx = np.argmin(np.abs(annotation.sample - rpeak_idx))
        closest_annotation = annotation.symbol[closest_annotation_idx]

        # Define the range for the current beat (centered around the R-peak)
        start_idx = max(rpeak_idx - pre_R_window, 0)
        end_idx = min(rpeak_idx + post_R_window, len(channel))

        # Extract the heartbeat segment
        beat = channel[start_idx:end_idx]

        # Create the plot
        fig, ax = plt.subplots(figsize=(1.66, 1.38), dpi=100)
        ax.plot(beat, color='black')
        ax.set_title(closest_annotation)
        ax.set_xlim(0, len(beat))
        ax.axis('off')

        # Save the figure
        fig.savefig(
            f'../../heartbeat_images/{patient}_{idx}.png',
            bbox_inches='tight',
            pad_inches=0
        )
        print(f'    Patient {patient}: image of heartbeat {idx} created.')

        # Close the plot to free up resources
        plt.close(fig)

