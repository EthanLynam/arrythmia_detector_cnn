import wfdb 
import numpy
import os
import matplotlib.pyplot as plt
from biosppy.signals import ecg
from ecgdetectors import Detectors

from denoise import denoise_signal
from baseline_wander import remove_baseline_wander

# record_db = MIT-BIH database
record_db = "../../MIT-BIH-DB"

# split all files ending in .dat, returns list of filenames (001, 002 etc)
records = [f.split('.')[0] for f in os.listdir(record_db) if f.endswith('.dat')]

sampling_rate = 360
detectors = Detectors(sampling_rate)

# Loop through each patient_nums record. Each record has a set of 3 files:
# 001.dat, 001.hea, 001.atr, which are contained inside the locally stored MIT-BIH-DB
for patient_num in records:
    print(f"Processing record: {patient_num}")

    record_data = wfdb.rdsamp(f'../../MIT-BIH-DB/{patient_num}')
    annotation = wfdb.rdann(f'../../MIT-BIH-DB/{patient_num}', 'atr')
    
    # Get the ECG values from the file and choose the ecg_data to process 
    ecg_data = record_data[0].transpose()
    ecg_data = ecg_data[0]  # This will be MLII for all records but 102, 104, 114 which will be V5

    # Denoise the data
    ecg_data = denoise_signal(ecg_data)

    # Remove baseline wander
    ecg_data = remove_baseline_wander(ecg_data)

    # method with biospyy.signals ecg
    rpeaks_indices = ecg.ecg(signal=ecg_data, sampling_rate=360, show=False) #identify ecg features, including r peak

    # used for how much of the ECG data will be contained in an image
    pre_R_window = 100  # Number of samples before the R-peak
    post_R_window = 100  # Number of samples after the R-peak

    for idx, rpeak_idx in enumerate(rpeaks_indices['rpeaks']):
        # Find the annotation closest to the R-peak
        closest_annotation_idx = numpy.argmin(numpy.abs(annotation.sample - rpeak_idx))
        closest_annotation = annotation.symbol[closest_annotation_idx]

        # Define the range for the current beat (centered around the R-peak)
        start_idx = max(rpeak_idx - pre_R_window, 0)
        end_idx = min(rpeak_idx + post_R_window, len(ecg_data))

        # Extract the heartbeat segment
        beat = ecg_data[start_idx:end_idx]

        # Create the plot
        fig, ax = plt.subplots(figsize=(1.66, 1.38), dpi=100)
        ax.plot(beat, color='black')
        ax.set_title(closest_annotation)
        ax.set_xlim(0, len(beat))
        ax.axis('off')

        # Map the annotation symbol to a label
        annotation_map = {
            'N': 'NOR', 
            'V': 'PVC', 
            '/': 'PAB', 
            'R': 'RBB', 
            'L': 'LBB',
            'A': 'APC', 
            '!': 'VFW', 
            'E': 'VEB'
        }
        full_name = annotation_map.get(closest_annotation, 'OTHER')

        # Save the figure
        fig.savefig(
            f'../../created-images/{full_name}/{patient_num}_{idx}.png',
            bbox_inches='tight',
            pad_inches=0
        )
        print(f'    Patient {patient_num}: image of heartbeat {idx} created around index {rpeak_idx}.')

        # Close the plot to free up resources
        plt.close(fig)
