"""
Takes in ECG data from the locally stored MIT-BIH database
and transforms it into 128 x 128 images of the heartbeats
on record, all centered around the R-peak.

WARNING: this file takes some time to run due to the 
size of the ecg records. I have implemented multiprocessing
to use all CPU cores to speed it up, but be warned 
CTRL + C will not end the process once it has started
due to the nature of multiprocess.
"""

import os
import multiprocessing
import wfdb
import numpy
import matplotlib.pyplot as plt

from scripts.ecg_denoise import denoise_signal
from scripts.ecg_baseline_wander import remove_baseline_wander
from scripts.detect_rpeaks import detect_rpeaks
from scripts.augment_images import beat_augment

RECORDS_DB = "data/mit_bih_records" # RECORD_DB = MIT-BIH database
IMAGES_PATH = "data/created_images"
PRE_R_WINDOW = 128  # Number of samples before the R-peak
POST_R_WINDOW = 128  # Number of samples after the R-peak
SAMPLING_RATE = 360 # Sa,pling rate of the data

# split all files ending in .dat, returns list of filenames (001, 002 etc)
records = [f.split('.')[0] for f in os.listdir(RECORDS_DB) if f.endswith('.dat')]

# Map the annotation symbol to a label
ann_translate = {
    'N': 'NOR', 
    'V': 'PVC', 
    '/': 'PAB', 
    'R': 'RBB', 
    'L': 'LBB',
    'A': 'APC', 
    '!': 'VFW', 
    'E': 'VEB'
}

def process_patient_record(patient_num):

    print(f"Processing record: {patient_num}")

    record_data = wfdb.rdsamp(f'data/mit_bih_records/{patient_num}')
    annotation = wfdb.rdann(f'data/mit_bih_records/{patient_num}', 'atr')

    ecg_data = record_data[0].transpose()

    # Two lead types to choose from.
    # This will be MLII for all records but 102, 104,
    # who's data is clearer on second lead (both on pacemakers)
    if patient_num == 114 or 207:
        ecg_data = ecg_data[1]
    else:
        ecg_data = ecg_data[0]

    ecg_data = denoise_signal(ecg_data) # Denoise the data
    ecg_data = remove_baseline_wander(ecg_data) # Remove baseline wander
    rpeaks_indices = detect_rpeaks(ecg_data, SAMPLING_RATE) # identify r peaks



    # This part of the code identifies noisy periods to be skipped later.
    noisy_periods = []  # List to store noisy periods
    SKIP_NOISY_PERIOD = False

    # Loop through all annotations and identify the noisy periods based on subtypes
    for i, annotation_sample in enumerate(annotation.sample):
        annotation_symbol = annotation.symbol[i]
        subtype = annotation.subtype[i]

        # Check for noisy periods based on subtype
        if (annotation_symbol == '~') and (subtype in [1, 2, 3]):  # Start of noisy period

            # Only mark the start of the noisy period if we aren't already in one
            if not SKIP_NOISY_PERIOD:

                SKIP_NOISY_PERIOD = True
                start_sample = annotation_sample

        elif (annotation_symbol == '~') and (subtype == 0):  # Clean period (end of noisy period)

            # Only mark the end of the noisy period if we are currently in one
            if SKIP_NOISY_PERIOD:

                SKIP_NOISY_PERIOD = False
                noisy_periods.append((start_sample, annotation_sample))

        elif (annotation_symbol == '~') and (subtype == -1):
            continue



    for idx, rpeak_idx in enumerate(rpeaks_indices['rpeaks']):

        # Check if the current R-peak is within a noisy period
        IN_NOISY_PERIOD = False
        for start_sample, end_sample in noisy_periods:
            if start_sample <= rpeak_idx <= end_sample:
                IN_NOISY_PERIOD = True
                break

        # If we're in a noisy period, skip the heartbeat
        if IN_NOISY_PERIOD:
            print(f' Patient {patient_num}: {idx} SKIPPED - NOISY DATA.')
            continue


        # Find the annotation closest to the R-peak
        closest_annotation_idx = numpy.argmin(numpy.abs(annotation.sample - rpeak_idx))
        closest_annotation = annotation.symbol[closest_annotation_idx]
        full_ann = ann_translate.get(closest_annotation, 'OTHER') # use ann map to translate

        # Define the range for the current beat (centered around the R-peak)
        start_idx = max(rpeak_idx - PRE_R_WINDOW, 0)
        end_idx = min(rpeak_idx + POST_R_WINDOW, len(ecg_data))
        beat = ecg_data[start_idx:end_idx] # Extract the heartbeat segment

        # create the directory if it doesnt exist for future users
        os.makedirs(f'{IMAGES_PATH}/{full_ann}', exist_ok=True)

        # Create the plot
        # 3.31, 3.04 for 256 x 256 sized image
        fig, ax = plt.subplots(figsize=(1.66, 1.67), dpi=100)
        ax.plot(beat, color='black')
        ax.set_xlim(0, len(beat))
        ax.axis('off')

        # Save the figure in the created images folder for viewing
        fig.savefig(
            f'{IMAGES_PATH}/{full_ann}/{patient_num}_{idx}.png',
            dpi=100,
            bbox_inches=None,
            pad_inches=0
        )
        print(f' Patient {patient_num}: HEARTBEAT {idx} - IMAGE CREATED.')

        # creates extra augmented images of any arrythmias
        beat_augment(beat, closest_annotation, idx, patient_num)


if __name__ == '__main__':
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(process_patient_record, records)
