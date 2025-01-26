"""
This module takes in ECG data from the locally stored MIT-BIH database
and transforms it into 128 x 128 images of each individual beat, 
centered around the R-peak of each heartbeat. It uses 
denoising and baseline wander functions from seperate modules to 
make the data clearer, for easier identification of R-peak location anf
heartbeat type.

"""

import os
import wfdb
import numpy
import matplotlib.pyplot as plt
from biosppy.signals import ecg

from denoise import denoise_signal
from baseline_wander import remove_baseline_wander
from beat_augment import beat_augment

RECORD_DB = "../../MIT-BIH-DB" # RECORD_DB = MIT-BIH database
PRE_R_WINDOW = 128  # Number of samples before the R-peak
POST_R_WINDOW = 128  # Number of samples after the R-peak
SAMPLING_RATE = 360 # Sa,pling rate of the data

# split all files ending in .dat, returns list of filenames (001, 002 etc)
records = [f.split('.')[0] for f in os.listdir(RECORD_DB) if f.endswith('.dat')]

# Loop through each patient_nums record. Each record has a set of 3 files:
# 001.dat, 001.hea, 001.atr, which are contained inside the locally stored MIT-BIH-DB
for patient_num in records:
    print(f"Processing record: {patient_num}")

    record_data = wfdb.rdsamp(f'../../MIT-BIH-DB/{patient_num}')
    annotation = wfdb.rdann(f'../../MIT-BIH-DB/{patient_num}', 'atr')

    ecg_data = record_data[0].transpose()

    # Get the ECG values from the file and choose the ecg_data to process
    # This will be MLII for all records but 102, 104,
    # which will be V5 (these patients are on a pacemaker)
    if patient_num == 114:
        ecg_data = ecg_data[1]  # MLII is contained in the second lead for record 114
    else:
        ecg_data = ecg_data[0]

    ecg_data = denoise_signal(ecg_data) # Denoise the data
    ecg_data = remove_baseline_wander(ecg_data) # Remove baseline wander

    # identify ecg features, including r peak
    rpeaks_indices = ecg.ecg(
        signal=ecg_data,
        sampling_rate=SAMPLING_RATE,
        show=False
        )


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
                print(f" Patient {patient_num}: Noisy period starts at sample {annotation_sample}.")

        elif (annotation_symbol == '~') and (subtype == 0):  # Clean period (end of noisy period)

            # Only mark the end of the noisy period if we are currently in one
            if SKIP_NOISY_PERIOD:

                SKIP_NOISY_PERIOD = False
                noisy_periods.append((start_sample, annotation_sample))
                print(f" Patient {patient_num}: Noisy period ends at sample {annotation_sample}.")

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
            print(f' Patient {patient_num}: image of heartbeat {idx} skipped as data is too noisy.')
            continue


        # Find the annotation closest to the R-peak
        closest_annotation_idx = numpy.argmin(numpy.abs(annotation.sample - rpeak_idx))
        closest_annotation = annotation.symbol[closest_annotation_idx]

        # Define the range for the current beat (centered around the R-peak)
        start_idx = max(rpeak_idx - PRE_R_WINDOW, 0)
        end_idx = min(rpeak_idx + POST_R_WINDOW, len(ecg_data))

        # Extract the heartbeat segment
        beat = ecg_data[start_idx:end_idx]

        # Create the plot
        # 3.31, 3.04 for 256 x 256 sized image
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

        # construct the directory path
        directory = f'../../created-images/{full_name}'

        # create the directory if it doesnt exist for future users
        os.makedirs(directory, exist_ok=True)

        # Save the figure in the created images folder for viewing
        fig.savefig(
            f'{directory}/{patient_num}_{idx}.png',
            bbox_inches='tight',
            pad_inches=0
        )
        print(f' Patient {patient_num}: image of heartbeat {idx} created around index {rpeak_idx}.')

        # creates extra augmented images of any arrythmias
        beat_augment(beat, closest_annotation, idx, patient_num)

        # Close the plot to free up resources
        plt.close(fig)
